"""`diffusion chat` — interactive REPL with live in-terminal denoising previews.

Loads the model once, then for each prompt streams the image as it emerges from
noise (Kitty graphics protocol, e.g. Ghostty), and saves the final image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diffusion.utils import ui
from diffusion.utils.console import console, quiet_diffusion_libraries

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _Settings:
    steps: int
    width: int
    height: int
    seed: int | None
    negative_prompt: str | None
    outdir: Path
    guidance_scale: float | None = None
    force_size: bool = False
    low_mem: bool = False
    sampler: str = "default"  # diffusers scheduler class name currently on the pipe


def run_chat(
    repo_id: str,
    *,
    steps: int,
    width: int,
    height: int,
    seed: int | None,
    negative_prompt: str | None,
    device: str | None,
    dtype: str | None,
    low_mem: bool,
    force_size: bool,
    rows: int,
    outdir: Path,
    sampler: str | None = None,
) -> None:
    """Run the interactive chat REPL with live in-terminal denoising previews.

    Loads the model once, then loops reading prompts; each prompt streams the
    image as it emerges from noise and saves the final result to ``outdir``.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id of the model to run.
    steps : int
        Number of denoising steps per generation.
    width, height : int
        Output image dimensions in pixels.
    seed : int or None
        Random seed for reproducibility, or None for a random seed.
    negative_prompt : str or None
        Text describing what to avoid, or None.
    device : str or None
        Device override (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``), or None to autodetect.
    dtype : str or None
        Torch dtype override, or None to autodetect.
    low_mem : bool
        If True, enable memory-saving optimizations (e.g. CPU offload).
    force_size : bool
        If True, bypass the pre-flight memory safety check for the requested size.
    rows : int
        Number of terminal rows to use for the inline preview.
    outdir : Path
        Directory where generated images are saved.
    sampler : str or None
        Initial sampler/scheduler name (e.g. ``"euler"``, ``"dpm++"``), or None to
        keep the model's default. Switch it live with ``/sampler``.
    """
    _quiet_libraries()

    from diffusion.core import generate, registry, samplers
    from diffusion.core.generate import write_sidecar
    from diffusion.utils import prompt as prompt_input
    from diffusion.utils.terminal_image import detect_protocol

    repo_id = registry.resolve_repo(repo_id)
    protocol = detect_protocol()
    if protocol == "none":
        console.print(
            "[yellow]No inline-image terminal detected.[/yellow] Previews are disabled; "
            "images will still be saved. (Force with DIFFUSION_FORCE_KITTY=1.)"
        )

    with ui.loading_status(f"Loading {repo_id} …"):
        pipe, family, plan = generate.load_pipeline(
            repo_id,
            device_override=device,
            dtype_override=dtype,
            low_mem=low_mem,
            sampler=sampler,
        )
    console.print(ui.model_ready_panel(repo_id, family, plan.device, plan.dtype))
    console.print(ui.help_panel())

    session = prompt_input.build_session()
    settings = _Settings(
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        negative_prompt=negative_prompt,
        outdir=outdir,
        force_size=force_size,
        low_mem=low_mem,
        sampler=samplers.current_sampler(pipe),
    )
    counter = 0

    while True:
        try:
            line = prompt_input.read_prompt(session)
        except (EOFError, KeyboardInterrupt):
            _close()
            return

        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            _close()
            return
        if line == "/help":
            console.print(ui.help_panel())
            continue
        if line.startswith("/"):
            _handle_command(line, settings, pipe)
            continue

        counter += 1
        _generate_one(
            generate,
            write_sidecar,
            pipe,
            family,
            plan,
            line,
            settings,
            protocol=protocol,
            rows=rows,
            repo_id=repo_id,
            index=counter,
        )


def _generate_one(
    generate,
    write_sidecar,
    pipe,
    family,
    plan,
    prompt,
    settings,
    *,
    protocol,
    rows,
    repo_id,
    index,
):
    console.print(
        f"[dim]Generating {settings.width}×{settings.height}, {settings.steps} steps …[/dim]"
    )
    image, elapsed = ui.run_with_preview(
        generate,
        pipe,
        family,
        plan,
        prompt=prompt,
        negative_prompt=settings.negative_prompt,
        steps=settings.steps,
        width=settings.width,
        height=settings.height,
        seed=settings.seed,
        low_mem=settings.low_mem,
        force_size=settings.force_size,
        protocol=protocol,
        rows=rows,
        guidance_scale=settings.guidance_scale,
    )

    settings.outdir.mkdir(parents=True, exist_ok=True)
    output = settings.outdir / f"chat_{index:03d}.png"
    image.save(output)
    write_sidecar(
        output,
        repo_id=repo_id,
        family=family,
        prompt=prompt,
        negative_prompt=settings.negative_prompt,
        steps=settings.steps,
        width=settings.width,
        height=settings.height,
        seed=settings.seed,
        sampler=settings.sampler,
        device=plan.device,
        dtype=plan.dtype,
        elapsed_s=round(elapsed, 2),
    )
    console.print(
        ui.result_panel(
            output,
            settings.seed,
            settings.steps,
            f"{settings.width}×{settings.height}",
            elapsed,
        )
    )


def _close() -> None:
    """Print a soft farewell and leave the cursor on a clean line."""
    console.print("[dim]✦ session ended[/dim]")


def _handle_command(line: str, settings: _Settings, pipe) -> None:
    from diffusion.core import memory, samplers
    from diffusion.utils.errors import DiffusionError

    parts = line.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""
    try:
        if cmd == "/steps":
            settings.steps = max(1, int(arg))
        elif cmd == "/seed":
            settings.seed = None if arg in ("", "none", "random") else int(arg)
        elif cmd == "/neg":
            settings.negative_prompt = arg or None
        elif cmd == "/cfg":
            settings.guidance_scale = None if arg in ("", "none", "default") else float(arg)
        elif cmd == "/size":
            w_str, h_str = arg.lower().split("x")
            width, height = int(w_str), int(h_str)
            memory.validate_dimensions(width, height)  # reject bad sizes before they stick
            settings.width, settings.height = width, height
        elif cmd == "/sampler":
            if not arg:
                console.print(
                    f"[dim]current:[/dim] {settings.sampler}   "
                    f"[dim]available:[/dim] {', '.join(samplers.available_samplers())}"
                )
                return
            samplers.apply_sampler(pipe, arg)  # raises InvalidSamplerError if unknown
            settings.sampler = samplers.current_sampler(pipe)
        else:
            console.print(f"[yellow]Unknown command:[/yellow] {cmd}. Type /help.")
            return
    except ValueError:
        console.print(f"[red]Bad argument for {cmd}:[/red] '{arg}'")
        return
    except DiffusionError as exc:
        console.print(f"[red]{exc.message}[/red]")
        if exc.hint:
            console.print(f"[dim]{exc.hint}[/dim]")
        return
    console.print(ui.settings_table(settings))


def _quiet_libraries() -> None:
    """Suppress torch/transformers/diffusers log spam and progress bars."""
    quiet_diffusion_libraries()
