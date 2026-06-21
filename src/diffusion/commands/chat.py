"""`diffusion chat` — interactive REPL with live in-terminal denoising previews.

Loads the model once, then for each prompt streams the image as it emerges from
noise (Kitty graphics protocol, e.g. Ghostty), and saves the final image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from diffusion.utils.console import console


@dataclass
class _Settings:
    steps: int
    width: int
    height: int
    seed: int | None
    negative_prompt: str | None
    outdir: Path


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
    rows: int,
    outdir: Path,
) -> None:
    _quiet_libraries()

    from diffusion.core import generate
    from diffusion.core.generate import write_sidecar
    from diffusion.utils.terminal_image import detect_protocol

    protocol = detect_protocol()
    if protocol == "none":
        console.print(
            "[yellow]No inline-image terminal detected.[/yellow] Previews are disabled; "
            "images will still be saved. (Force with DIFFUSION_FORCE_KITTY=1.)"
        )

    console.print(f"Loading [bold]{repo_id}[/bold] …")
    pipe, kind, plan = generate.load_pipeline(
        repo_id, device_override=device, dtype_override=dtype, low_mem=low_mem
    )
    console.print(
        f"[green]✓[/green] [bold]{repo_id}[/bold] ([cyan]{kind}[/cyan]) ready on "
        f"[magenta]{plan.device}[/magenta]/{plan.dtype}."
    )
    _print_help()

    settings = _Settings(
        steps=steps, width=width, height=height, seed=seed,
        negative_prompt=negative_prompt, outdir=outdir,
    )
    counter = 0

    while True:
        try:
            line = console.input("\n[bold cyan]›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return

        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            console.print("Bye.")
            return
        if line == "/help":
            _print_help()
            continue
        if line.startswith("/"):
            _handle_command(line, settings)
            continue

        counter += 1
        _generate_one(
            generate, write_sidecar, pipe, kind, plan, line, settings,
            protocol=protocol, rows=rows, repo_id=repo_id, index=counter,
        )


def _generate_one(
    generate, write_sidecar, pipe, kind, plan, prompt, settings, *,
    protocol, rows, repo_id, index,
):
    from diffusion.utils.terminal_image import KittyRenderer

    renderer = KittyRenderer(rows=rows) if protocol == "kitty" else None
    total = settings.steps

    def on_preview(step_index, image):
        renderer.show(image, status=_status_bar(step_index + 1, total))

    console.print(
        f"[dim]Generating {settings.width}×{settings.height}, {settings.steps} steps …[/dim]"
    )
    start = time.perf_counter()
    image = generate.run_inference(
        pipe, kind, plan, prompt=prompt, negative_prompt=settings.negative_prompt,
        steps=settings.steps, width=settings.width, height=settings.height,
        seed=settings.seed, low_mem=False,
        on_preview=on_preview if renderer else None,
    )
    elapsed = time.perf_counter() - start

    if renderer is not None:
        renderer.show(image, status=_status_bar(total, total, done=True, secs=elapsed))
        renderer.finish()

    settings.outdir.mkdir(parents=True, exist_ok=True)
    output = settings.outdir / f"chat_{index:03d}.png"
    image.save(output)
    write_sidecar(
        output, repo_id=repo_id, kind=kind, prompt=prompt,
        negative_prompt=settings.negative_prompt, steps=settings.steps,
        width=settings.width, height=settings.height, seed=settings.seed,
        device=plan.device, dtype=plan.dtype, elapsed_s=round(elapsed, 2),
    )
    console.print(f"[green]✓[/green] {output} ({elapsed:.1f}s)")


def _handle_command(line: str, settings: _Settings) -> None:
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
        elif cmd == "/size":
            w, h = arg.lower().split("x")
            settings.width, settings.height = int(w), int(h)
        else:
            console.print(f"[yellow]Unknown command:[/yellow] {cmd}. Type /help.")
            return
    except ValueError:
        console.print(f"[red]Bad argument for {cmd}:[/red] '{arg}'")
        return
    console.print(
        f"[dim]steps={settings.steps} size={settings.width}×{settings.height} "
        f"seed={settings.seed} neg={settings.negative_prompt!r}[/dim]"
    )


def _status_bar(step: int, total: int, *, done: bool = False, secs: float = 0.0, width: int = 24) -> str:
    step = min(step, total)
    filled = round(width * step / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    if done:
        return f"  ✓ {bar} {step}/{total}  ({secs:.1f}s)"
    return f"  {bar} {step}/{total}  denoising…"


def _quiet_libraries() -> None:
    """Suppress torch/transformers/diffusers log spam and progress bars."""
    import logging
    import os
    import warnings

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "critical")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore")
    for name in ("transformers", "diffusers", "accelerate", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # Disable the libraries' tqdm progress bars (separate from logging).
    for mod in ("diffusers.utils.logging", "transformers.utils.logging"):
        try:
            __import__(mod, fromlist=["disable_progress_bar"]).disable_progress_bar()
        except Exception:
            pass


def _print_help() -> None:
    console.print(
        "[dim]Type a prompt to generate. Commands: /steps N · /size WxH · "
        "/seed N|random · /neg <text> · /help · /exit[/dim]"
    )
