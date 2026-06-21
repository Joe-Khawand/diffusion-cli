"""`diffusion run` — generate an image from a prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_generate(
    *,
    repo_id: str,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    width: int,
    height: int,
    output: Path,
    seed: int | None,
    device: str | None,
    dtype: str | None,
    low_mem: bool,
) -> None:
    """Generate a single image from ``prompt`` and write it to ``output``.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id of the model to run.
    prompt : str
        Text prompt describing the desired image.
    negative_prompt : str or None
        Text describing what to avoid, or None.
    steps : int
        Number of denoising steps.
    width, height : int
        Output image dimensions in pixels.
    output : Path
        Destination path for the saved image.
    seed : int or None
        Random seed for reproducibility, or None for a random seed.
    device : str or None
        Device override (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``), or None to autodetect.
    dtype : str or None
        Torch dtype override, or None to autodetect.
    low_mem : bool
        If True, enable memory-saving optimizations (e.g. CPU offload).
    """
    from diffusion.core import generate as generate_module
    from diffusion.core.generate import write_sidecar
    from diffusion.utils import ui
    from diffusion.utils.console import console
    from diffusion.utils.terminal_image import detect_protocol

    protocol = detect_protocol()
    if protocol == "none":
        console.print(
            "[yellow]No inline-image terminal detected.[/yellow] Previews are disabled; "
            "the image will still be saved. (Force with DIFFUSION_FORCE_KITTY=1.)"
        )

    with ui.loading_status(f"Loading {repo_id} …"):
        pipe, kind, plan = generate_module.load_pipeline(
            repo_id, device_override=device, dtype_override=dtype, low_mem=low_mem
        )
    console.print(ui.model_ready_panel(repo_id, kind, plan.device, plan.dtype))
    console.print(f"[dim]Generating {width}×{height}, {steps} steps …[/dim]")

    image, elapsed = ui.run_with_preview(
        generate_module,
        pipe,
        kind,
        plan,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        low_mem=low_mem,
        protocol=protocol,
        rows=20,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    write_sidecar(
        output,
        repo_id=repo_id,
        kind=kind,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        device=plan.device,
        dtype=plan.dtype,
        elapsed_s=round(elapsed, 2),
    )
    console.print(ui.result_panel(output, seed, steps, f"{width}×{height}", elapsed))
