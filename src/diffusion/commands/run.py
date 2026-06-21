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
    from diffusion.core.generate import generate

    generate(
        repo_id=repo_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        output=output,
        seed=seed,
        device_override=device,
        dtype_override=dtype,
        low_mem=low_mem,
    )
