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
    init_image: Path | None = None,
    mask_image: Path | None = None,
    control_image: Path | None = None,
    controlnet: str | None = None,
    strength: float | None = None,
    guidance_scale: float | None = None,
) -> None:
    """Generate a single image from ``prompt`` and write it to ``output``.

    The task is inferred from the inputs: passing ``mask_image`` runs inpainting,
    ``init_image`` runs image-to-image, otherwise text-to-image. ``controlnet`` (a
    HuggingFace repo id) plus ``control_image`` adds ControlNet conditioning.

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
        Output image dimensions in pixels (ignored by img2img/inpaint pipelines).
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
    init_image : Path or None
        Init image for img2img / inpaint.
    mask_image : Path or None
        Mask for inpaint (white = repaint).
    control_image : Path or None
        Pre-processed control map (e.g. canny/depth) for ControlNet.
    controlnet : str or None
        HuggingFace repo id of a ControlNet model to condition on.
    strength : float or None
        Denoising strength for img2img/inpaint (0-1).
    guidance_scale : float or None
        Classifier-free guidance scale override.
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
        init_image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        controlnet_repo=controlnet,
        strength=strength,
        guidance_scale=guidance_scale,
    )
