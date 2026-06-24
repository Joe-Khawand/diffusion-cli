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
    force_size: bool = False,
    init_image: Path | None = None,
    mask_image: Path | None = None,
    control_image: Path | None = None,
    controlnet: str | None = None,
    strength: float | None = None,
    guidance_scale: float | None = None,
    sampler: str | None = None,
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
    force_size : bool
        If True, bypass the pre-flight memory safety check for the requested size.
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
    sampler : str or None
        Sampler/scheduler name to use (e.g. ``"euler"``, ``"dpm++"``), or None to
        keep the model's default scheduler.
    """
    from diffusion.utils.console import quiet_diffusion_libraries

    quiet_diffusion_libraries()

    from diffusion.core import generate as generate_module
    from diffusion.core import registry
    from diffusion.core.generate import load_image, write_sidecar
    from diffusion.core.models import Task
    from diffusion.utils import ui
    from diffusion.utils.console import console
    from diffusion.utils.terminal_image import detect_protocol

    repo_id = registry.resolve_repo(repo_id)
    protocol = detect_protocol()
    if protocol == "none":
        console.print(
            "[yellow]No inline-image terminal detected.[/yellow] Previews are disabled; "
            "the image will still be saved. (Force with DIFFUSION_FORCE_KITTY=1.)"
        )

    # The task is inferred from the inputs: a mask → inpaint, an init image → img2img.
    if mask_image is not None:
        task = Task.INPAINT
    elif init_image is not None:
        task = Task.IMG2IMG
    else:
        task = Task.TEXT2IMG

    with ui.loading_status(f"Loading {repo_id} …"):
        pipe, family, plan = generate_module.load_pipeline(
            repo_id,
            task=task,
            controlnet_repo=controlnet,
            device_override=device,
            dtype_override=dtype,
            low_mem=low_mem,
            sampler=sampler,
        )
    console.print(ui.model_ready_panel(repo_id, family, plan.device, plan.dtype))
    console.print(f"[dim]Generating {task} · {width}×{height}, {steps} steps …[/dim]")

    image, elapsed = ui.run_with_preview(
        generate_module,
        pipe,
        family,
        plan,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        low_mem=low_mem,
        force_size=force_size,
        protocol=protocol,
        rows=20,
        init_image=load_image(init_image),
        mask_image=load_image(mask_image),
        control_image=load_image(control_image),
        strength=strength,
        guidance_scale=guidance_scale,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    write_sidecar(
        output,
        repo_id=repo_id,
        family=family,
        task=str(task),
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        controlnet=controlnet,
        strength=strength,
        guidance_scale=guidance_scale,
        sampler=type(pipe.scheduler).__name__,
        device=plan.device,
        dtype=plan.dtype,
        elapsed_s=round(elapsed, 2),
    )
    console.print(ui.result_panel(output, seed, steps, f"{width}×{height}", elapsed))
