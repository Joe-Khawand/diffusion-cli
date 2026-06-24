"""Image generation orchestration: load → route → optimize → infer → save.

All heavy imports (torch, diffusers) live inside functions so importing this
module stays cheap. Split into :func:`load_pipeline` (do once) and
:func:`run_inference` (do per prompt) so interactive mode can reuse a pipeline.

Call arguments are filtered against each pipeline's ``__call__`` signature, so a
single code path drives the dozens of diffusers families without per-family
special-casing — we pass what a pipeline accepts and let it ignore the rest.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from diffusion.core.models import DeviceInfo, ModelFamily, Task
from diffusion.utils.console import console
from diffusion.utils.errors import InsufficientMemoryError, UnsupportedPipelineError

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

# Called with (step_index, preview_image) after each denoising step. The preview
# is None for families without a latent->RGB projection; callbacks still fire so
# progress reporting works regardless of whether a preview can be rendered.
PreviewCallback = Callable[[int, "Image | None"], None]


def _auto_class(task: Task):
    """Return the diffusers Auto pipeline class for ``task`` (resolved lazily)."""
    from diffusers import (
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        AutoPipelineForText2Image,
    )

    return {
        Task.TEXT2IMG: AutoPipelineForText2Image,
        Task.IMG2IMG: AutoPipelineForImage2Image,
        Task.INPAINT: AutoPipelineForInpainting,
    }[task]


def load_pipeline(
    repo_id: str,
    *,
    task: Task = Task.TEXT2IMG,
    controlnet_repo: str | None = None,
    device_override: str | None,
    dtype_override: str | None,
    low_mem: bool,
    sampler: str | None = None,
):
    """Resolve, load, and optimize a pipeline. Returns (pipe, family, plan).

    If ``sampler`` is given, the pipeline's default scheduler is swapped for the
    named sampler (see :mod:`diffusion.core.samplers`); otherwise the model's own
    default scheduler is kept.
    """
    from diffusion.core import cache, hardware, optimize
    from diffusion.core.detect import detect_family
    from diffusion.utils.console import suppress_transformers_docstring_noise

    snapshot = cache.resolve_local(repo_id)
    family = detect_family(snapshot)
    if not family.supported:
        raise UnsupportedPipelineError(
            repo_id, "no recognized diffusion pipeline in model_index.json"
        )

    plan = hardware.resolve(
        family=family, device_override=device_override, dtype_override=dtype_override
    )

    # Importing diffusers/transformers model modules triggers @auto_docstring,
    # which prints cosmetic "[ERROR] ... not documented" lines to stdout.
    with suppress_transformers_docstring_noise():
        load_kwargs: dict = {"torch_dtype": hardware.torch_dtype(plan.dtype)}
        # We download only the fp16 weight variant when one exists, so tell
        # diffusers to load it (it falls back to fp32 per-component otherwise).
        variant = cache.detect_variant(snapshot)
        if variant is not None:
            load_kwargs["variant"] = variant
        if controlnet_repo is not None:
            from diffusers import ControlNetModel

            cn_snapshot = cache.resolve_local(controlnet_repo)
            cn_kwargs: dict = {"torch_dtype": hardware.torch_dtype(plan.dtype)}
            cn_variant = cache.detect_variant(cn_snapshot)
            if cn_variant is not None:
                cn_kwargs["variant"] = cn_variant
            load_kwargs["controlnet"] = ControlNetModel.from_pretrained(cn_snapshot, **cn_kwargs)

        auto_cls = _auto_class(task)
        try:
            pipe = auto_cls.from_pretrained(snapshot, **load_kwargs)
        except ValueError as exc:
            raise UnsupportedPipelineError(
                repo_id, f"diffusers could not route this model to a {task} pipeline"
            ) from exc

    if sampler is not None:
        from diffusion.core import samplers

        samplers.apply_sampler(pipe, sampler)

    optimize.apply_optimizations(pipe, plan.device, family, low_mem=low_mem)
    return pipe, family, plan


def run_inference(
    pipe,
    family: ModelFamily,
    plan: DeviceInfo,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    width: int,
    height: int,
    seed: int | None,
    low_mem: bool,
    force_size: bool = False,
    init_image: Image | None = None,
    mask_image: Image | None = None,
    control_image: Image | None = None,
    strength: float | None = None,
    guidance_scale: float | None = None,
    on_preview: PreviewCallback | None = None,
) -> Image:
    """Run one generation on an already-loaded pipeline; return the image.

    Only arguments the pipeline's ``__call__`` actually accepts are forwarded, so
    families that ignore ``width``/``height`` (img2img) or lack a negative prompt
    (FLUX) work without bespoke handling.

    For text-to-image, the requested ``width``/``height`` are validated and a
    pre-flight memory check refuses sizes that would likely exhaust RAM/VRAM
    (unless ``force_size`` is set). img2img/inpaint take their size from the init
    image, so the check is skipped there.
    """
    import torch

    from diffusion.core import memory

    if init_image is None and mask_image is None:
        memory.validate_dimensions(width, height)
        if not force_size:
            memory.check_memory(
                width=width,
                height=height,
                dtype=plan.dtype,
                steps=steps,
                family=family,
                device=plan.device,
            )

    params = inspect.signature(pipe.__call__).parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def accepts(name: str) -> bool:
        return has_var_kw or name in params

    generator = None
    if seed is not None:
        gen_device = "cpu" if (low_mem or family.memory_heavy) else plan.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    candidates: dict = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
    }
    if negative_prompt is not None and family.supports_negative_prompt:
        candidates["negative_prompt"] = negative_prompt
    if guidance_scale is not None:
        candidates["guidance_scale"] = guidance_scale
    if strength is not None:
        candidates["strength"] = strength
    if generator is not None:
        candidates["generator"] = generator
    if init_image is not None:
        candidates["image"] = init_image
    if mask_image is not None:
        candidates["mask_image"] = mask_image
    if control_image is not None:
        # ControlNet img2img/inpaint expose `control_image`; plain ControlNet
        # text2img takes the control map as `image`.
        candidates["control_image" if accepts("control_image") else "image"] = control_image

    call_kwargs = {k: v for k, v in candidates.items() if accepts(k)}

    if on_preview is not None and accepts("callback_on_step_end"):
        from diffusion.core.preview import latents_to_preview

        # Silence the pipeline's own tqdm bar so it doesn't fight the inline render.
        with contextlib.suppress(Exception):
            pipe.set_progress_bar_config(disable=True)

        def _callback(_pipe, step_index, _timestep, callback_kwargs):
            # The preview image is None for families without a latent->RGB
            # projection; we still fire on_preview every step so progress reporting
            # advances regardless. Both are best-effort — never break generation.
            preview = None
            try:
                preview = latents_to_preview(callback_kwargs.get("latents"), family)
            except Exception:
                preview = None
            with contextlib.suppress(Exception):
                on_preview(step_index, preview)
            return callback_kwargs

        call_kwargs["callback_on_step_end"] = _callback
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    try:
        return pipe(**call_kwargs).images[0]
    except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)) or any(
            kw in str(exc).lower() for kw in ("out of memory", "alloc")
        )
        if not is_oom:
            raise  # an unrelated RuntimeError — leave it untouched
        if plan.device == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        raise InsufficientMemoryError.from_oom(
            width=width, height=height, device=plan.device
        ) from exc


def load_image(path: Path | None) -> Image | None:
    """Load an image from ``path`` as RGB, or return None if ``path`` is None."""
    if path is None:
        return None
    from PIL import Image as PILImage

    return PILImage.open(path).convert("RGB")


def generate(
    *,
    repo_id: str,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    width: int,
    height: int,
    output: Path,
    seed: int | None,
    device_override: str | None,
    dtype_override: str | None,
    low_mem: bool,
    init_image: Path | None = None,
    mask_image: Path | None = None,
    control_image: Path | None = None,
    controlnet_repo: str | None = None,
    strength: float | None = None,
    guidance_scale: float | None = None,
    sampler: str | None = None,
) -> Path:
    """One-shot generation: load, run, and save image + sidecar.

    The task is derived from the inputs: a mask → inpaint, an init image →
    img2img, otherwise text-to-image.
    """
    if mask_image is not None:
        task = Task.INPAINT
    elif init_image is not None:
        task = Task.IMG2IMG
    else:
        task = Task.TEXT2IMG

    pipe, family, plan = load_pipeline(
        repo_id,
        task=task,
        controlnet_repo=controlnet_repo,
        device_override=device_override,
        dtype_override=dtype_override,
        low_mem=low_mem,
        sampler=sampler,
    )
    console.print(
        f"Loaded [bold]{repo_id}[/bold] ([cyan]{family.label}[/cyan]) on "
        f"[magenta]{plan.device}[/magenta]/{plan.dtype}. Generating {task} · "
        f"{width}×{height}, {steps} steps …"
    )
    start = time.perf_counter()
    image = run_inference(
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
        init_image=load_image(init_image),
        mask_image=load_image(mask_image),
        control_image=load_image(control_image),
        strength=strength,
        guidance_scale=guidance_scale,
    )
    elapsed = time.perf_counter() - start

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
        controlnet=controlnet_repo,
        strength=strength,
        guidance_scale=guidance_scale,
        sampler=type(pipe.scheduler).__name__,
        device=plan.device,
        dtype=plan.dtype,
        elapsed_s=round(elapsed, 2),
    )
    console.print(f"[green]✓[/green] Saved [bold]{output}[/bold] in {elapsed:.1f}s")
    return output


def write_sidecar(output: Path, **metadata) -> None:
    """Write a JSON sidecar next to the image recording how it was made.

    A ``family`` is recorded under the stable ``kind`` slug for back-compat.
    """
    family = metadata.pop("family", None)
    if family is not None:
        metadata["kind"] = family.id
    sidecar = output.with_suffix(output.suffix + ".json")
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
