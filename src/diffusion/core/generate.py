"""Image generation orchestration: load → route → optimize → infer → save.

All heavy imports (torch, diffusers) live inside functions so importing this
module stays cheap. Split into :func:`load_pipeline` (do once) and
:func:`run_inference` (do per prompt) so interactive mode can reuse a pipeline.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from diffusion.core.models import DeviceInfo, PipelineKind
from diffusion.utils.console import console
from diffusion.utils.errors import UnsupportedPipelineError

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

# Called with (step_index, preview_image) after each denoising step.
PreviewCallback = Callable[[int, "Image"], None]


def load_pipeline(
    repo_id: str, *, device_override: str | None, dtype_override: str | None, low_mem: bool
):
    """Resolve, load, and optimize a pipeline. Returns (pipe, kind, plan)."""
    from diffusion.utils.console import suppress_transformers_docstring_noise

    # Importing diffusers/transformers model modules triggers @auto_docstring,
    # which prints cosmetic "[ERROR] ... not documented" lines to stdout.
    with suppress_transformers_docstring_noise():
        from diffusers import AutoPipelineForText2Image

        from diffusion.core import cache, hardware, optimize
        from diffusion.core.detect import detect_kind

        snapshot = cache.resolve_local(repo_id)
        kind = detect_kind(snapshot)
        if not kind.is_supported:
            raise UnsupportedPipelineError(
                repo_id, "no recognized diffusion pipeline in model_index.json"
            )

        plan = hardware.resolve(
            kind=kind, device_override=device_override, dtype_override=dtype_override
        )
        pipe = AutoPipelineForText2Image.from_pretrained(
            snapshot, torch_dtype=hardware.torch_dtype(plan.dtype)
        )
    optimize.apply_optimizations(pipe, plan.device, kind, low_mem=low_mem)
    return pipe, kind, plan


def run_inference(
    pipe,
    kind: PipelineKind,
    plan: DeviceInfo,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    width: int,
    height: int,
    seed: int | None,
    low_mem: bool,
    on_preview: PreviewCallback | None = None,
) -> Image:
    """Run one generation on an already-loaded pipeline; return the image."""
    import torch

    generator = None
    if seed is not None:
        gen_device = "cpu" if (low_mem or kind.is_memory_heavy) else plan.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    call_kwargs: dict = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
    }
    if negative_prompt is not None and kind is not PipelineKind.FLUX:
        # FLUX has no negative prompt / CFG path by default.
        call_kwargs["negative_prompt"] = negative_prompt
    if generator is not None:
        call_kwargs["generator"] = generator

    if on_preview is not None:
        from diffusion.core.preview import latents_to_preview

        # Silence the pipeline's own tqdm bar so it doesn't fight the inline render.
        with contextlib.suppress(Exception):
            pipe.set_progress_bar_config(disable=True)

        def _callback(_pipe, step_index, _timestep, callback_kwargs):
            try:
                preview = latents_to_preview(callback_kwargs["latents"], kind)
                if preview is not None:
                    on_preview(step_index, preview)
            except Exception:
                pass  # previews are best-effort; never break generation
            return callback_kwargs

        call_kwargs["callback_on_step_end"] = _callback
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    return pipe(**call_kwargs).images[0]


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
) -> Path:
    """One-shot generation: load, run, and save image + sidecar."""
    pipe, kind, plan = load_pipeline(
        repo_id, device_override=device_override, dtype_override=dtype_override, low_mem=low_mem
    )
    console.print(
        f"Loaded [bold]{repo_id}[/bold] ([cyan]{kind}[/cyan]) on "
        f"[magenta]{plan.device}[/magenta]/{plan.dtype}. Generating "
        f"{width}×{height}, {steps} steps …"
    )
    start = time.perf_counter()
    image = run_inference(
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
    )
    elapsed = time.perf_counter() - start

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
    console.print(f"[green]✓[/green] Saved [bold]{output}[/bold] in {elapsed:.1f}s")
    return output


def write_sidecar(output: Path, **metadata) -> None:
    """Write a JSON sidecar next to the image recording how it was made."""
    metadata["kind"] = str(metadata["kind"])
    sidecar = output.with_suffix(output.suffix + ".json")
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
