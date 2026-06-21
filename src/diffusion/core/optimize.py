"""Apply memory/perf optimizations to a loaded diffusers pipeline.

Critical rule: ``pipe.to(device)`` and CPU offload are mutually exclusive —
accelerate manages device placement when offload is enabled, and calling
``.to()`` on top of it breaks generation (a common diffusers/MPS bug).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusion.core.models import PipelineKind


def apply_optimizations(pipe, device: str, kind: PipelineKind, *, low_mem: bool) -> None:
    """Configure ``pipe`` for ``device``; returns nothing (mutates in place).

    - SD1.5/SDXL on adequate memory: place on device, no slicing (keeps it fast).
    - FLUX/SD3 or ``--low-mem``: enable slicing + sequential CPU offload instead
      of ``.to(device)``.
    """
    use_offload = low_mem or kind.is_memory_heavy

    if use_offload:
        # Slicing reduces peak memory; pair with offload for large models / low VRAM.
        _try(pipe, "enable_attention_slicing")
        _try(pipe, "enable_vae_slicing")
        _try(pipe, "enable_vae_tiling")
        # Sequential offload keeps only the active submodule on-device.
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        # NOTE: deliberately do NOT call pipe.to(device) here.
    else:
        pipe.to(device)


def _try(pipe, method: str) -> None:
    fn = getattr(pipe, method, None)
    if callable(fn):
        fn()
