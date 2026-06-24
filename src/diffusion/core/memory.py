"""Memory guardrails: detect available memory and estimate a run's footprint.

The point of this module is to refuse a generation that would likely exhaust
RAM/VRAM *before* it allocates anything — on unified-memory machines (Macs) an
oversized request can swap the whole system to death rather than failing
cleanly. We compare a conservative estimate of the incremental activation
memory against the memory currently available.

Torch/psutil are imported lazily inside functions so importing this module stays
cheap (matching the rest of ``core``). ``models.py`` stays torch-free; this
module may depend on it for the :class:`ModelFamily` type only.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from diffusion.utils.errors import DiffusionError, InsufficientMemoryError

if TYPE_CHECKING:
    from diffusion.core.models import ModelFamily

# Env var that bypasses the pre-flight memory check entirely (mirrors the
# DIFFUSION_FORCE_KITTY / DIFFUSION_NO_BORDER convention used elsewhere).
_SKIP_ENV = "DIFFUSION_SKIP_MEM_CHECK"

# --- Estimate constants (coarse, empirical; see estimate_peak_activation_bytes) ---
# Calibrated against SD1.5 fp16 so 512² passes comfortably and 1024² lands in
# multi-GB territory. They are intentionally rough — the SAFETY factor and the
# --force-size / DIFFUSION_SKIP_MEM_CHECK escape hatch exist precisely because
# this is an approximation, not a measurement.
_FEATURE_K = 1500  # linear feature-map term
_ATTN_K = 8  # quadratic self-attention term (the catastrophic driver)
_SAFETY = 2.0  # generous headroom multiplier


def available_bytes(device: str) -> int | None:
    """Return best-effort available memory for ``device``, in bytes.

    Parameters
    ----------
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    int or None
        Free VRAM for cuda, available system RAM (unified memory) for mps/cpu, or
        ``None`` if it cannot be determined — callers should then *not* block.
    """
    try:
        if device == "cuda":
            import torch

            free, _total = torch.cuda.mem_get_info()
            return int(free)
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def total_bytes(device: str) -> int | None:
    """Return total memory for ``device`` in bytes, or ``None`` if unknown."""
    try:
        if device == "cuda":
            import torch

            return int(torch.cuda.get_device_properties(0).total_memory)
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def estimate_peak_activation_bytes(
    *, width: int, height: int, dtype: str, steps: int, family: ModelFamily
) -> int:
    """Conservatively estimate the *incremental* peak activation memory, in bytes.

    This is a heuristic, not a measurement. By the time generation runs the model
    weights are already resident, so the relevant quantity is the activation
    memory a forward pass allocates on top of that.

    The dominant, catastrophic term is the self-attention score matrix, which
    scales with the square of the latent token count ``(w/8)·(h/8)``: going from
    512² to 1024² multiplies the token count by 4 and the score matrix by 16.
    A smaller linear term covers the feature maps. ``steps`` is deliberately
    *excluded* — peak activation is per-step, not cumulative — so high-step,
    low-resolution runs are not falsely blocked.

    Parameters
    ----------
    width, height : int
        Requested image dimensions in pixels.
    dtype : str
        Resolved dtype string (``"float16"``/``"bfloat16"`` → 2 bytes, else 4).
    steps : int
        Denoising steps (accepted for signature symmetry; intentionally unused).
    family : ModelFamily
        Provides ``latent_channels`` and the ``memory_heavy`` flag (SDXL/FLUX-class).

    Returns
    -------
    int
        Estimated peak activation memory in bytes, including a safety factor.
    """
    del steps  # peak activation is per-step; see docstring.
    tokens = (width / 8) * (height / 8)
    bpe = 2 if dtype in ("float16", "bfloat16") else 4
    chan = family.latent_channels
    heavy = 2.0 if family.memory_heavy else 1.0

    linear = _FEATURE_K * tokens * chan * bpe
    quadratic = _ATTN_K * (tokens**2) * bpe
    peak = heavy * (linear + quadratic)
    return int(_SAFETY * peak)


def _bytes_per_element(name: str) -> int:
    """Bytes per tensor element for a precision (``fp16``) or dtype (``float16``)."""
    return 4 if name in ("fp32", "float32") else 2


def estimate_runtime_peak_bytes(
    *,
    weight_bytes: int,
    weight_precision: str,
    width: int,
    height: int,
    family: ModelFamily,
    device: str,
) -> int:
    """Estimate peak memory to run the model on ``device``, in bytes.

    Peak ≈ resident weights (cast to the device's load dtype) + activations. The
    weights term is derived from a known on-disk size: ``weight_bytes`` are the
    bytes of the ``weight_precision`` variant, so ``params`` ≈
    ``weight_bytes / bytes(weight_precision)``, and resident memory scales that by
    the device's load dtype (e.g. fp16 on MPS). This is why the readout is largely
    independent of which precision variant the user downloads.

    Parameters
    ----------
    weight_bytes : int
        On-disk size of the model weights for ``weight_precision``.
    weight_precision : str
        Precision of ``weight_bytes`` (``"fp16"``/``"bf16"``/``"fp32"``).
    width, height : int
        Image dimensions in pixels.
    family : ModelFamily
        The detected model family.
    device : str
        Target device (``"cuda"``/``"mps"``/``"cpu"``).

    Returns
    -------
    int
        Estimated peak memory in bytes (weights + activations).
    """
    from diffusion.core import hardware

    load_dtype = hardware.select_dtype(device, family)
    params = weight_bytes / _bytes_per_element(weight_precision)
    resident = params * _bytes_per_element(load_dtype)
    activations = estimate_peak_activation_bytes(
        width=width, height=height, dtype=load_dtype, steps=1, family=family
    )
    return int(resident + activations)


def check_memory(
    *, width: int, height: int, dtype: str, steps: int, family: ModelFamily, device: str
) -> None:
    """Raise :class:`InsufficientMemoryError` if a run likely won't fit in memory.

    No-op when the override env var is set or when available memory cannot be
    determined (degrade to "don't block" rather than guess).

    Parameters
    ----------
    width, height : int
        Requested image dimensions in pixels.
    dtype : str
        Resolved dtype string.
    steps : int
        Denoising steps (forwarded to the estimate; currently unused there).
    family : ModelFamily
        The detected model family.
    device : str
        Resolved device (``"cuda"``/``"mps"``/``"cpu"``).

    Raises
    ------
    InsufficientMemoryError
        If the estimated peak activation exceeds available memory.
    """
    if os.environ.get(_SKIP_ENV):
        return
    avail = available_bytes(device)
    if avail is None:
        return
    need = estimate_peak_activation_bytes(
        width=width, height=height, dtype=dtype, steps=steps, family=family
    )
    if need > avail:
        raise InsufficientMemoryError(
            width=width, height=height, device=device, need_bytes=need, avail_bytes=avail
        )


def validate_dimensions(width: int, height: int) -> None:
    """Reject non-positive or non-multiple-of-8 dimensions with a clear error.

    Latent VAEs downsample by 8, so dimensions that aren't multiples of 8 get
    silently rounded or fail deep inside diffusers; catch them early.

    Parameters
    ----------
    width, height : int
        Requested image dimensions in pixels.

    Raises
    ------
    DiffusionError
        If a dimension is non-positive or not a multiple of 8.
    """
    for name, val in (("width", width), ("height", height)):
        if val <= 0:
            raise DiffusionError(f"{name} must be positive, got {val}.")
        if val % 8 != 0:
            raise DiffusionError(
                f"{name} must be a multiple of 8, got {val}.",
                hint=f"Try {round(val / 8) * 8}.",
            )
