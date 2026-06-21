"""Hardware detection and dtype selection.

Imports torch lazily inside functions so the CLI layer stays light.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from diffusion.core.models import DeviceInfo

if TYPE_CHECKING:
    import torch

    from diffusion.core.models import ModelFamily

_VALID_DEVICES = {"mps", "cuda", "cpu"}
_VALID_DTYPES = {"float16", "bfloat16", "float32"}


def enable_mps_fallback() -> None:
    """Allow unsupported MPS ops to fall back to CPU instead of crashing.

    Must be set before torch initializes the MPS backend.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def detect_device(override: str | None = None) -> str:
    """Return the best available device, honoring an explicit override."""
    import torch

    if override is not None:
        if override not in _VALID_DEVICES:
            from diffusion.utils.errors import DiffusionError

            raise DiffusionError(f"Unknown device '{override}'. Choose mps, cuda, or cpu.")
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str, family: ModelFamily, override: str | None = None) -> str:
    """Pick a sensible dtype string for the device/family."""
    if override is not None:
        if override not in _VALID_DTYPES:
            from diffusion.utils.errors import DiffusionError

            raise DiffusionError(
                f"Unknown dtype '{override}'. Choose float16, bfloat16, or float32."
            )
        return override
    if device == "cpu":
        return "float32"
    if device == "cuda":
        # Each family declares its preferred cuda dtype (bf16 for the big DiTs).
        return family.cuda_dtype
    # mps: float16 is the pragmatic fast default for SD/SDXL.
    return "float16"


def resolve(
    *, family: ModelFamily, device_override: str | None, dtype_override: str | None
) -> DeviceInfo:
    """Resolve the full device + dtype plan for a run."""
    enable_mps_fallback()
    device = detect_device(device_override)
    dtype = select_dtype(device, family, dtype_override)
    return DeviceInfo(device=device, dtype=dtype)


def torch_dtype(dtype: str) -> torch.dtype:
    """Map a dtype string to a real ``torch.dtype``."""
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
