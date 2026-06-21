"""Shared, dependency-light data types.

This module MUST stay free of torch/diffusers imports so it can be imported by
the CLI layer without paying the heavy-import cost. Any torch types are referred
to via string annotations only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class PipelineKind(StrEnum):
    """Diffusion pipeline family detected from a model's metadata."""

    SD15 = "sd1.5"
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX = "flux"
    UNKNOWN = "unknown"

    @property
    def is_supported(self) -> bool:
        """Phase 1 supports all detected families (FLUX/SD3 best-effort)."""
        return self is not PipelineKind.UNKNOWN

    @property
    def is_memory_heavy(self) -> bool:
        """FLUX/SD3 are large; default to CPU offload on consumer memory."""
        return self in (PipelineKind.FLUX, PipelineKind.SD3)


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved hardware target for a generation run.

    ``dtype`` is the string form (e.g. ``"float16"``) to keep this torch-free;
    callers resolve it to a real ``torch.dtype`` at load time.
    """

    device: str  # "mps" | "cuda" | "cpu"
    dtype: str  # "float16" | "bfloat16" | "float32"


@dataclass
class ModelEntry:
    """A diffusion model present in the local HuggingFace cache."""

    repo_id: str
    kind: PipelineKind
    size_on_disk: int
    size_on_disk_str: str
    last_modified: float | None
    commit_hash: str | None
    local_path: Path
    components: list[str] = field(default_factory=list)
