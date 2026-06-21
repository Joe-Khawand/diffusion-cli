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


class Task(StrEnum):
    """A generation task, selecting which Auto pipeline class to load."""

    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"


@dataclass(frozen=True)
class PreviewSpec:
    """Linear latent→RGB projection for fast live previews.

    ``factors`` is a ``channels x 3`` matrix mapping each latent channel to an
    (R, G, B) contribution; ``bias`` is added afterwards. Values are calibrated
    against the model's *working* latent space (the same approximation ComfyUI
    uses), so previews are cheap but approximate.
    """

    factors: list[list[float]]
    bias: list[float]
    channels: int


@dataclass(frozen=True)
class ModelFamily:
    """A diffusion model architecture and the knobs the runner needs for it.

    ``id`` is a stable slug (e.g. ``"sdxl"``) written to image sidecars and shown
    in listings; keep it stable across releases. ``class_names`` are the diffusers
    pipeline ``_class_name`` values (across text2img/img2img/inpaint/controlnet
    variants) that map to this family.
    """

    id: str
    label: str
    class_names: tuple[str, ...] = ()
    memory_heavy: bool = False
    latent_channels: int = 4
    supports_negative_prompt: bool = True
    cuda_dtype: str = "float16"  # "float16" | "bfloat16"
    preview: PreviewSpec | None = None
    example_repos: tuple[str, ...] = ()
    supported: bool = True

    def __str__(self) -> str:  # so f-strings / sidecars render the slug
        return self.id


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
    family: ModelFamily
    size_on_disk: int
    size_on_disk_str: str
    last_modified: float | None
    commit_hash: str | None
    local_path: Path
    components: list[str] = field(default_factory=list)
