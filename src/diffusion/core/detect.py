"""Pipeline-family auto-detection.

Pure functions: no torch, no diffusers, no network. Detection reads the JSON
metadata that diffusers-format repos ship (``model_index.json`` and component
``config.json`` files) and maps it to a :class:`ModelFamily` from the registry.

Signal priority:
  1. ``model_index.json`` ``_class_name`` mapped via the registry (authoritative).
  2. Any other ``*Pipeline`` ``_class_name`` -> permissive ``GENERIC`` family.
  3. Component fingerprint (which sub-folders/keys are present).
  4. ``unet/config.json`` cross-attention dim (SD1.5 vs SDXL disambiguation).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from diffusion.core import registry

if TYPE_CHECKING:
    from pathlib import Path

    from diffusion.core.models import ModelFamily

_SDXL = registry.require("StableDiffusionXLPipeline")
_SD15 = registry.require("StableDiffusionPipeline")
_SD3 = registry.require("StableDiffusion3Pipeline")
_FLUX = registry.require("FluxPipeline")


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def detect_family(snapshot_dir: Path) -> ModelFamily:
    """Detect the model family from a local diffusers snapshot directory."""
    index = _read_json(snapshot_dir / "model_index.json")
    if index is None:
        return registry.UNKNOWN

    class_name = index.get("_class_name")
    class_name = class_name if isinstance(class_name, str) else None

    # Non-image pipelines (video/audio) are flagged, not optimistically routed.
    if registry.is_non_image(class_name):
        return registry.UNKNOWN

    # 1. Authoritative: a curated family for this diffusers class.
    family = registry.by_class_name(class_name)
    if family is not None:
        # The SD line class is shared by SD1.5/SD2.x — disambiguate XL.
        if family is _SD15:
            return _refine_sd_line(snapshot_dir, index)
        return family

    # 2. Unknown but recognizable diffusers pipeline -> trust AutoPipeline.
    if class_name and class_name.endswith("Pipeline"):
        return registry.GENERIC

    # 3. No class name: fall back to a component fingerprint.
    return _fingerprint(snapshot_dir, index)


def _fingerprint(snapshot_dir: Path, index: dict) -> ModelFamily:
    keys = set(index.keys())
    has_unet = "unet" in keys
    has_transformer = "transformer" in keys

    if has_transformer and "text_encoder_3" in keys:
        return _SD3
    if has_transformer and not has_unet:
        return _FLUX
    if has_unet and "text_encoder_2" in keys:
        return _SDXL
    if has_unet:
        return _refine_sd_line(snapshot_dir, index)
    return registry.UNKNOWN


def _refine_sd_line(snapshot_dir: Path, index: dict) -> ModelFamily:
    """Distinguish SDXL from SD1.5 using component hints, then UNet config."""
    if "text_encoder_2" in index:
        return _SDXL

    unet_config = _read_json(snapshot_dir / "unet" / "config.json")
    if unet_config is not None:
        if unet_config.get("addition_embed_type") == "text_time":
            return _SDXL
        cross_dim = unet_config.get("cross_attention_dim")
        if isinstance(cross_dim, int) and cross_dim >= 2048:
            return _SDXL
    return _SD15


def list_components(snapshot_dir: Path) -> list[str]:
    """Return the diffusers component names declared in ``model_index.json``."""
    index = _read_json(snapshot_dir / "model_index.json")
    if index is None:
        return []
    return sorted(k for k, v in index.items() if not k.startswith("_") and isinstance(v, list))
