"""Pipeline-type auto-detection.

Pure functions: no torch, no diffusers, no network. Detection reads the JSON
metadata that diffusers-format repos ship (``model_index.json`` and component
``config.json`` files) and maps it to a :class:`PipelineKind`.

Signal priority:
  1. ``model_index.json`` ``_class_name`` (authoritative).
  2. Component fingerprint (which sub-folders/keys are present).
  3. ``unet/config.json`` cross-attention dim (SD1.5 vs SDXL disambiguation).
"""

from __future__ import annotations

import json
from pathlib import Path

from diffusion.core.models import PipelineKind

# Concrete diffusers pipeline class names -> family.
_CLASS_NAME_MAP: dict[str, PipelineKind] = {
    "StableDiffusionPipeline": PipelineKind.SD15,
    "StableDiffusionImg2ImgPipeline": PipelineKind.SD15,
    "StableDiffusionXLPipeline": PipelineKind.SDXL,
    "StableDiffusionXLImg2ImgPipeline": PipelineKind.SDXL,
    "StableDiffusion3Pipeline": PipelineKind.SD3,
    "FluxPipeline": PipelineKind.FLUX,
}


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def detect_kind(snapshot_dir: Path) -> PipelineKind:
    """Detect the pipeline family from a local model snapshot directory."""
    index = _read_json(snapshot_dir / "model_index.json")
    if index is None:
        return PipelineKind.UNKNOWN

    # 1. Authoritative class name.
    class_name = index.get("_class_name")
    if isinstance(class_name, str) and class_name in _CLASS_NAME_MAP:
        kind = _CLASS_NAME_MAP[class_name]
        # SD line class is shared by SD1.5/SD2.x — disambiguate XL via components.
        if kind is PipelineKind.SD15:
            return _refine_sd_line(snapshot_dir, index)
        return kind

    # 2. Component fingerprint fallback.
    return _fingerprint(snapshot_dir, index)


def _fingerprint(snapshot_dir: Path, index: dict) -> PipelineKind:
    keys = set(index.keys())
    has_unet = "unet" in keys
    has_transformer = "transformer" in keys

    if has_transformer and "text_encoder_3" in keys:
        return PipelineKind.SD3
    if has_transformer and not has_unet:
        return PipelineKind.FLUX
    if has_unet and "text_encoder_2" in keys:
        return PipelineKind.SDXL
    if has_unet:
        return _refine_sd_line(snapshot_dir, index)
    return PipelineKind.UNKNOWN


def _refine_sd_line(snapshot_dir: Path, index: dict) -> PipelineKind:
    """Distinguish SDXL from SD1.5 using component hints, then UNet config."""
    if "text_encoder_2" in index:
        return PipelineKind.SDXL

    unet_config = _read_json(snapshot_dir / "unet" / "config.json")
    if unet_config is not None:
        if unet_config.get("addition_embed_type") == "text_time":
            return PipelineKind.SDXL
        cross_dim = unet_config.get("cross_attention_dim")
        if isinstance(cross_dim, int) and cross_dim >= 2048:
            return PipelineKind.SDXL
    return PipelineKind.SD15


def list_components(snapshot_dir: Path) -> list[str]:
    """Return the diffusers component names declared in ``model_index.json``."""
    index = _read_json(snapshot_dir / "model_index.json")
    if index is None:
        return []
    return sorted(k for k, v in index.items() if not k.startswith("_") and isinstance(v, list))
