"""Tests for pipeline auto-detection. Offline, no torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from diffusion.core.detect import detect_kind, list_components
from diffusion.core.models import PipelineKind

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("model_index", "expected"),
    [
        ("model_index_sd15.json", PipelineKind.SD15),
        ("model_index_sdxl.json", PipelineKind.SDXL),
        ("model_index_sd3.json", PipelineKind.SD3),
        ("model_index_flux.json", PipelineKind.FLUX),
        ("model_index_unknown.json", PipelineKind.UNKNOWN),
    ],
)
def test_detect_by_class_name(make_snapshot, model_index: str, expected: PipelineKind) -> None:
    snap = make_snapshot(model_index=model_index)
    assert detect_kind(snap) is expected


def test_missing_model_index_is_unknown(tmp_path: Path) -> None:
    assert detect_kind(tmp_path) is PipelineKind.UNKNOWN


def test_fingerprint_without_class_name(make_snapshot, fixtures_dir: Path) -> None:
    # Strip _class_name to force the component-fingerprint path.
    import json

    snap = make_snapshot(model_index="model_index_sdxl.json")
    data = json.loads((snap / "model_index.json").read_text())
    del data["_class_name"]
    (snap / "model_index.json").write_text(json.dumps(data))
    assert detect_kind(snap) is PipelineKind.SDXL


def test_sd_line_disambiguated_by_unet_config(make_snapshot) -> None:
    # SD-class index but UNet config reveals SDXL dims.
    import json

    snap = make_snapshot(model_index="model_index_sd15.json", unet_config="config_unet_sdxl.json")
    data = json.loads((snap / "model_index.json").read_text())
    del data["_class_name"]
    (snap / "model_index.json").write_text(json.dumps(data))
    assert detect_kind(snap) is PipelineKind.SDXL


def test_list_components(make_snapshot) -> None:
    snap = make_snapshot(model_index="model_index_sdxl.json")
    components = list_components(snap)
    assert "unet" in components and "vae" in components
    assert all(not c.startswith("_") for c in components)
