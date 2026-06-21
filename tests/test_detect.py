"""Tests for pipeline-family auto-detection. Offline, no torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from diffusion.core import registry
from diffusion.core.detect import detect_family, list_components

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("model_index", "expected_id"),
    [
        ("model_index_sd15.json", "sd1.5"),
        ("model_index_sdxl.json", "sdxl"),
        ("model_index_sd3.json", "sd3"),
        ("model_index_flux.json", "flux"),
        ("model_index_pixart.json", "pixart-sigma"),
        ("model_index_unknown.json", "unknown"),  # audio pipeline → flagged
    ],
)
def test_detect_by_class_name(make_snapshot, model_index: str, expected_id: str) -> None:
    snap = make_snapshot(model_index=model_index)
    assert detect_family(snap).id == expected_id


def test_unknown_image_pipeline_falls_back_to_generic(make_snapshot) -> None:
    # A recognizable but uncurated *Pipeline → trust AutoPipeline via GENERIC.
    snap = make_snapshot(model_index="model_index_generic.json")
    assert detect_family(snap) is registry.GENERIC


def test_missing_model_index_is_unknown(tmp_path: Path) -> None:
    assert detect_family(tmp_path) is registry.UNKNOWN


def test_non_image_pipeline_is_unknown(make_snapshot) -> None:
    # The 'unknown' fixture is an audio pipeline; it must not be run as an image.
    snap = make_snapshot(model_index="model_index_unknown.json")
    assert detect_family(snap).supported is False


def test_fingerprint_without_class_name(make_snapshot) -> None:
    # Strip _class_name to force the component-fingerprint path.
    import json

    snap = make_snapshot(model_index="model_index_sdxl.json")
    data = json.loads((snap / "model_index.json").read_text())
    del data["_class_name"]
    (snap / "model_index.json").write_text(json.dumps(data))
    assert detect_family(snap).id == "sdxl"


def test_sd_line_disambiguated_by_unet_config(make_snapshot) -> None:
    # SD-class index but UNet config reveals SDXL dims.
    import json

    snap = make_snapshot(model_index="model_index_sd15.json", unet_config="config_unet_sdxl.json")
    data = json.loads((snap / "model_index.json").read_text())
    del data["_class_name"]
    (snap / "model_index.json").write_text(json.dumps(data))
    assert detect_family(snap).id == "sdxl"


def test_list_components(make_snapshot) -> None:
    snap = make_snapshot(model_index="model_index_sdxl.json")
    components = list_components(snap)
    assert "unet" in components and "vae" in components
    assert all(not c.startswith("_") for c in components)
