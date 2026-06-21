"""Tests for the model family registry. Offline, no torch."""

from __future__ import annotations

from diffusion.core import registry


def test_by_class_name_maps_known_pipelines() -> None:
    assert registry.by_class_name("StableDiffusionXLPipeline").id == "sdxl"
    assert registry.by_class_name("FluxImg2ImgPipeline").id == "flux"
    assert registry.by_class_name("QwenImageInpaintPipeline").id == "qwen-image"
    assert registry.by_class_name("Unrecognized") is None
    assert registry.by_class_name(None) is None


def test_family_ids_and_class_names_are_unique() -> None:
    ids = [f.id for f in registry.FAMILIES]
    assert len(ids) == len(set(ids)), "duplicate family ids"

    seen: set[str] = set()
    for fam in registry.FAMILIES:
        for cls in fam.class_names:
            assert cls not in seen, f"class name {cls} mapped to two families"
            seen.add(cls)


def test_preview_specs_match_declared_channels() -> None:
    for fam in registry.FAMILIES:
        if fam.preview is not None:
            assert len(fam.preview.factors) == fam.preview.channels
            assert all(len(row) == 3 for row in fam.preview.factors)
            assert len(fam.preview.bias) == 3
            assert fam.preview.channels == fam.latent_channels


def test_non_image_detection() -> None:
    assert registry.is_non_image("StableVideoDiffusionPipeline") is True
    assert registry.is_non_image("MusicLDMPipeline") is True
    assert registry.is_non_image("StableDiffusionXLPipeline") is False


def test_fallbacks() -> None:
    assert registry.GENERIC.supported is True
    assert registry.GENERIC.preview is None
    assert registry.UNKNOWN.supported is False
