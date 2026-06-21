"""Tests for optimization routing. Pipeline is a MagicMock — no real diffusers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from diffusion.core import registry
from diffusion.core.optimize import apply_optimizations

_SDXL = registry.by_class_name("StableDiffusionXLPipeline")
_FLUX = registry.by_class_name("FluxPipeline")
_SD3 = registry.by_class_name("StableDiffusion3Pipeline")


def test_sdxl_default_places_on_device_without_offload() -> None:
    pipe = MagicMock()
    apply_optimizations(pipe, "mps", _SDXL, low_mem=False)
    pipe.to.assert_called_once_with("mps")
    pipe.enable_sequential_cpu_offload.assert_not_called()
    # Default fast path should not slice.
    pipe.enable_attention_slicing.assert_not_called()


def test_low_mem_uses_offload_and_not_to() -> None:
    pipe = MagicMock()
    apply_optimizations(pipe, "mps", _SDXL, low_mem=True)
    pipe.to.assert_not_called()
    pipe.enable_sequential_cpu_offload.assert_called_once()
    pipe.enable_attention_slicing.assert_called_once()


@pytest.mark.parametrize("family", [_FLUX, _SD3])
def test_heavy_models_offload_by_default(family) -> None:
    pipe = MagicMock()
    apply_optimizations(pipe, "mps", family, low_mem=False)
    pipe.to.assert_not_called()
    pipe.enable_sequential_cpu_offload.assert_called_once()


def test_missing_methods_are_skipped() -> None:
    # A pipeline lacking optional slicing methods should not error.
    pipe = MagicMock(spec=["enable_sequential_cpu_offload", "to"])
    apply_optimizations(pipe, "cpu", _FLUX, low_mem=False)
    pipe.enable_sequential_cpu_offload.assert_called_once()
