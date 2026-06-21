"""Tests for device + dtype selection."""

from __future__ import annotations

import pytest

from diffusion.core import hardware, registry
from diffusion.utils.errors import DiffusionError

_SDXL = registry.by_class_name("StableDiffusionXLPipeline")
_FLUX = registry.by_class_name("FluxPipeline")
_SD3 = registry.by_class_name("StableDiffusion3Pipeline")


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_detect_device(mocker, cuda, mps, expected) -> None:
    mocker.patch("torch.cuda.is_available", return_value=cuda)
    mocker.patch("torch.backends.mps.is_available", return_value=mps)
    assert hardware.detect_device() == expected


def test_detect_device_override() -> None:
    assert hardware.detect_device("cpu") == "cpu"


def test_detect_device_bad_override() -> None:
    with pytest.raises(DiffusionError):
        hardware.detect_device("tpu")


@pytest.mark.parametrize(
    ("device", "family", "expected"),
    [
        ("cpu", _SDXL, "float32"),
        ("mps", _SDXL, "float16"),
        ("mps", _FLUX, "float16"),
        ("cuda", _SDXL, "float16"),
        ("cuda", _FLUX, "bfloat16"),
        ("cuda", _SD3, "bfloat16"),
    ],
)
def test_select_dtype(device, family, expected) -> None:
    assert hardware.select_dtype(device, family) == expected


def test_select_dtype_override() -> None:
    assert hardware.select_dtype("mps", _SDXL, "bfloat16") == "bfloat16"


def test_select_dtype_bad_override() -> None:
    with pytest.raises(DiffusionError):
        hardware.select_dtype("mps", _SDXL, "int4")


def test_enable_mps_fallback(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    hardware.enable_mps_fallback()
    import os

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
