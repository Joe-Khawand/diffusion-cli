"""Tests for device + dtype selection."""

from __future__ import annotations

import pytest

from diffusion.core import hardware
from diffusion.core.models import PipelineKind
from diffusion.utils.errors import DiffusionError


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
    ("device", "kind", "expected"),
    [
        ("cpu", PipelineKind.SDXL, "float32"),
        ("mps", PipelineKind.SDXL, "float16"),
        ("mps", PipelineKind.FLUX, "float16"),
        ("cuda", PipelineKind.SDXL, "float16"),
        ("cuda", PipelineKind.FLUX, "bfloat16"),
        ("cuda", PipelineKind.SD3, "bfloat16"),
    ],
)
def test_select_dtype(device, kind, expected) -> None:
    assert hardware.select_dtype(device, kind) == expected


def test_select_dtype_override() -> None:
    assert hardware.select_dtype("mps", PipelineKind.SDXL, "bfloat16") == "bfloat16"


def test_select_dtype_bad_override() -> None:
    with pytest.raises(DiffusionError):
        hardware.select_dtype("mps", PipelineKind.SDXL, "int4")


def test_enable_mps_fallback(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    hardware.enable_mps_fallback()
    import os

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
