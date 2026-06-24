"""Tests for the memory guardrail: estimate, dimension validation, and the check."""

from __future__ import annotations

import pytest

from diffusion.core import memory, registry
from diffusion.utils.errors import DiffusionError, InsufficientMemoryError

_SD = registry.by_class_name("StableDiffusionPipeline")

_GB = 1024**3


def _estimate(width: int, height: int) -> int:
    return memory.estimate_peak_activation_bytes(
        width=width, height=height, dtype="float16", steps=25, family=_SD
    )


def test_estimate_monotonic_in_size() -> None:
    assert _estimate(512, 512) < _estimate(768, 768) < _estimate(1024, 1024)


def test_estimate_quadratic_dominates_512_to_1024() -> None:
    # Doubling each dimension quadruples the latent tokens; the attention term
    # (the catastrophic driver) scales ~16x, so total should be well over 10x.
    assert _estimate(1024, 1024) / _estimate(512, 512) > 10


def test_estimate_ignores_steps() -> None:
    low = memory.estimate_peak_activation_bytes(
        width=512, height=512, dtype="float16", steps=5, family=_SD
    )
    high = memory.estimate_peak_activation_bytes(
        width=512, height=512, dtype="float16", steps=200, family=_SD
    )
    assert low == high


@pytest.mark.parametrize("dim", [512, 768, 1024, 1000])
def test_validate_dimensions_accepts_multiples_of_8(dim: int) -> None:
    memory.validate_dimensions(dim, dim)  # should not raise


@pytest.mark.parametrize("bad", [700, 513, 1])
def test_validate_dimensions_rejects_non_multiples_of_8(bad: int) -> None:
    with pytest.raises(DiffusionError):
        memory.validate_dimensions(bad, 512)


@pytest.mark.parametrize("bad", [0, -8])
def test_validate_dimensions_rejects_non_positive(bad: int) -> None:
    with pytest.raises(DiffusionError):
        memory.validate_dimensions(bad, 512)


def test_check_memory_raises_when_oversized(mocker) -> None:
    mocker.patch.object(memory, "available_bytes", return_value=1 * _GB)
    with pytest.raises(InsufficientMemoryError):
        memory.check_memory(
            width=1024, height=1024, dtype="float16", steps=25, family=_SD, device="mps"
        )


def test_check_memory_passes_for_normal_size(mocker) -> None:
    mocker.patch.object(memory, "available_bytes", return_value=16 * _GB)
    memory.check_memory(
        width=512, height=512, dtype="float16", steps=25, family=_SD, device="mps"
    )  # should not raise


def test_check_memory_degrades_when_unknown(mocker) -> None:
    mocker.patch.object(memory, "available_bytes", return_value=None)
    memory.check_memory(
        width=4096, height=4096, dtype="float16", steps=25, family=_SD, device="mps"
    )  # unknown memory → don't block


def test_check_memory_bypassed_by_env(mocker, monkeypatch) -> None:
    monkeypatch.setenv("DIFFUSION_SKIP_MEM_CHECK", "1")
    avail = mocker.patch.object(memory, "available_bytes", return_value=1 * _GB)
    memory.check_memory(
        width=1024, height=1024, dtype="float16", steps=25, family=_SD, device="mps"
    )  # should not raise
    avail.assert_not_called()


def test_estimate_runtime_peak_monotonic_in_size() -> None:
    def peak(dim: int) -> int:
        return memory.estimate_runtime_peak_bytes(
            weight_bytes=2 * _GB,
            weight_precision="fp16",
            width=dim,
            height=dim,
            family=_SD,
            device="cpu",
        )

    assert peak(512) < peak(768) < peak(1024)


def test_estimate_runtime_peak_includes_weights() -> None:
    # The weights term should dominate at small sizes; fp16 weights on a cpu
    # (float32 load) roughly double in memory.
    peak = memory.estimate_runtime_peak_bytes(
        weight_bytes=2 * _GB,
        weight_precision="fp16",
        width=512,
        height=512,
        family=_SD,
        device="cpu",
    )
    assert peak > 3 * _GB  # ~4 GB resident weights (fp16→fp32) + activations


def test_available_bytes_uses_psutil_for_mps(mocker) -> None:
    fake = mocker.Mock(available=8 * _GB)
    mocker.patch("psutil.virtual_memory", return_value=fake)
    assert memory.available_bytes("mps") == 8 * _GB


def test_available_bytes_returns_none_on_failure(mocker) -> None:
    mocker.patch("psutil.virtual_memory", side_effect=RuntimeError("boom"))
    assert memory.available_bytes("cpu") is None
