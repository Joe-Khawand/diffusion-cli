"""Orchestration tests for generate(): diffusers/torch are mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from diffusion.core import registry
from diffusion.core.generate import generate, run_inference
from diffusion.utils.errors import UnsupportedPipelineError

_SDXL = registry.by_class_name("StableDiffusionXLPipeline")
_FLUX = registry.by_class_name("FluxPipeline")


@pytest.fixture
def patched(mocker, tmp_path):
    """Patch the heavy boundaries generate() crosses."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    mocker.patch("diffusion.core.cache.resolve_local", return_value=snapshot)

    fake_image = MagicMock()
    fake_pipe = MagicMock()
    fake_pipe.return_value = MagicMock(images=[fake_image])
    auto = mocker.patch("diffusers.AutoPipelineForText2Image")
    auto.from_pretrained.return_value = fake_pipe

    optimize = mocker.patch("diffusion.core.optimize.apply_optimizations")
    return MagicMock(snapshot=snapshot, image=fake_image, pipe=fake_pipe, optimize=optimize)


def test_generate_saves_image_and_sidecar(mocker, patched, tmp_path):
    mocker.patch("diffusion.core.detect.detect_family", return_value=_SDXL)
    out = tmp_path / "out.png"

    result = generate(
        repo_id="org/sdxl",
        prompt="a cat",
        negative_prompt="blurry",
        steps=10,
        width=512,
        height=512,
        output=out,
        seed=42,
        device_override="cpu",
        dtype_override=None,
        low_mem=False,
    )

    assert result == out
    patched.image.save.assert_called_once_with(out)
    # negative_prompt should be forwarded for SDXL (MagicMock accepts **kwargs)
    _, kwargs = patched.pipe.call_args
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["num_inference_steps"] == 10

    sidecar = out.with_suffix(".png.json")
    meta = json.loads(sidecar.read_text())
    assert meta["repo_id"] == "org/sdxl" and meta["seed"] == 42 and meta["kind"] == "sdxl"
    assert meta["task"] == "text2img"


def test_generate_rejects_unknown_pipeline(mocker, patched, tmp_path):
    mocker.patch("diffusion.core.detect.detect_family", return_value=registry.UNKNOWN)
    with pytest.raises(UnsupportedPipelineError):
        generate(
            repo_id="org/whatever",
            prompt="x",
            negative_prompt=None,
            steps=5,
            width=512,
            height=512,
            output=tmp_path / "o.png",
            seed=None,
            device_override="cpu",
            dtype_override=None,
            low_mem=False,
        )


def test_flux_omits_negative_prompt(mocker, patched, tmp_path):
    mocker.patch("diffusion.core.detect.detect_family", return_value=_FLUX)
    generate(
        repo_id="org/flux",
        prompt="a cat",
        negative_prompt="blurry",
        steps=4,
        width=512,
        height=512,
        output=tmp_path / "o.png",
        seed=None,
        device_override="cpu",
        dtype_override=None,
        low_mem=False,
    )
    _, kwargs = patched.pipe.call_args
    assert "negative_prompt" not in kwargs


class _NarrowPipe:
    """A pipeline whose __call__ only accepts a few keyword args."""

    def __init__(self) -> None:
        self.captured: dict | None = None

    def __call__(self, *, prompt, num_inference_steps, image=None, strength=None, generator=None):
        self.captured = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "image": image,
            "strength": strength,
        }
        return SimpleNamespace(images=["IMG"])


class _CallbackPipe:
    """A pipeline that fires ``callback_on_step_end`` each step, like diffusers.

    ``latents`` (if given) is handed to the callback so preview families can
    project it; None mimics a family without a latent->RGB projection.
    """

    def __init__(self, latents=None) -> None:
        self._latents = latents

    def __call__(
        self,
        *,
        prompt,
        num_inference_steps,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        **kwargs,
    ):
        for i in range(num_inference_steps):
            if callback_on_step_end is not None:
                callback_on_step_end(self, i, 0, {"latents": self._latents})
        return SimpleNamespace(images=["IMG"])


def test_run_inference_reports_progress_without_preview():
    # Regression: families without a latent->RGB projection (e.g. Sana) must still
    # fire the per-step callback so the progress bar advances — previously the
    # callback was gated on a non-None preview, so progress never updated.
    sana = registry.family_by_id("sana")
    assert sana is not None and sana.preview is None

    plan = SimpleNamespace(device="cpu", dtype="float32")
    seen: list[tuple[int, object]] = []
    run_inference(
        _CallbackPipe(),
        sana,
        plan,
        prompt="x",
        negative_prompt=None,
        steps=5,
        width=512,
        height=512,
        seed=None,
        low_mem=False,
        on_preview=lambda i, image: seen.append((i, image)),
    )

    assert [i for i, _ in seen] == [0, 1, 2, 3, 4]  # fired every step
    assert all(image is None for _, image in seen)  # no preview, but progress still ran


def test_run_inference_yields_preview_for_preview_families():
    import torch

    plan = SimpleNamespace(device="cpu", dtype="float32")
    latents = torch.zeros(1, _SDXL.preview.channels, 8, 8)  # (B, C, H, W)
    seen: list[object] = []
    run_inference(
        _CallbackPipe(latents=latents),
        _SDXL,
        plan,
        prompt="x",
        negative_prompt=None,
        steps=3,
        width=512,
        height=512,
        seed=None,
        low_mem=False,
        on_preview=lambda i, image: seen.append(image),
    )

    assert len(seen) == 3 and all(image is not None for image in seen)


def test_run_inference_filters_unsupported_kwargs(mocker):
    # width/height/negative_prompt are dropped because __call__ doesn't accept them;
    # strength is forwarded because it does.
    plan = SimpleNamespace(device="cpu", dtype="float32")
    pipe = _NarrowPipe()
    img = run_inference(
        pipe,
        _SDXL,
        plan,
        prompt="a cat",
        negative_prompt="blurry",
        steps=7,
        width=1024,
        height=1024,
        seed=None,
        low_mem=False,
        force_size=True,
        strength=0.5,
    )
    assert img == "IMG"
    assert pipe.captured == {
        "prompt": "a cat",
        "num_inference_steps": 7,
        "image": None,
        "strength": 0.5,
    }
