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
        strength=0.5,
    )
    assert img == "IMG"
    assert pipe.captured == {
        "prompt": "a cat",
        "num_inference_steps": 7,
        "image": None,
        "strength": 0.5,
    }
