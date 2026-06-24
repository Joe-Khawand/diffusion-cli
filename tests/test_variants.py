"""Tests for variant enumeration and the pull-time picker. Offline: HF is mocked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from diffusion.commands import variants as variants_cmd
from diffusion.core import registry
from diffusion.core import variants as variants_core
from diffusion.utils.errors import DiffusionError

_SD15 = registry.require("StableDiffusionPipeline")

# A synthetic repo tree: fp16 + fp32 in safetensors, plus a folder entry (no size).
_TREE = [
    SimpleNamespace(path="model_index.json", size=500),
    SimpleNamespace(path="unet", size=None),  # RepoFolder: skipped
    SimpleNamespace(path="unet/config.json", size=700),
    SimpleNamespace(path="unet/diffusion_pytorch_model.fp16.safetensors", size=1_700_000_000),
    SimpleNamespace(path="unet/diffusion_pytorch_model.safetensors", size=3_400_000_000),
    SimpleNamespace(path="vae/config.json", size=600),
    SimpleNamespace(path="vae/diffusion_pytorch_model.fp16.safetensors", size=160_000_000),
    SimpleNamespace(path="vae/diffusion_pytorch_model.safetensors", size=320_000_000),
]


def _patch_tree(mocker, tree):
    mocker.patch("huggingface_hub.HfApi.list_repo_tree", return_value=tree)


def test_list_variants_enumerates_fp16_and_fp32(mocker):
    _patch_tree(mocker, _TREE)
    found = variants_core.list_variants("org/sd15")
    assert [v.precision for v in found] == ["fp16", "fp32"]  # leanest-first


def test_list_variants_sizes_and_recommendation(mocker):
    _patch_tree(mocker, _TREE)
    by_prec = {v.precision: v for v in variants_core.list_variants("org/sd15")}

    assert by_prec["fp16"].weight_bytes == 1_700_000_000 + 160_000_000
    assert by_prec["fp32"].weight_bytes == 3_400_000_000 + 320_000_000
    # download adds the tiny metadata (model_index + both configs) on top of weights.
    assert by_prec["fp16"].download_bytes == by_prec["fp16"].weight_bytes + 500 + 700 + 600

    assert by_prec["fp16"].recommended is True  # fp16 present → recommended
    assert by_prec["fp32"].recommended is False
    assert by_prec["fp16"].diffusers_variant == "fp16"
    assert by_prec["fp32"].diffusers_variant is None


def test_list_variants_single_precision(mocker):
    _patch_tree(
        mocker,
        [
            SimpleNamespace(path="unet/config.json", size=700),
            SimpleNamespace(path="unet/diffusion_pytorch_model.safetensors", size=3_400_000_000),
        ],
    )
    found = variants_core.list_variants("org/fp32only")
    assert [v.precision for v in found] == ["fp32"]
    assert found[0].recommended is True


def _variants():
    return [
        variants_core.Variant("fp16", "fp16", 1_800_000_000, 1_800_001_000, "fp16", True),
        variants_core.Variant("fp32", "fp32", 3_700_000_000, 3_700_001_000, None, False),
    ]


def test_choose_variant_honors_requested():
    chosen = variants_cmd.choose_variant(
        _variants(), requested="fp32", repo_id="org/x", family=_SD15, device="cpu"
    )
    assert chosen.precision == "fp32"


def test_choose_variant_unknown_requested_raises():
    with pytest.raises(DiffusionError) as exc:
        variants_cmd.choose_variant(
            _variants(), requested="int4", repo_id="org/x", family=_SD15, device="cpu"
        )
    assert "int4" in exc.value.message


def test_choose_variant_non_tty_returns_recommended(mocker):
    mocker.patch("sys.stdin.isatty", return_value=False)
    chosen = variants_cmd.choose_variant(
        _variants(), requested=None, repo_id="org/x", family=_SD15, device="cpu"
    )
    assert chosen.precision == "fp16"


def test_choose_variant_single_skips_prompt(mocker):
    one = [_variants()[0]]
    # Even on a TTY, a single variant should not prompt.
    mocker.patch("sys.stdin.isatty", return_value=True)
    chosen = variants_cmd.choose_variant(
        one, requested=None, repo_id="org/x", family=_SD15, device="cpu"
    )
    assert chosen.precision == "fp16"
