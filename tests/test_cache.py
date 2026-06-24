"""Tests for the cache layer. Offline: HF hub calls are mocked."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from diffusion.core import cache, registry
from diffusion.utils.errors import ModelNotCachedError

FIXTURES = Path(__file__).parent / "fixtures"


def _make_repo_dir(tmp_path: Path, repo_id: str, model_index: str | None) -> Path:
    snap = tmp_path / repo_id.replace("/", "--") / "snapshot"
    snap.mkdir(parents=True)
    if model_index is not None:
        shutil.copy(FIXTURES / model_index, snap / "model_index.json")
    return snap


def _fake_repo(repo_id: str, snapshot_path: Path, *, repo_type="model", size=1234, commit="abc123"):
    revision = SimpleNamespace(
        commit_hash=commit, snapshot_path=str(snapshot_path), last_modified=1.0, size_on_disk=size
    )
    return SimpleNamespace(
        repo_id=repo_id,
        repo_type=repo_type,
        repo_path=snapshot_path.parent,
        size_on_disk=size,
        size_on_disk_str="1.2K",
        last_modified=1.0,
        revisions=[revision],
    )


def test_list_models_filters_non_diffusion(tmp_path, mocker):
    sdxl = _make_repo_dir(tmp_path, "org/sdxl", "model_index_sdxl.json")
    llm = _make_repo_dir(tmp_path, "org/llm", None)  # no model_index.json
    cache_info = SimpleNamespace(repos=[_fake_repo("org/sdxl", sdxl), _fake_repo("org/llm", llm)])
    mocker.patch("huggingface_hub.scan_cache_dir", return_value=cache_info)

    entries = cache.list_models()
    assert [e.repo_id for e in entries] == ["org/sdxl"]
    assert entries[0].family.id == "sdxl"

    all_entries = cache.list_models(include_all=True)
    assert {e.repo_id for e in all_entries} == {"org/sdxl", "org/llm"}


def test_get_info_found_and_missing(tmp_path, mocker):
    sd15 = _make_repo_dir(tmp_path, "org/sd15", "model_index_sd15.json")
    cache_info = SimpleNamespace(repos=[_fake_repo("org/sd15", sd15)])
    mocker.patch("huggingface_hub.scan_cache_dir", return_value=cache_info)

    entry = cache.get_info("org/sd15")
    assert entry.family.id == "sd1.5"
    assert "unet" in entry.components

    with pytest.raises(ModelNotCachedError):
        cache.get_info("org/missing")


def test_remove_calls_delete_revisions(tmp_path, mocker):
    sd15 = _make_repo_dir(tmp_path, "org/sd15", "model_index_sd15.json")
    execute = mocker.Mock()
    delete_revisions = mocker.Mock(return_value=SimpleNamespace(execute=execute))
    cache_info = SimpleNamespace(
        repos=[_fake_repo("org/sd15", sd15, commit="deadbeef")],
        delete_revisions=delete_revisions,
    )
    mocker.patch("huggingface_hub.scan_cache_dir", return_value=cache_info)

    freed = cache.remove("org/sd15")
    delete_revisions.assert_called_once_with("deadbeef")
    execute.assert_called_once()
    assert freed == 1234


def test_remove_missing_raises(mocker):
    cache_info = SimpleNamespace(repos=[], delete_revisions=mocker.Mock())
    mocker.patch("huggingface_hub.scan_cache_dir", return_value=cache_info)
    with pytest.raises(ModelNotCachedError):
        cache.remove("org/missing")


def test_pull_returns_family(tmp_path, mocker):
    snap = _make_repo_dir(tmp_path, "org/flux", "model_index_flux.json")
    mocker.patch("huggingface_hub.HfApi.list_repo_files", return_value=["model_index.json"])
    mocker.patch("huggingface_hub.snapshot_download", return_value=str(snap))
    path, family = cache.pull("org/flux")
    assert path == snap
    assert family.id == "flux"


# --- Lean download selection (the SD 1.5 ~35 GB → few GB fix) ---------------
_SD15_REPO_FILES = [
    "model_index.json",
    "safety_checker/config.json",
    "safety_checker/model.fp16.safetensors",
    "safety_checker/model.safetensors",
    "safety_checker/pytorch_model.bin",
    "safety_checker/pytorch_model.fp16.bin",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.safetensors",
    "text_encoder/pytorch_model.bin",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.non_ema.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "v1-5-pruned.safetensors",
    "v1-5-pruned-emaonly.safetensors",
    "model.ckpt",
    "README.md",
]


def test_select_download_files_picks_one_fp16_variant_per_component():
    selected = set(cache.select_files(_SD15_REPO_FILES))
    weights = {f for f in selected if f.endswith((".safetensors", ".bin"))}
    assert weights == {
        "safety_checker/model.fp16.safetensors",
        "text_encoder/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    }


def test_select_download_files_keeps_all_metadata():
    selected = set(cache.select_files(_SD15_REPO_FILES))
    assert {"model_index.json", "unet/config.json", "tokenizer/merges.txt"} <= selected
    assert "README.md" not in selected  # non-config files are dropped


def test_select_download_files_drops_bloat():
    selected = set(cache.select_files(_SD15_REPO_FILES))
    for dropped in (
        "v1-5-pruned.safetensors",  # top-level single-file checkpoint
        "model.ckpt",  # single-file ckpt format
        "unet/diffusion_pytorch_model.non_ema.safetensors",  # training-only EMA weights
        "unet/diffusion_pytorch_model.bin",  # .bin when safetensors exists
        "unet/diffusion_pytorch_model.safetensors",  # fp32 when fp16 exists
    ):
        assert dropped not in selected


def test_select_component_falls_back_when_no_fp16():
    files = ["text_encoder/config.json", "text_encoder/model.safetensors"]
    assert set(cache.select_files(files)) == {
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
    }


def test_select_component_uses_bin_when_no_safetensors():
    files = ["text_encoder/config.json", "text_encoder/pytorch_model.bin"]
    assert "text_encoder/pytorch_model.bin" in cache.select_files(files)


def test_resolve_allow_patterns_falls_back_on_api_error(mocker):
    mocker.patch("huggingface_hub.HfApi.list_repo_files", side_effect=RuntimeError("offline"))
    assert cache._resolve_allow_patterns("org/x") == cache._ALLOW_PATTERNS


def test_detect_variant_fp16(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")
    assert cache.detect_variant(tmp_path) == "fp16"


def test_detect_variant_bf16(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.bf16.safetensors").write_bytes(b"")
    assert cache.detect_variant(tmp_path) == "bf16"


def test_detect_variant_none(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    assert cache.detect_variant(tmp_path) is None


# --- Per-variant selection (drives `pull --variant` and the variants listing) ---
_MULTI_PREC_FILES = [
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.bf16.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",  # only fp16 here
]


def test_available_precisions_order():
    assert cache.available_precisions(_MULTI_PREC_FILES) == ["fp16", "bf16", "fp32"]


def test_select_files_fp32_picks_plain_weights():
    weights = {f for f in cache.select_files(_MULTI_PREC_FILES, variant="fp32") if "model" in f}
    assert weights == {
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",  # vae lacks fp32 → falls back
    }


def test_select_files_bf16_picks_bf16():
    weights = {f for f in cache.select_files(_MULTI_PREC_FILES, variant="bf16") if "model" in f}
    assert "unet/diffusion_pytorch_model.bf16.safetensors" in weights


def test_peek_family_reads_only_model_index(tmp_path, mocker):
    snap = _make_repo_dir(tmp_path, "org/sdxl", "model_index_sdxl.json")
    mocker.patch("huggingface_hub.hf_hub_download", return_value=str(snap / "model_index.json"))
    assert cache.peek_family("org/sdxl").id == "sdxl"


def test_peek_family_missing_index_is_unknown(mocker):
    from huggingface_hub.errors import EntryNotFoundError

    mocker.patch("huggingface_hub.hf_hub_download", side_effect=EntryNotFoundError("nope"))
    assert cache.peek_family("org/llm") is registry.UNKNOWN


def test_resolve_local_missing(mocker):
    from huggingface_hub.errors import LocalEntryNotFoundError

    mocker.patch(
        "huggingface_hub.snapshot_download",
        side_effect=LocalEntryNotFoundError("nope"),
    )
    with pytest.raises(ModelNotCachedError):
        cache.resolve_local("org/missing")
