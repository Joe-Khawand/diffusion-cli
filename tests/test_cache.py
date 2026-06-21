"""Tests for the cache layer. Offline: HF hub calls are mocked."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from diffusion.core import cache
from diffusion.core.models import PipelineKind
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
    cache_info = SimpleNamespace(
        repos=[_fake_repo("org/sdxl", sdxl), _fake_repo("org/llm", llm)]
    )
    mocker.patch("huggingface_hub.scan_cache_dir", return_value=cache_info)

    entries = cache.list_models()
    assert [e.repo_id for e in entries] == ["org/sdxl"]
    assert entries[0].kind is PipelineKind.SDXL

    all_entries = cache.list_models(include_all=True)
    assert {e.repo_id for e in all_entries} == {"org/sdxl", "org/llm"}


def test_get_info_found_and_missing(tmp_path, mocker):
    sd15 = _make_repo_dir(tmp_path, "org/sd15", "model_index_sd15.json")
    cache_info = SimpleNamespace(repos=[_fake_repo("org/sd15", sd15)])
    mocker.patch("huggingface_hub.scan_cache_dir", return_value=cache_info)

    entry = cache.get_info("org/sd15")
    assert entry.kind is PipelineKind.SD15
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


def test_pull_returns_kind(tmp_path, mocker):
    snap = _make_repo_dir(tmp_path, "org/flux", "model_index_flux.json")
    mocker.patch("huggingface_hub.snapshot_download", return_value=str(snap))
    path, kind = cache.pull("org/flux")
    assert path == snap
    assert kind is PipelineKind.FLUX


def test_resolve_local_missing(mocker):
    from huggingface_hub.errors import LocalEntryNotFoundError

    mocker.patch(
        "huggingface_hub.snapshot_download",
        side_effect=LocalEntryNotFoundError("nope"),
    )
    with pytest.raises(ModelNotCachedError):
        cache.resolve_local("org/missing")
