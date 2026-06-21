"""CLI plumbing tests for the metadata commands (cache layer mocked)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from diffusion.cli import app
from diffusion.core.models import ModelEntry, PipelineKind
from diffusion.utils.errors import ModelNotCachedError

runner = CliRunner()


def _entry(repo_id: str) -> ModelEntry:
    return ModelEntry(
        repo_id=repo_id,
        kind=PipelineKind.SDXL,
        size_on_disk=6_900_000_000,
        size_on_disk_str="6.9G",
        last_modified=1.0,
        commit_hash="abc123",
        local_path=Path("/cache") / repo_id,
        components=["unet", "vae"],
    )


def test_list_shows_models(mocker) -> None:
    mocker.patch("diffusion.core.cache.list_models", return_value=[_entry("org/sdxl")])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "org/sdxl" in result.output


def test_info_shows_metadata(mocker) -> None:
    mocker.patch("diffusion.core.cache.get_info", return_value=_entry("org/sdxl"))
    result = runner.invoke(app, ["info", "org/sdxl"])
    assert result.exit_code == 0
    assert "sdxl" in result.output and "unet" in result.output


def test_info_missing_exits_nonzero(mocker) -> None:
    mocker.patch(
        "diffusion.core.cache.get_info", side_effect=ModelNotCachedError("org/missing")
    )
    # entrypoint() maps DiffusionError to a clean exit; invoke it via the wrapper.
    from diffusion.cli import entrypoint

    mocker.patch("sys.argv", ["diffusion", "info", "org/missing"])
    try:
        entrypoint()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit")


def test_remove_with_yes(mocker) -> None:
    mocker.patch("diffusion.core.cache.get_info", return_value=_entry("org/sdxl"))
    remove = mocker.patch("diffusion.core.cache.remove", return_value=6_900_000_000)
    result = runner.invoke(app, ["remove", "org/sdxl", "--yes"])
    assert result.exit_code == 0
    remove.assert_called_once_with("org/sdxl")
