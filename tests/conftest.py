"""Shared pytest fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def make_snapshot(tmp_path: Path):
    """Build a fake model snapshot dir from fixture JSON files.

    Usage::

        snap = make_snapshot(model_index="model_index_sdxl.json")
        snap = make_snapshot(model_index="...", unet_config="config_unet_sdxl.json")
    """

    def _make(model_index: str | None = None, unet_config: str | None = None) -> Path:
        snap = tmp_path / "snapshot"
        snap.mkdir(exist_ok=True)
        if model_index is not None:
            shutil.copy(FIXTURES / model_index, snap / "model_index.json")
        if unet_config is not None:
            (snap / "unet").mkdir(exist_ok=True)
            shutil.copy(FIXTURES / unet_config, snap / "unet" / "config.json")
        return snap

    return _make


def load_fixture(name: str) -> dict:
    with (FIXTURES / name).open() as fh:
        return json.load(fh)
