"""Guard the fast-startup property: `diffusion --help` must not import torch."""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from diffusion.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("pull", "run", "chat", "serve", "list", "info", "remove", "catalog"):
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "diffusion" in result.output


def test_importing_cli_does_not_import_torch() -> None:
    # Subprocess so we measure a clean import, not one polluted by the test session.
    code = (
        "import sys; import diffusion.cli; "
        "assert 'torch' not in sys.modules, 'torch imported at startup'; "
        "assert 'diffusers' not in sys.modules, 'diffusers imported at startup'; "
        "assert 'fastapi' not in sys.modules, 'fastapi imported at startup'; "
        "assert 'uvicorn' not in sys.modules, 'uvicorn imported at startup'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
