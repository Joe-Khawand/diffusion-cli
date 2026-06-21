"""Tests for the shared UI presentation helpers. Pure, no GPU."""

from __future__ import annotations

import re

from diffusion.utils import ui

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI.sub("", text)


def test_status_line_half_progress() -> None:
    line = ui.status_line(5, 10, elapsed=2.0)
    plain = _strip(line)
    # 24-wide bar, half filled.
    assert plain.count("█") == 12
    assert plain.count("░") == 12
    assert "50%" in plain
    assert "5/10" in plain
    # Running state: a spinner glyph, not the done check.
    assert "✓" not in plain
    assert any(g in plain for g in ui._SPINNER)


def test_status_line_done_state() -> None:
    line = ui.status_line(10, 10, elapsed=3.5, done=True)
    plain = _strip(line)
    assert "✓" in plain
    assert "100%" in plain
    assert "10/10" in plain
    assert plain.count("█") == 24
    assert plain.count("░") == 0


def test_status_line_clamps_and_eta() -> None:
    # Step beyond total is clamped; percentage caps at 100.
    line = ui.status_line(15, 10, elapsed=1.0)
    plain = _strip(line)
    assert "100%" in plain
    assert "10/10" in plain


def test_status_line_zero_total_does_not_crash() -> None:
    line = ui.status_line(0, 0, elapsed=0.0)
    plain = _strip(line)
    assert "0/0" in plain
    assert "100%" in plain  # frac defaults to 1.0 when total is 0


def test_status_line_eta_when_running() -> None:
    # elapsed/step * remaining = 2/2 * 8 = 8.0s
    line = ui.status_line(2, 10, elapsed=2.0)
    plain = _strip(line)
    assert "eta 8.0s" in plain


def test_make_progress_is_a_progress() -> None:
    from rich.progress import Progress

    assert isinstance(ui.make_progress(), Progress)
