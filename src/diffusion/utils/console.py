"""Shared rich console. Import-light (rich only)."""

from __future__ import annotations

from rich.console import Console

console = Console()
err_console = Console(stderr=True)
