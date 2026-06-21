"""`diffusion list` — show cached models."""

from __future__ import annotations

from diffusion.utils.console import console


def run_list(*, all_models: bool = False) -> None:
    from rich.table import Table

    from diffusion.core import cache

    entries = cache.list_models(include_all=all_models)
    if not entries:
        console.print("[dim]No cached models found. Try 'diffusion pull <repo_id>'.[/dim]")
        return

    table = Table(title="Cached models")
    table.add_column("Repo", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Size", justify="right")
    for entry in entries:
        table.add_row(entry.repo_id, str(entry.kind), entry.size_on_disk_str)
    console.print(table)
