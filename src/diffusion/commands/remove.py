"""`diffusion remove` — delete a cached model."""

from __future__ import annotations

import typer

from diffusion.utils.console import console


def run_remove(repo_id: str, *, yes: bool = False) -> None:
    """Delete a cached model, prompting for confirmation unless ``yes`` is set.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id of the cached model to delete.
    yes : bool, default False
        If True, skip the interactive confirmation prompt.
    """
    from diffusion.core import cache

    # Verify it exists (raises ModelNotCachedError otherwise) before prompting.
    entry = cache.get_info(repo_id)

    if not yes:
        confirmed = typer.confirm(f"Delete '{repo_id}' ({entry.size_on_disk_str}) from the cache?")
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            return

    freed = cache.remove(repo_id)
    console.print(
        f"[green]✓[/green] Removed [bold]{repo_id}[/bold] (freed {cache._human_size(freed)})"
    )
