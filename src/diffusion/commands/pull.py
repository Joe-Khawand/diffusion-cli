"""`diffusion pull` — download a model into the cache."""

from __future__ import annotations

from diffusion.utils.console import console


def run_pull(repo_id: str) -> None:
    """Download ``repo_id`` into the local cache and report the detected pipeline.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id to download.
    """
    from diffusion.core import cache

    console.print(f"Pulling [bold]{repo_id}[/bold] …")
    path, kind = cache.pull(repo_id)
    if kind.is_supported:
        suffix = " [yellow](large; runs with CPU offload)[/yellow]" if kind.is_memory_heavy else ""
        console.print(
            f"[green]✓[/green] Downloaded [bold]{repo_id}[/bold] — "
            f"detected [cyan]{kind}[/cyan]{suffix}"
        )
    else:
        console.print(
            f"[yellow]![/yellow] Downloaded [bold]{repo_id}[/bold], but it is not a recognized "
            "text-to-image pipeline. 'diffusion run' may not work."
        )
    console.print(f"[dim]{path}[/dim]")
