"""`diffusion pull` — download a model into the cache."""

from __future__ import annotations

from diffusion.utils.console import console


def run_pull(repo_id: str) -> None:
    """Download ``repo_id`` into the local cache and report the detected pipeline.

    Detects the family first (fetching only ``model_index.json``) so non-diffusion
    repos are rejected before a multi-gigabyte download.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id to download.
    """
    from diffusion.core import cache

    console.print(f"Inspecting [bold]{repo_id}[/bold] …")
    family = cache.peek_family(repo_id)
    if not family.supported:
        console.print(
            f"[yellow]![/yellow] [bold]{repo_id}[/bold] has no diffusers "
            "'model_index.json' — it is not a runnable image pipeline. Skipping download."
        )
        return

    suffix = " [yellow](large; runs with CPU offload)[/yellow]" if family.memory_heavy else ""
    console.print(f"Detected [cyan]{family.label}[/cyan]{suffix}. Downloading …")
    path, _ = cache.pull(repo_id)
    console.print(
        f"[green]✓[/green] Downloaded [bold]{repo_id}[/bold] — [cyan]{family.label}[/cyan]"
    )
    console.print(f"[dim]{path}[/dim]")
