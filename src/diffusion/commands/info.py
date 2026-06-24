"""`diffusion info` — show metadata for a cached model."""

from __future__ import annotations

from diffusion.utils.console import console


def run_info(repo_id: str) -> None:
    """Print a table of cached metadata for ``repo_id``.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id of a cached model.
    """
    from rich.table import Table

    from diffusion.core import cache, registry

    entry = cache.get_info(repo_id)

    table = Table(show_header=False, title=f"{repo_id}")
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_row("Pipeline", f"{entry.family.label} ({entry.family.id})")
    table.add_row("Size on disk", entry.size_on_disk_str)
    table.add_row("Inference memory", f"{registry.vram_hint(entry.family)} (fp16, before offload)")
    table.add_row("Components", ", ".join(entry.components) or "—")
    table.add_row("Commit", entry.commit_hash or "—")
    table.add_row("Path", str(entry.local_path))
    if entry.family.memory_heavy:
        table.add_row("Note", "Large model — runs with CPU offload (slow on consumer hardware).")
    console.print(table)
