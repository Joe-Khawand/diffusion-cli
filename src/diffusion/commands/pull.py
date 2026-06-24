"""`diffusion pull` — download a model into the cache."""

from __future__ import annotations

from diffusion.utils.console import console


def run_pull(repo_id: str, *, variant: str | None = None) -> None:
    """Download ``repo_id`` into the local cache and report the detected pipeline.

    Detects the family first (fetching only ``model_index.json``) so non-diffusion
    repos are rejected before a multi-gigabyte download, then lets the user choose
    a precision variant (fp16/bf16/fp32) by size before downloading.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id to download, or a family slug (e.g. ``"sdxl"``)
        to pull that family's example model.
    variant : str or None
        Precision variant to download (``"fp16"``/``"bf16"``/``"fp32"``). None
        shows an interactive picker (TTY) or uses the recommended default.
    """
    from diffusion.commands import variants as variants_cmd
    from diffusion.core import cache, hardware, registry
    from diffusion.core import variants as variants_core

    resolved = registry.resolve_repo(repo_id)
    if resolved != repo_id:
        console.print(
            f"[dim]'{repo_id}' is a family slug → pulling its example [bold]{resolved}[/bold][/dim]"
        )
        repo_id = resolved

    console.print(f"Inspecting [bold]{repo_id}[/bold] …")
    family = cache.peek_family(repo_id)
    if not family.supported:
        console.print(
            f"[yellow]![/yellow] [bold]{repo_id}[/bold] has no diffusers "
            "'model_index.json' — it is not a runnable image pipeline. Skipping download."
        )
        return

    suffix = " [yellow](large; runs with CPU offload)[/yellow]" if family.memory_heavy else ""
    console.print(f"Detected [cyan]{family.label}[/cyan]{suffix}.")

    # Pick a precision variant. If listing fails (offline/API error), fall back to
    # the default lean download without a prompt.
    chosen = None
    try:
        found = variants_core.list_variants(repo_id)
    except Exception:
        found = []
    if found:
        chosen = variants_cmd.choose_variant(
            found,
            requested=variant,
            repo_id=repo_id,
            family=family,
            device=hardware.detect_device(),
        )
        console.print(
            f"Downloading [cyan]{chosen.label}[/cyan] "
            f"(~{cache.human_size(chosen.download_bytes)}) …"
        )
    else:
        console.print("Downloading …")

    path, _ = cache.pull(repo_id, variant=chosen.precision if chosen else variant)
    console.print(
        f"[green]✓[/green] Downloaded [bold]{repo_id}[/bold] — [cyan]{family.label}[/cyan]"
    )
    console.print(f"[dim]{path}[/dim]")
