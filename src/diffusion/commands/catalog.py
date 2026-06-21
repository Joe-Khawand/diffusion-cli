"""`diffusion catalog` — list supported families and example repos to pull."""

from __future__ import annotations

from diffusion.utils.console import console, suppress_transformers_docstring_noise


def run_catalog() -> None:
    """Print the curated model families and example HuggingFace repo ids.

    Only families whose pipeline classes exist in the *installed* diffusers are
    shown, so the catalog never advertises a model this install cannot run. Any
    other diffusers text-to-image repo also works via auto-detection.
    """
    from rich.table import Table

    from diffusion.core import registry

    # Importing diffusers (to check availability) prints cosmetic docstring noise.
    with suppress_transformers_docstring_noise():
        families = registry.available_families()
        hidden = len(registry.FAMILIES) - len(families)

    table = Table(title="Supported model families")
    table.add_column("Family", style="bold")
    table.add_column("Slug", style="cyan")
    table.add_column("Notes")
    table.add_column("Example repo to pull", style="green")

    for fam in families:
        notes = []
        if fam.memory_heavy:
            notes.append("large/offload")
        if fam.preview is not None:
            notes.append("live preview")
        if not fam.supports_negative_prompt:
            notes.append("no negative prompt")
        example = fam.example_repos[0] if fam.example_repos else "—"
        table.add_row(fam.label, fam.id, ", ".join(notes) or "—", example)

    console.print(table)
    if hidden:
        console.print(
            f"[dim]{hidden} more family(ies) are not available in this diffusers "
            "version and were hidden. Upgrade diffusers to enable them.[/dim]"
        )
    console.print(
        "[dim]Any other diffusers text-to-image repo also works via auto-detection. "
        "Pull one with 'diffusion pull <repo_id>'.[/dim]"
    )
