"""`diffusion variants` — list a repo's downloadable precision variants.

Also provides the interactive picker reused by ``diffusion pull`` so the user can
weigh download size and per-image memory before committing to a download.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from diffusion.utils.console import console

if TYPE_CHECKING:
    from diffusion.core.models import ModelFamily
    from diffusion.core.variants import Variant

# Image sizes shown in the memory readout (square; the common text2img presets).
_PREVIEW_SIZES = (512, 768, 1024)


def run_variants(repo_id: str) -> None:
    """Print the precision variants ``repo_id`` ships, with sizes and memory.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo id, or a family slug (e.g. ``"sdxl"``).
    """
    from diffusion.core import cache, hardware, registry, variants

    repo_id = registry.resolve_repo(repo_id)
    family = cache.peek_family(repo_id)
    if not family.supported:
        console.print(
            f"[yellow]![/yellow] [bold]{repo_id}[/bold] is not a runnable image pipeline."
        )
        return

    try:
        found = variants.list_variants(repo_id)
    except Exception:
        console.print(
            f"[yellow]Could not list files for {repo_id}.[/yellow] "
            "Check your connection (or SSL_CERT_FILE behind a proxy)."
        )
        return
    if not found:
        console.print(f"[yellow]No downloadable weight variants found for {repo_id}.[/yellow]")
        return
    device = hardware.detect_device()
    render_variants(repo_id, family, found, device)


def render_variants(repo_id: str, family: ModelFamily, found: list[Variant], device: str) -> None:
    """Print the variants table and a device-aware peak-memory readout."""
    from rich.table import Table

    from diffusion.core import cache, hardware, memory

    table = Table(title=f"{repo_id} — {family.label}")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Variant")
    table.add_column("Download", justify="right")
    table.add_column("", style="dim")  # recommended marker
    for i, v in enumerate(found, start=1):
        table.add_row(
            str(i),
            v.label,
            cache.human_size(v.download_bytes),
            "★ recommended" if v.recommended else "",
        )
    console.print(table)

    # Memory readout uses the recommended variant's weight size; runtime memory is
    # set by the device load dtype, so it's ~the same whichever variant you pick.
    ref = next((v for v in found if v.recommended), found[0])
    load_dtype = hardware.select_dtype(device, family)
    avail = memory.available_bytes(device)

    mem = Table(title=f"Estimated peak memory on {device} ({load_dtype})")
    mem.add_column("Image size")
    mem.add_column("Peak", justify="right")
    mem.add_column("Fits?" if avail is not None else "")
    for size in _PREVIEW_SIZES:
        peak = memory.estimate_runtime_peak_bytes(
            weight_bytes=ref.weight_bytes,
            weight_precision=ref.precision,
            width=size,
            height=size,
            family=family,
            device=device,
        )
        fits = ""
        if avail is not None:
            fits = "[green]✓[/green]" if peak <= avail else "[red]✗ exceeds free memory[/red]"
        mem.add_row(f"{size}×{size}", cache.human_size(peak), fits)
    console.print(mem)
    if avail is not None:
        console.print(f"[dim]~{cache.human_size(avail)} free now. [/dim]", end="")
    console.print(
        "[dim]Runtime memory is set by the device dtype (similar across variants); "
        "--low-mem lowers it via offload.[/dim]"
    )


def choose_variant(
    found: list[Variant],
    *,
    requested: str | None,
    repo_id: str,
    family: ModelFamily,
    device: str,
) -> Variant:
    """Pick a variant: honor ``requested``, else prompt (TTY) or use the default.

    Parameters
    ----------
    found : list of Variant
        Variants reported by :func:`diffusion.core.variants.list_variants`.
    requested : str or None
        The ``--variant`` flag value (e.g. ``"fp16"``), or None.
    repo_id : str
        Repo id, used for the rendered table title.
    family : ModelFamily
        Detected family, used for the memory readout.
    device : str
        Resolved device, used for the memory readout.

    Returns
    -------
    Variant
        The chosen variant.
    """
    from rich.prompt import IntPrompt

    from diffusion.utils.errors import DiffusionError

    recommended = next((v for v in found if v.recommended), found[0])

    if requested is not None:
        match = next((v for v in found if v.precision == requested), None)
        if match is None:
            options = ", ".join(v.precision for v in found)
            raise DiffusionError(
                f"Variant '{requested}' is not available for this model.",
                hint=f"Available: {options}.",
            )
        return match

    if len(found) == 1 or not sys.stdin.isatty():
        return recommended

    render_variants(repo_id, family, found, device)
    default_index = found.index(recommended) + 1
    choice = IntPrompt.ask(
        "Download which variant?",
        choices=[str(i) for i in range(1, len(found) + 1)],
        default=default_index,
    )
    return found[choice - 1]
