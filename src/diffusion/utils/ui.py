"""Shared Rich presentation for the CLI.

Import-light: only ``rich`` + stdlib at module load. Heavy bits (torch/PIL via the
``generate`` module, and the Kitty renderer) are imported lazily inside
:func:`run_with_preview`.

Presentation rules around the Kitty graphics protocol:

- During inline-image generation, :class:`~diffusion.utils.terminal_image.KittyRenderer`
  pins a fixed screen region with raw escape sequences. Rich ``Live``/``Progress``/
  ``status`` manage their own cursor region and MUST NOT overlap it. So when previewing
  an image we enhance ONLY the status string passed to ``KittyRenderer.show(status=...)``
  via :func:`status_line` (raw ANSI is fine there).
- Rich :func:`make_progress` is used ONLY on the no-Kitty fallback path; :func:`loading_status`
  only during model load, when no image is active.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from diffusion.utils.console import console

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

# Spinner frames for the hand-written (non-Rich) Kitty status line.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Raw ANSI color codes for the Kitty status string (Rich must not touch this region).
_C_CYAN = "\x1b[36m"
_C_GREEN = "\x1b[32m"
_C_DIM = "\x1b[2m"
_C_BOLD = "\x1b[1m"
_C_RESET = "\x1b[0m"


def model_ready_panel(repo_id: str, kind: object, device: str, dtype: str) -> Panel:
    """Build a panel announcing that a model is loaded and ready.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id of the loaded model.
    kind : object
        The detected :class:`~diffusion.core.models.PipelineKind` (rendered via ``str``).
    device : str
        Resolved device (``"mps"``, ``"cuda"``, ``"cpu"``).
    dtype : str
        Resolved dtype string (e.g. ``"float16"``).

    Returns
    -------
    rich.panel.Panel
        A rounded panel summarizing the ready model.
    """
    body = (
        f"[bold]{repo_id}[/bold]  [cyan]{kind}[/cyan]\n"
        f"[dim]device[/dim] [magenta]{device}[/magenta]   "
        f"[dim]dtype[/dim] [magenta]{dtype}[/magenta]"
    )
    return Panel(body, title="[green]✓ ready[/green]", border_style="green", expand=False)


def settings_table(settings: object) -> Table:
    """Build a compact table of the current generation settings.

    Parameters
    ----------
    settings : object
        Any object exposing ``steps``, ``width``, ``height``, ``seed`` and
        ``negative_prompt`` attributes (e.g. the chat ``_Settings`` dataclass).

    Returns
    -------
    rich.table.Table
        A compact two-column table of the active settings.
    """
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column(style="bold")
    seed = "random" if settings.seed is None else str(settings.seed)
    neg = settings.negative_prompt if settings.negative_prompt else "—"
    table.add_row("steps", str(settings.steps))
    table.add_row("size", f"{settings.width}×{settings.height}")
    table.add_row("seed", seed)
    table.add_row("negative", neg)
    return table


def help_panel() -> Panel:
    """Build a panel listing the interactive slash commands.

    Returns
    -------
    rich.panel.Panel
        A panel containing the command reference.
    """
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style="bold cyan")
    table.add_column(style="dim")
    table.add_row("/steps N", "number of denoising steps")
    table.add_row("/size WxH", "image dimensions in pixels")
    table.add_row("/seed N|random", "fix or randomize the seed")
    table.add_row("/neg <text>", "set the negative prompt")
    table.add_row("/help", "show this help")
    table.add_row("/exit", "quit")
    return Panel(
        table,
        title="commands",
        subtitle="[dim]type a prompt to generate[/dim]",
        border_style="cyan",
        expand=False,
    )


def result_panel(path: Path, seed: int | None, steps: int, size: str, elapsed: float) -> Panel:
    """Build a panel summarizing a finished generation.

    Parameters
    ----------
    path : pathlib.Path
        Where the image was saved.
    seed : int or None
        The seed used (``None`` means a random seed was used).
    steps : int
        Number of denoising steps.
    size : str
        Image size as a ``"WxH"`` string.
    elapsed : float
        Wall-clock generation time in seconds.

    Returns
    -------
    rich.panel.Panel
        A panel summarizing the saved result.
    """
    seed_str = "random" if seed is None else str(seed)
    body = (
        f"[bold]{path}[/bold]\n"
        f"[dim]seed[/dim] {seed_str}   [dim]steps[/dim] {steps}   "
        f"[dim]size[/dim] {size}   [dim]time[/dim] {elapsed:.1f}s"
    )
    return Panel(body, title="[green]✓ saved[/green]", border_style="green", expand=False)


def status_line(step: int, total: int, *, elapsed: float, done: bool = False) -> str:
    """Build a colorized ANSI status string for the Kitty preview region.

    This is hand-written ANSI (NOT a Rich ``Live``/``Progress``) because it is drawn
    inside the region pinned by :class:`~diffusion.utils.terminal_image.KittyRenderer`,
    which Rich must not manage.

    Parameters
    ----------
    step : int
        The current step (1-based). Clamped to ``[0, total]``.
    total : int
        The total number of steps.
    elapsed : float
        Wall-clock seconds since generation started.
    done : bool, default False
        If True, render the completed state (``✓`` and full bar) instead of a spinner.

    Returns
    -------
    str
        A colorized status string suitable for ``KittyRenderer.show(status=...)``.
    """
    width = 24
    step = max(0, min(step, total))
    frac = (step / total) if total else 1.0
    filled = round(width * frac)
    bar = f"{_C_GREEN}{'█' * filled}{_C_DIM}{'░' * (width - filled)}{_C_RESET}"
    pct = f"{round(frac * 100)}%"

    if done:
        glyph = f"{_C_GREEN}✓{_C_RESET}"
        tail = f"{_C_DIM}{elapsed:.1f}s{_C_RESET}"
    else:
        glyph = f"{_C_CYAN}{_SPINNER[step % len(_SPINNER)]}{_C_RESET}"
        remaining = total - step
        eta = (elapsed / step * remaining) if step else 0.0
        tail = f"{_C_DIM}{elapsed:.1f}s · eta {eta:.1f}s{_C_RESET}"

    return f"  {glyph} {bar} {_C_BOLD}{pct}{_C_RESET} {step}/{total}  {tail}"


def make_progress() -> Progress:
    """Build a configured Rich progress bar for the no-Kitty fallback path.

    Returns
    -------
    rich.progress.Progress
        A progress with spinner, bar, step count, and elapsed/remaining columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def loading_status(message: str):
    """Return a console status spinner for the model-load phase.

    Parameters
    ----------
    message : str
        The message to display beside the spinner.

    Returns
    -------
    rich.status.Status
        A context manager that shows a spinner while loading.
    """
    return console.status(f"[bold]{message}[/bold]", spinner="dots")


def run_with_preview(
    generate_module,
    pipe,
    kind,
    plan,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    width: int,
    height: int,
    seed: int | None,
    low_mem: bool,
    protocol: str,
    rows: int,
) -> tuple[Image, float]:
    """Run one generation with a live preview, shared by ``chat`` and ``run``.

    On a Kitty-capable terminal this streams the denoising image in a pinned region and
    updates the status string via :func:`status_line`. Otherwise it drives a Rich
    :func:`make_progress` bar (no inline image) from the per-step callback.

    Parameters
    ----------
    generate_module : module
        The ``diffusion.core.generate`` module (passed in to keep this import-light).
    pipe : object
        The already-loaded diffusion pipeline.
    kind : diffusion.core.models.PipelineKind
        The detected pipeline family.
    plan : diffusion.core.models.DeviceInfo
        The resolved device/dtype plan.
    prompt : str
        The text prompt.
    negative_prompt : str or None
        Text describing what to avoid, or None.
    steps : int
        Number of denoising steps.
    width, height : int
        Output image dimensions in pixels.
    seed : int or None
        Random seed, or None for a random seed.
    low_mem : bool
        Whether memory-saving optimizations are active (affects generator device).
    protocol : str
        ``"kitty"`` for inline previews, anything else for the Rich fallback.
    rows : int
        Number of terminal rows for the inline preview region.

    Returns
    -------
    tuple of (PIL.Image.Image, float)
        The final image and the elapsed wall-clock time in seconds.
    """
    start = time.perf_counter()

    if protocol == "kitty":
        from diffusion.utils.terminal_image import KittyRenderer

        renderer = KittyRenderer(rows=rows)

        def on_preview(step_index: int, image: Image) -> None:
            elapsed = time.perf_counter() - start
            renderer.show(image, status=status_line(step_index + 1, steps, elapsed=elapsed))

        image = generate_module.run_inference(
            pipe,
            kind,
            plan,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            low_mem=low_mem,
            on_preview=on_preview,
        )
        elapsed = time.perf_counter() - start
        renderer.show(image, status=status_line(steps, steps, elapsed=elapsed, done=True))
        renderer.finish()
        return image, elapsed

    # Fallback: no inline image, drive a Rich progress bar from the step callback.
    with make_progress() as progress:
        task = progress.add_task("[cyan]denoising[/cyan]", total=steps)

        def on_preview(step_index: int, _image: Image) -> None:
            progress.update(task, completed=step_index + 1)

        image = generate_module.run_inference(
            pipe,
            kind,
            plan,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            low_mem=low_mem,
            on_preview=on_preview,
        )
    elapsed = time.perf_counter() - start
    return image, elapsed
