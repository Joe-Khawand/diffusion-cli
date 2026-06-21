"""Shared Rich presentation for the CLI — a warm "sunset" glow-up.

Import-light: only ``rich`` + stdlib at module load. Heavy bits (torch/PIL via the
``generate`` module, and the Kitty renderer) are imported lazily inside
:func:`run_with_preview`.

Presentation rules around the Kitty graphics protocol:

- During inline-image generation, :class:`~diffusion.utils.terminal_image.KittyRenderer`
  pins a fixed screen region with raw escape sequences. Rich ``Live``/``Progress``/
  ``status`` manage their own cursor region and MUST NOT overlap it. So when previewing
  an image we enhance ONLY the status string passed to ``KittyRenderer.show(status=...)``
  via :func:`status_line` (raw 24-bit ANSI is fine there).
- Rich :func:`make_progress` is used ONLY on the no-Kitty fallback path; :func:`loading_status`
  only during model load, when no image is active.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

from rich.box import ROUNDED
from rich.console import Group
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
from rich.text import Text

from diffusion.utils.console import console

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

    class _SettingsLike(Protocol):
        """Structural type for objects accepted by :func:`settings_table`."""

        steps: int
        width: int
        height: int
        seed: int | None
        negative_prompt: str | None


# --- Warm "sunset" palette -------------------------------------------------
# Gradient stops, gold → coral → rose → plum. Used both for Rich styling and for
# the hand-written 24-bit ANSI status line in the Kitty region.
_SUNSET: tuple[tuple[int, int, int], ...] = (
    (255, 205, 112),  # gold
    (255, 138, 76),   # coral
    (255, 94, 118),   # rose
    (199, 77, 148),   # plum
)
_GOLD = "#ffcd70"
_CORAL = "#ff8a4c"
_ROSE = "#ff5e76"
_PLUM = "#c74d94"

# Block glyphs for progress bars.
_FILL = "▰"
_EMPTY = "▱"

# Big ASCII logo (a figlet-style "DIFFUSION").
_LOGO = (
    r" ___  _  ___ ___ _   _ ___ ___ ___  _  _ ",
    r"|   \| || __| __| | | / __|_ _/ _ \| \| |",
    r"| |) | || _|| _|| |_| \__ \| | (_) | .` |",
    r"|___/|_||_| |_|  \___/|___/___\___/|_|\_|",
)


# --- Color helpers ---------------------------------------------------------
def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _gradient(n: int) -> list[tuple[int, int, int]]:
    """Return ``n`` RGB colors interpolated across the sunset stops."""
    if n <= 0:
        return []
    if n == 1:
        return [_SUNSET[0]]
    segs = len(_SUNSET) - 1
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        pos = i / (n - 1) * segs
        k = min(int(pos), segs - 1)
        out.append(_lerp(_SUNSET[k], _SUNSET[k + 1], pos - k))
    return out


def _hex(c: tuple[int, int, int]) -> str:
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def _ansi(c: tuple[int, int, int]) -> str:
    return f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m"


def _gradient_block(lines: tuple[str, ...]) -> Text:
    """Render multi-line text with a horizontal sunset gradient (stable per column)."""
    width = max(len(line) for line in lines)
    grad = _gradient(width)
    text = Text()
    for li, line in enumerate(lines):
        for col, ch in enumerate(line):
            text.append(ch, style=_hex(grad[col]))
        if li < len(lines) - 1:
            text.append("\n")
    return text


# --- Panels & tables -------------------------------------------------------
def model_ready_panel(repo_id: str, kind: object, device: str, dtype: str) -> Group:
    """Build the startup banner: gradient logo + a "ready" line.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id of the loaded model (shown as the panel subtitle).
    kind : object
        The detected :class:`~diffusion.core.models.PipelineKind` (rendered via ``str``).
    device : str
        Resolved device (``"mps"``, ``"cuda"``, ``"cpu"``).
    dtype : str
        Resolved dtype string (e.g. ``"float16"``).

    Returns
    -------
    rich.console.Group
        The logo panel followed by a colorized ready line.
    """
    logo = Panel(
        _gradient_block(_LOGO),
        box=ROUNDED,
        border_style=_CORAL,
        padding=(0, 2),
        expand=False,
    )
    repo = Text("   ")
    repo.append(repo_id, style="dim")
    ready = Text("   ")
    ready.append(str(kind), style=f"bold {_GOLD}")
    ready.append("  ·  ", style="dim")
    ready.append(device, style=_CORAL)
    ready.append("  ·  ", style="dim")
    ready.append(dtype, style=_ROSE)
    ready.append("      ")
    ready.append("ready to dream ✦", style=f"italic {_PLUM}")
    return Group(logo, repo, ready)


def settings_table(settings: _SettingsLike) -> Table:
    """Build a compact table of the current generation settings.

    Parameters
    ----------
    settings : _SettingsLike
        Any object exposing ``steps``, ``width``, ``height``, ``seed`` and
        ``negative_prompt`` attributes (e.g. the chat ``_Settings`` dataclass).

    Returns
    -------
    rich.table.Table
        A compact two-column table of the active settings.
    """
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column(style=f"bold {_GOLD}")
    seed = "random" if settings.seed is None else str(settings.seed)
    neg = settings.negative_prompt if settings.negative_prompt else "—"
    table.add_row("✦ steps", str(settings.steps))
    table.add_row("✦ size", f"{settings.width}×{settings.height}")
    table.add_row("✦ seed", seed)
    table.add_row("✦ negative", neg)
    return table


def help_panel() -> Panel:
    """Build a panel listing the interactive slash commands.

    Returns
    -------
    rich.panel.Panel
        A panel containing the command reference.
    """
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style=f"bold {_CORAL}")
    table.add_column(style="dim")
    table.add_row("/steps N", "number of denoising steps")
    table.add_row("/size WxH", "image dimensions in pixels")
    table.add_row("/seed N|random", "fix or randomize the seed")
    table.add_row("/neg <text>", "set the negative prompt")
    table.add_row("/help", "show this help")
    table.add_row("/exit", "quit")
    return Panel(
        table,
        title=f"[{_GOLD}]✦ commands[/]",
        subtitle="[dim]type a prompt to generate[/dim]",
        border_style=_PLUM,
        box=ROUNDED,
        expand=False,
    )


def result_panel(path: Path, seed: int | None, steps: int, size: str, elapsed: float) -> Text:
    """Build a one-line "saved" summary for a finished generation.

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
    rich.text.Text
        A colorized, single-line summary of the saved result.
    """
    seed_str = "random" if seed is None else str(seed)
    line = Text("\n")
    line.append("✦ ", style=_GOLD)
    line.append("saved  ", style=f"bold {_CORAL}")
    line.append(str(path), style=_ROSE)
    line.append(
        f"   ·  {size} · {steps} steps · seed {seed_str} · {elapsed:.1f}s",
        style="dim",
    )
    return line


def status_line(step: int, total: int, *, elapsed: float, done: bool = False) -> str:
    """Build a colorized 24-bit ANSI status string for the Kitty preview region.

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
        If True, render the completed state (full bar + "done") instead of an ETA.

    Returns
    -------
    str
        A colorized status string suitable for ``KittyRenderer.show(status=...)``.
    """
    reset = "\x1b[0m"
    dim = "\x1b[2m"
    width = 24
    step = max(0, min(step, total))
    frac = (step / total) if total else 1.0
    filled = round(width * frac)
    grad = _gradient(width)

    bar = "".join(
        (_ansi(grad[i]) + _FILL) if i < filled else (dim + _EMPTY + reset)
        for i in range(width)
    ) + reset
    pct = f"{_ansi(_SUNSET[0])}\x1b[1m{round(frac * 100):>3}%{reset}"

    if done:
        tail = f"{dim}{elapsed:.1f}s{reset}  {_ansi(_SUNSET[0])}✦ done{reset}"
    else:
        remaining = total - step
        eta = (elapsed / step * remaining) if step else 0.0
        tail = f"{dim}eta {eta:.0f}s{reset}  {_ansi(_SUNSET[1])}denoising…{reset}"

    return f"  {bar}  {pct} {dim}{step}/{total}{reset}  {tail}"


def make_progress() -> Progress:
    """Build a configured Rich progress bar for the no-Kitty fallback path.

    Returns
    -------
    rich.progress.Progress
        A progress with spinner, bar, step count, and elapsed/remaining columns.
    """
    return Progress(
        SpinnerColumn(style=_CORAL),
        TextColumn(f"[{_CORAL}]{{task.description}}"),
        BarColumn(complete_style=_CORAL, finished_style=_GOLD, pulse_style=_ROSE),
        TextColumn("[dim]{task.completed}/{task.total}"),
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
    return console.status(f"[bold {_GOLD}]{message}[/]", spinner="dots", spinner_style=_CORAL)


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
        task = progress.add_task("denoising", total=steps)

        def on_step(step_index: int, _image: Image) -> None:
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
            on_preview=on_step,
        )
    elapsed = time.perf_counter() - start
    return image, elapsed
