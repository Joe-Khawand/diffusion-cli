"""Inline image rendering in the terminal via the Kitty graphics protocol.

Supported by Ghostty and Kitty. We transmit PNG bytes base64-encoded in 4 KB
chunks. For animation (e.g. denoising steps) the renderer pins a fixed region:
it hides the cursor, reserves rows, saves an anchor, and on each frame restores
to the anchor, repaints a status line, and redraws the image with the Kitty
"do not move cursor" policy (C=1) so nothing drifts.

The region, top to bottom, is::

    status line
    <gap blank lines>
    [ top border ]
    [ │ ] image rows [ │ ]
    [ bottom border ]

When a ``border_palette`` is supplied, an animated box is drawn around the image:
the image is scaled to exactly ``rows`` by ``cols`` cells, so a text frame of
``cols + 2`` by ``rows + 2`` aligns to the cell grid. The border colors "march" by
one step each frame for a flowing effect.

References: https://sw.kovidgoyal.net/kitty/graphics-protocol/
"""

from __future__ import annotations

import base64
import itertools
import os
import sys
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

# Each renderer (i.e. each generation) gets its own image id from this counter,
# so a new generation's in-place redraws never disturb earlier images.
_ID_COUNTER = itertools.count(1991)
_CHUNK = 4096
_RESET = "\x1b[0m"


def detect_protocol() -> str:
    """Return 'kitty' if the terminal supports the Kitty graphics protocol."""
    if os.environ.get("DIFFUSION_FORCE_KITTY") == "1":
        return "kitty"
    if os.environ.get("DIFFUSION_NO_IMAGES") == "1":
        return "none"
    term = os.environ.get("TERM", "")
    prog = os.environ.get("TERM_PROGRAM", "")
    if (
        "ghostty" in term
        or prog == "ghostty"
        or "kitty" in term
        or os.environ.get("KITTY_WINDOW_ID")
    ):
        return "kitty"
    return "none"


def _png_bytes(image: Image) -> bytes:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _display_cells(image: Image, rows: int) -> tuple[int, int]:
    """Compute (rows, cols) preserving aspect ratio (cells are ~2x taller than wide)."""
    w, h = image.size
    cols = max(1, round(rows * 2 * (w / h)))
    return rows, cols


def _kitty_transmit(
    image: Image, rows: int, cols: int, *, image_id: int = 1991, no_move: bool = False
) -> str:
    """Build the Kitty escape sequence that transmits + displays an image."""
    data = base64.b64encode(_png_bytes(image)).decode("ascii")
    chunks = [data[i : i + _CHUNK] for i in range(0, len(data), _CHUNK)] or [""]

    move = "C=1," if no_move else ""
    out: list[str] = []
    for idx, chunk in enumerate(chunks):
        more = 1 if idx < len(chunks) - 1 else 0
        if idx == 0:
            # a=T transmit+display, f=100 PNG, i=id, q=2 suppress replies, r/c size.
            control = f"a=T,f=100,i={image_id},q=2,{move}r={rows},c={cols},m={more}"
        else:
            control = f"m={more}"
        out.append(f"\x1b_G{control};{chunk}\x1b\\")
    return "".join(out)


def _march(text: str, palette: list[str], offset: int) -> str:
    """Colorize ``text`` one char at a time, cycling through ``palette`` from ``offset``."""
    n = len(palette)
    return "".join(palette[(offset + i) % n] + ch for i, ch in enumerate(text)) + _RESET


class KittyRenderer:
    """Pins an image (plus a status line above it) to a fixed region.

    Call :meth:`show` per frame and :meth:`finish` when done.

    Parameters
    ----------
    rows : int, default 20
        Image height in terminal rows.
    gap : int, default 1
        Blank lines between the status line and the image (or its border).
    border_palette : list of str, optional
        ANSI color escape codes. When given, an animated box is drawn around the
        image, its colors marching one step per frame. ``None`` disables the border.
    """

    def __init__(
        self, rows: int = 20, *, gap: int = 1, border_palette: list[str] | None = None
    ) -> None:
        self.rows = rows
        self.gap = gap
        self.border_palette = border_palette
        self.image_id = next(_ID_COUNTER)
        self._started = False
        self._frame = 0

    def _region_lines(self) -> int:
        base = 1 + self.gap + self.rows
        return base + (2 if self.border_palette else 0)

    def _begin(self) -> None:
        lines = self._region_lines()
        out = (
            "\x1b[?25l"  # hide cursor
            + "\n" * lines  # reserve space (scrolls if at screen bottom)
            + f"\x1b[{lines}A"  # move back up to the top of the region
            + "\x1b7"  # save the anchor (DECSC)
        )
        sys.stdout.write(out)
        sys.stdout.flush()
        self._started = True

    @staticmethod
    def _at(down: int, col: int) -> str:
        """Escape sequence to jump to (anchor + ``down`` rows, absolute column ``col``)."""
        seq = "\x1b8"  # restore to anchor (DECRC)
        if down:
            seq += f"\x1b[{down}B"
        return seq + f"\x1b[{col}G"

    def show(self, image: Image, status: str = "") -> None:
        """Render ``image`` in place, replacing the previous frame.

        Parameters
        ----------
        image : Image
            The frame to display.
        status : str, default ""
            Optional status text drawn above the image.
        """
        if not self._started:
            self._begin()
        rows, cols = _display_cells(image, self.rows)
        # Delete ONLY this generation's previous frame. d=i scopes the delete to image
        # id `i`; without it, d defaults to 'a' = delete ALL visible images.
        delete = f"\x1b_Ga=d,d=i,i={self.image_id},q=2\x1b\\"

        parts = ["\x1b8", "\x1b[2K\r", status]  # restore anchor, clear status line, repaint

        if self.border_palette:
            pal = self.border_palette
            off = self._frame * 2  # *2 so the march is visible even with few steps
            top_row = 1 + self.gap
            # Top and bottom edges.
            parts.append(self._at(top_row, 1) + _march("┌" + "─" * cols + "┐", pal, off))
            parts.append(self._at(top_row + rows + 1, 1) + _march("└" + "─" * cols + "┘", pal, off))
            # Left and right edges, one cell per image row.
            for i in range(rows):
                edge = pal[(off + i) % len(pal)] + "│" + _RESET
                parts.append(self._at(top_row + 1 + i, 1) + edge)
                parts.append(self._at(top_row + 1 + i, cols + 2) + edge)
            # Image, inset one cell inside the frame.
            parts.append(self._at(top_row + 1, 2) + delete)
            parts.append(_kitty_transmit(image, rows, cols, image_id=self.image_id, no_move=True))
        else:
            parts.append(self._at(1 + self.gap, 1) + delete)
            parts.append(_kitty_transmit(image, rows, cols, image_id=self.image_id, no_move=True))

        self._frame += 1
        sys.stdout.write("".join(parts))
        sys.stdout.flush()

    def finish(self) -> None:
        """Move below the region and restore the cursor."""
        if self._started:
            lines = self._region_lines()
            sys.stdout.write("\x1b8" + f"\x1b[{lines}B" + "\r" + "\x1b[?25h")
            sys.stdout.flush()
        self._started = False
