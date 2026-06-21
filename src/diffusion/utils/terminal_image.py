"""Inline image rendering in the terminal via the Kitty graphics protocol.

Supported by Ghostty and Kitty. We transmit PNG bytes base64-encoded in 4 KB
chunks. For animation (e.g. denoising steps) the renderer pins a fixed region:
it hides the cursor, reserves rows, saves an anchor, and on each frame restores
to the anchor, repaints a status line, and redraws the image with the Kitty
"do not move cursor" policy (C=1) so nothing drifts.

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


class KittyRenderer:
    """Pins an image (plus an optional status line above it) to a fixed region.

    Call :meth:`show` per frame and :meth:`finish` when done. The region is
    ``rows + 1`` lines tall (one line for the status bar).
    """

    def __init__(self, rows: int = 20) -> None:
        self.rows = rows
        self.image_id = next(_ID_COUNTER)
        self._started = False

    def _begin(self) -> None:
        lines = self.rows + 1
        out = (
            "\x1b[?25l"  # hide cursor
            + "\n" * lines  # reserve space (scrolls if at screen bottom)
            + f"\x1b[{lines}A"  # move back up to the top of the region
            + "\x1b7"  # save the anchor (DECSC)
        )
        sys.stdout.write(out)
        sys.stdout.flush()
        self._started = True

    def show(self, image: Image, status: str = "") -> None:
        if not self._started:
            self._begin()
        rows, cols = _display_cells(image, self.rows)
        parts = [
            "\x1b8",  # restore to anchor (DECRC)
            "\x1b[2K\r",  # clear the status line
            status,
            "\n\r",  # drop to the image row, column 0
            f"\x1b_Ga=d,i={self.image_id},q=2\x1b\\",  # delete only this gen's prev frame
            _kitty_transmit(image, rows, cols, image_id=self.image_id, no_move=True),
        ]
        sys.stdout.write("".join(parts))
        sys.stdout.flush()

    def finish(self) -> None:
        """Move below the region and restore the cursor."""
        if self._started:
            lines = self.rows + 1
            sys.stdout.write("\x1b8" + f"\x1b[{lines}B" + "\r" + "\x1b[?25h")
            sys.stdout.flush()
        self._started = False
