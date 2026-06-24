"""Shared rich console. Import-light (rich only)."""

from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import TextIO

console = Console()
err_console = Console(stderr=True)


class _LineFilteredStream:
    """Wrap a text stream, dropping whole lines for which ``drop`` returns True.

    transformers 5.x emits ``@auto_docstring`` validation messages with a bare
    ``print()`` at module-import time (transformers/utils/auto_docstring.py),
    so they cannot be silenced via logging. This filters them at the stream
    level while passing all other output through untouched.
    """

    def __init__(self, target: TextIO, drop: Callable[[str], bool]) -> None:
        self._target = target
        self._drop = drop
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not self._drop(line):
                self._target.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        self._target.flush()

    def __getattr__(self, name: str):
        # Delegate everything else (e.g. isatty, encoding) to the real stream.
        return getattr(self._target, name)


@contextmanager
def suppress_transformers_docstring_noise() -> Generator[None]:
    """Drop transformers' ``@auto_docstring`` ``[ERROR] ... not documented`` spam.

    These are emitted via ``print()`` to stdout when model modules are imported,
    are purely cosmetic, and have no env-var or logging gate in transformers 5.x.
    """

    def _is_noise(line: str) -> bool:
        return line.startswith("[ERROR] ") and "but not documented" in line

    original = sys.stdout
    sys.stdout = _LineFilteredStream(original, _is_noise)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = original


def quiet_diffusion_libraries() -> None:
    """Suppress torch/transformers/diffusers log spam and progress bars."""
    import contextlib
    import logging
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "critical")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore")
    for name in ("transformers", "diffusers", "accelerate", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # Disable the libraries' tqdm progress bars and lower their own verbosity.
    for mod in ("diffusers.utils.logging", "transformers.utils.logging"):
        with contextlib.suppress(Exception):
            lib_logging = __import__(mod, fromlist=["disable_progress_bar", "set_verbosity_error"])
            lib_logging.disable_progress_bar()
            lib_logging.set_verbosity_error()
