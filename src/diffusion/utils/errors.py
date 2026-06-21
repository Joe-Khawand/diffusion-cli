"""Typed errors and friendly CLI mapping.

These exceptions carry user-facing messages. The CLI layer catches
``DiffusionError`` and prints ``.message`` without a traceback, exiting non-zero.
"""

from __future__ import annotations


class DiffusionError(Exception):
    """Base class for expected, user-facing errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ModelNotCachedError(DiffusionError):
    """The requested repo is not present in the local cache."""

    def __init__(self, repo_id: str) -> None:
        super().__init__(
            f"Model '{repo_id}' is not in the local cache.",
            hint=f"Run 'diffusion pull {repo_id}' first.",
        )
        self.repo_id = repo_id


class UnsupportedPipelineError(DiffusionError):
    """The model is not a diffusion text-to-image pipeline we can run."""

    def __init__(self, repo_id: str, detail: str) -> None:
        super().__init__(
            f"Model '{repo_id}' is not a supported text-to-image pipeline: {detail}",
            hint="Phase 1 supports SD 1.5, SDXL, SD3, and FLUX text-to-image models.",
        )
        self.repo_id = repo_id


class DownloadError(DiffusionError):
    """A download failed (offline, repo not found, auth required)."""
