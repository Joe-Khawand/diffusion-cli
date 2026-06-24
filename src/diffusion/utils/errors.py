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
    """The model is not a diffusion image pipeline we can run."""

    def __init__(self, repo_id: str, detail: str) -> None:
        super().__init__(
            f"Model '{repo_id}' is not a supported image pipeline: {detail}",
            hint="Run 'diffusion catalog' to see supported families. Most diffusers "
            "text-to-image repos work; video/audio pipelines do not.",
        )
        self.repo_id = repo_id


class DownloadError(DiffusionError):
    """A download failed (offline, repo not found, auth required)."""


class InvalidSamplerError(DiffusionError):
    """The requested sampler name isn't one we recognize."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            f"Unknown sampler '{name}'.",
            hint=f"Available samplers: {', '.join(available)}.",
        )
        self.name = name


_GB = 1024**3
_MEM_HINT = (
    "Try a smaller size (e.g. 512×512 or 768×768), add --low-mem, or bypass this "
    "check with --force-size / DIFFUSION_SKIP_MEM_CHECK=1."
)


class InsufficientMemoryError(DiffusionError):
    """A generation's estimated memory exceeds available RAM/VRAM."""

    def __init__(
        self, *, width: int, height: int, device: str, need_bytes: int, avail_bytes: int
    ) -> None:
        super().__init__(
            f"{width}×{height} likely needs ~{need_bytes / _GB:.1f} GB on {device}, "
            f"but only ~{avail_bytes / _GB:.1f} GB is available.",
            hint=_MEM_HINT,
        )
        self.width = width
        self.height = height
        self.device = device

    @classmethod
    def from_oom(cls, *, width: int, height: int, device: str) -> InsufficientMemoryError:
        """Build an error for a real out-of-memory crash (exact sizes unknown)."""
        self = cls.__new__(cls)
        DiffusionError.__init__(
            self,
            f"Ran out of memory generating {width}×{height} on {device}.",
            hint=_MEM_HINT,
        )
        self.width = width
        self.height = height
        self.device = device
        return self
