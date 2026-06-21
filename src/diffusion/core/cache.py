"""Local model cache management.

Thin wrappers over ``huggingface_hub``'s cache: we reuse HF's content-addressed,
ref-counted cache rather than maintaining a parallel registry. This module owns
download (``pull``), enumeration (``list``/``info``), resolution, and deletion.
"""

from __future__ import annotations

from pathlib import Path

from diffusion.core import registry
from diffusion.core.detect import detect_family, list_components
from diffusion.core.models import ModelEntry, ModelFamily
from diffusion.utils.errors import DownloadError, ModelNotCachedError

# Files needed to run a text-to-image pipeline. Skip optional onnx/openvino/
# non-safetensors duplicates to keep downloads lean.
_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.model",
    "*.bin",  # some text encoders ship only .bin
]


def pull(repo_id: str) -> tuple[Path, ModelFamily]:
    """Download ``repo_id`` into the HF cache; return (snapshot_path, family)."""
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id, allow_patterns=_ALLOW_PATTERNS))
    except Exception as exc:  # mapped to a friendly DownloadError (or re-raised)
        raise _download_error(repo_id, exc) from exc
    return path, detect_family(path)


def peek_family(repo_id: str) -> ModelFamily:
    """Detect a repo's family by fetching only its ``model_index.json``.

    Lets ``pull`` report what a repo is (and reject non-diffusion repos) before
    committing to a multi-gigabyte download.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        index_path = Path(hf_hub_download(repo_id, "model_index.json"))
    except EntryNotFoundError:
        return registry.UNKNOWN  # has no model_index.json → not a diffusers pipeline
    except Exception as exc:  # mapped to a friendly DownloadError (or re-raised)
        raise _download_error(repo_id, exc) from exc
    return detect_family(index_path.parent)


def _download_error(repo_id: str, exc: Exception) -> DownloadError:
    """Map a huggingface_hub error to a friendly :class:`DownloadError`."""
    from huggingface_hub.errors import (
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    if isinstance(exc, RepositoryNotFoundError):
        return DownloadError(
            f"Repository '{repo_id}' was not found on HuggingFace.",
            hint="Check the repo id, or run 'huggingface-cli login' if it is private.",
        )
    if isinstance(exc, GatedRepoError):
        return DownloadError(
            f"Repository '{repo_id}' is gated and requires accepting its license.",
            hint="Accept the license on the model page, then 'huggingface-cli login'.",
        )
    if isinstance(exc, LocalEntryNotFoundError):
        return DownloadError(
            f"Could not reach HuggingFace to download '{repo_id}'.",
            hint="Check your internet connection. Behind a corporate proxy, set "
            "SSL_CERT_FILE to a CA bundle that includes your proxy's root certificate.",
        )
    raise exc


def resolve_local(repo_id: str) -> Path:
    """Return the cached snapshot path for ``repo_id`` without hitting the network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except (LocalEntryNotFoundError, FileNotFoundError) as exc:
        raise ModelNotCachedError(repo_id) from exc


def _is_diffusion_repo(snapshot_path: Path) -> bool:
    return (snapshot_path / "model_index.json").is_file()


def list_models(*, include_all: bool = False) -> list[ModelEntry]:
    """Enumerate cached model repos as :class:`ModelEntry` records."""
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir()
    entries: list[ModelEntry] = []
    for repo in cache_info.repos:
        if repo.repo_type != "model":
            continue
        revision = _latest_revision(repo)
        snapshot_path = Path(revision.snapshot_path) if revision else repo.repo_path
        is_diffusion = revision is not None and _is_diffusion_repo(snapshot_path)
        if not is_diffusion and not include_all:
            continue
        family = detect_family(snapshot_path) if is_diffusion else registry.UNKNOWN
        entries.append(
            ModelEntry(
                repo_id=repo.repo_id,
                family=family,
                size_on_disk=repo.size_on_disk,
                size_on_disk_str=repo.size_on_disk_str,
                last_modified=repo.last_modified,
                commit_hash=revision.commit_hash if revision else None,
                local_path=snapshot_path,
                components=list_components(snapshot_path) if is_diffusion else [],
            )
        )
    entries.sort(key=lambda e: e.repo_id)
    return entries


def get_info(repo_id: str) -> ModelEntry:
    """Return the cached :class:`ModelEntry` for ``repo_id``, or raise."""
    for entry in list_models(include_all=True):
        if entry.repo_id == repo_id:
            return entry
    raise ModelNotCachedError(repo_id)


def remove(repo_id: str) -> int:
    """Delete all cached revisions of ``repo_id``. Returns freed bytes (estimate)."""
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir()
    commit_hashes: list[str] = []
    freed = 0
    for repo in cache_info.repos:
        if repo.repo_type == "model" and repo.repo_id == repo_id:
            freed += repo.size_on_disk
            commit_hashes.extend(rev.commit_hash for rev in repo.revisions)
    if not commit_hashes:
        raise ModelNotCachedError(repo_id)
    cache_info.delete_revisions(*commit_hashes).execute()
    return freed


def _latest_revision(repo):
    """Pick the most recently modified revision of a cached repo."""
    revisions = list(repo.revisions)
    if not revisions:
        return None
    return max(revisions, key=lambda r: r.last_modified)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}T"
