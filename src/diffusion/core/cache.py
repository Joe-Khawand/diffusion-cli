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

# Fallback allow-list used only when we cannot list the repo's files up front.
# Skips onnx/openvino/flax weights but still grabs redundant fp32/.bin variants;
# the per-file selection in ``select_files`` is the lean path.
_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.model",
    "*.bin",  # some text encoders ship only .bin
]

# Metadata we always keep: pipeline/component configs, tokenizers, sentencepiece.
_META_EXTS = (".json", ".txt", ".model")
# Weight formats we can load.
_WEIGHT_EXTS = (".safetensors", ".bin")
# Weight formats we never want (single-file checkpoints, flax/tf/onnx variants).
_SKIP_WEIGHT_EXTS = (".ckpt", ".pt", ".pth", ".onnx", ".onnx_data", ".msgpack", ".h5", ".tflite")


# Precision variants, in preference order (leanest/most-compatible first).
_PRECISIONS = ("fp16", "bf16", "fp32")


def pull(repo_id: str, *, variant: str | None = None) -> tuple[Path, ModelFamily]:
    """Download ``repo_id`` into the HF cache; return (snapshot_path, family).

    Picks a single lean weight variant per component so a repo that ships
    fp32 + fp16 + .bin + single-file checkpoints downloads only what we load —
    e.g. SD 1.5 drops from ~35 GB to a few GB. ``variant`` selects a specific
    precision (``"fp16"``/``"bf16"``/``"fp32"``); ``None`` prefers fp16.
    """
    from huggingface_hub import snapshot_download

    try:
        allow = _resolve_allow_patterns(repo_id, variant)
        path = Path(snapshot_download(repo_id, allow_patterns=allow))
    except Exception as exc:  # mapped to a friendly DownloadError (or re-raised)
        raise _download_error(repo_id, exc) from exc
    return path, detect_family(path)


def _resolve_allow_patterns(repo_id: str, variant: str | None = None) -> list[str]:
    """List the repo's files and pick a lean subset; fall back to static patterns."""
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(repo_id)
    except Exception:
        return _ALLOW_PATTERNS  # offline / API error: degrade to the static list
    return select_files(files, variant=variant) or _ALLOW_PATTERNS


def _precision_of(filename: str) -> str:
    """Classify a weight filename's precision by its variant suffix."""
    name = filename.rsplit("/", 1)[-1].lower()
    if ".fp16" in name:
        return "fp16"
    if ".bf16" in name:
        return "bf16"
    return "fp32"


def available_precisions(files: list[str]) -> list[str]:
    """Return the precision variants present in ``files``, leanest-first."""
    present = {_precision_of(f) for f in files if is_component_weight(f)}
    return [p for p in _PRECISIONS if p in present]


def is_component_weight(f: str) -> bool:
    """Return True if ``f`` is a loadable per-component weight (not metadata/ckpt)."""
    lower = f.lower()
    return (
        lower.endswith(_WEIGHT_EXTS)
        and "/" in f
        and "non_ema" not in lower
        and "non-ema" not in lower
    )


def select_files(files: list[str], *, variant: str | None = None) -> list[str]:
    """Choose the leanest runnable file set from a repo's full file list.

    Keeps all metadata; for weights, picks one variant per component (safetensors
    over .bin, then the requested ``variant`` precision — falling back per
    component when absent, preferring fp16 → bf16 → fp32). Drops training-only
    ``non_ema`` weights and top-level single-file checkpoints.
    """
    keep: list[str] = []
    by_component: dict[str, list[str]] = {}
    for f in files:
        lower = f.lower()
        if not lower.endswith(_WEIGHT_EXTS + _SKIP_WEIGHT_EXTS):
            if lower.endswith(_META_EXTS):
                keep.append(f)
            continue
        if lower.endswith(_SKIP_WEIGHT_EXTS):
            continue  # non-pytorch format or single-file checkpoint
        if "/" not in f:
            continue  # top-level single-file checkpoint (e.g. v1-5-pruned.safetensors)
        if "non_ema" in lower or "non-ema" in lower:
            continue  # EMA-excluded training weights; inference uses the standard ones
        by_component.setdefault(f.split("/", 1)[0], []).append(f)

    for comp_files in by_component.values():
        keep.extend(_select_component_weights(comp_files, variant))
    return keep


def _select_component_weights(comp_files: list[str], variant: str | None) -> list[str]:
    """Pick one weight variant for a component: safetensors > .bin, then precision.

    Honors the requested ``variant`` precision when present; otherwise falls back
    to fp16 → bf16 → fp32 so a component that lacks the chosen precision still loads.
    """
    has_safetensors = any(f.lower().endswith(".safetensors") for f in comp_files)
    candidates = [
        f for f in comp_files if f.lower().endswith(".safetensors") or not has_safetensors
    ]
    if variant is not None:
        matched = [f for f in candidates if _precision_of(f) == variant]
        if matched:
            return matched
    for pref in ("fp16", "bf16"):  # leanest available wins when no/absent variant
        pref_files = [f for f in candidates if _precision_of(f) == pref]
        if pref_files:
            return pref_files
    return candidates  # fp32 only


def detect_variant(snapshot_dir: Path) -> str | None:
    """Return the diffusers ``variant`` for a snapshot's weights, or None for fp32.

    The pipeline must be loaded with the matching ``variant`` when only
    fp16/bf16 weight files were downloaded, otherwise diffusers looks for the
    (absent) fp32 files. diffusers falls back to non-variant weights per
    component, so this is safe even for mixed snapshots.
    """
    for ext in _WEIGHT_EXTS:
        for path in snapshot_dir.rglob(f"*{ext}"):
            precision = _precision_of(path.name)
            if precision != "fp32":
                return precision
    return None


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


def human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``"2.5G"``)."""
    size = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}T"
