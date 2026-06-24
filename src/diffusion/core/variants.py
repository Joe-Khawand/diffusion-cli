"""Enumerate the precision variants a HuggingFace repo ships.

Lets ``pull`` (and the read-only ``variants`` command) show the user which weight
variants are available — fp16 / bf16 / fp32, distinguished by filename suffix —
with the download size of each, so they can choose before committing to a
multi-gigabyte download. True low-bit quantizations (GGUF/int8) live in separate
repos and are out of scope here.

Torch-free; ``HfApi`` is imported lazily so the CLI stays light.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffusion.core import cache

# Human-facing precision labels, leanest-first (matches cache._PRECISIONS order).
_LABELS = {
    "fp16": "fp16 — half precision",
    "bf16": "bf16 — brain float",
    "fp32": "fp32 — full precision",
}
# diffusers' ``variant=`` string per precision; fp32 is the unsuffixed default.
_DIFFUSERS_VARIANT = {"fp16": "fp16", "bf16": "bf16", "fp32": None}


@dataclass(frozen=True)
class Variant:
    """A downloadable precision variant of a model repo."""

    precision: str  # "fp16" | "bf16" | "fp32"
    label: str
    weight_bytes: int  # summed size of this variant's weight files
    download_bytes: int  # weight_bytes + shared config/tokenizer bytes
    diffusers_variant: str | None  # "fp16"/"bf16", or None for fp32
    recommended: bool


def list_variants(repo_id: str) -> list[Variant]:
    """List the precision variants ``repo_id`` ships, with download sizes.

    Returns variants ordered fp16, bf16, fp32, with exactly one marked
    ``recommended`` (fp16 if present, else bf16, else fp32). Raises whatever
    ``HfApi`` raises on network/API errors — callers decide how to degrade.
    """
    from huggingface_hub import HfApi

    # list_repo_tree yields RepoFile (has .size) and RepoFolder (no .size); keep files.
    sizes: dict[str, int] = {}
    for entry in HfApi().list_repo_tree(repo_id, recursive=True):
        size = getattr(entry, "size", None)
        if size is not None:
            sizes[entry.path] = size
    files = list(sizes)
    precisions = cache.available_precisions(files)
    recommended = precisions[0] if precisions else "fp32"

    variants: list[Variant] = []
    for precision in precisions:
        selected = cache.select_files(files, variant=precision)
        weight_bytes = sum(sizes.get(f, 0) for f in selected if cache.is_component_weight(f))
        download_bytes = sum(sizes.get(f, 0) for f in selected)
        variants.append(
            Variant(
                precision=precision,
                label=_LABELS.get(precision, precision),
                weight_bytes=weight_bytes,
                download_bytes=download_bytes,
                diffusers_variant=_DIFFUSERS_VARIANT.get(precision),
                recommended=precision == recommended,
            )
        )
    return variants
