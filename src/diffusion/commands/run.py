"""`diffusion run` — generate an image from a prompt."""

from __future__ import annotations

from pathlib import Path


def run_generate(
    *,
    repo_id: str,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    width: int,
    height: int,
    output: Path,
    seed: int | None,
    device: str | None,
    dtype: str | None,
    low_mem: bool,
) -> None:
    from diffusion.core.generate import generate

    generate(
        repo_id=repo_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        output=output,
        seed=seed,
        device_override=device,
        dtype_override=dtype,
        low_mem=low_mem,
    )
