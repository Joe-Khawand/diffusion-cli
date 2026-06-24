"""Typer CLI entrypoint.

Import-light by design: only typer + stdlib at module load so `diffusion --help`
is fast and never imports torch. Command bodies delegate to `diffusion.commands.*`,
which perform heavy imports lazily inside their functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from diffusion import __version__

app = typer.Typer(
    name="diffusion",
    help="Unified local diffusion runner — pull, run, and manage HuggingFace diffusion models.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"diffusion {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Unified local diffusion runner."""


@app.command()
def pull(
    repo_id: Annotated[
        str, typer.Argument(help="HuggingFace repo id, e.g. 'stabilityai/sdxl-turbo'.")
    ],
    variant: Annotated[
        str | None,
        typer.Option("--variant", help="Precision to download: fp16, bf16, or fp32."),
    ] = None,
) -> None:
    """Download a diffusion model from HuggingFace into the local cache."""
    from diffusion.commands.pull import run_pull

    run_pull(repo_id, variant=variant)


@app.command()
def variants(
    repo_id: Annotated[str, typer.Argument(help="HuggingFace repo id to inspect.")],
) -> None:
    """List a repo's downloadable precision variants with sizes and memory load."""
    from diffusion.commands.variants import run_variants

    run_variants(repo_id)


@app.command(name="list")
def list_models(
    all_models: Annotated[
        bool, typer.Option("--all", help="Include cached repos that aren't diffusion pipelines.")
    ] = False,
) -> None:
    """List diffusion models in the local cache."""
    from diffusion.commands.list_cmd import run_list

    run_list(all_models=all_models)


@app.command()
def info(
    repo_id: Annotated[str, typer.Argument(help="HuggingFace repo id to inspect.")],
) -> None:
    """Show metadata, pipeline type, and size for a model."""
    from diffusion.commands.info import run_info

    run_info(repo_id)


@app.command()
def catalog() -> None:
    """List supported model families and example HuggingFace repos to pull."""
    from diffusion.commands.catalog import run_catalog

    run_catalog()


@app.command()
def remove(
    repo_id: Annotated[str, typer.Argument(help="HuggingFace repo id to delete from the cache.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a cached model from disk."""
    from diffusion.commands.remove import run_remove

    run_remove(repo_id, yes=yes)


@app.command()
def run(
    repo_id: Annotated[str, typer.Argument(help="HuggingFace repo id to run.")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Text prompt.")],
    negative_prompt: Annotated[
        str | None, typer.Option("--negative-prompt", help="Things to avoid.")
    ] = None,
    steps: Annotated[int, typer.Option("--steps", help="Number of inference steps.")] = 25,
    width: Annotated[int, typer.Option("--width", help="Image width in pixels.")] = 512,
    height: Annotated[int, typer.Option("--height", help="Image height in pixels.")] = 512,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output image path.")] = Path(
        "output.png"
    ),
    seed: Annotated[
        int | None, typer.Option("--seed", help="Random seed for reproducibility.")
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="Force device: mps, cuda, or cpu.")
    ] = None,
    dtype: Annotated[
        str | None, typer.Option("--dtype", help="Force dtype: float16, bfloat16, float32.")
    ] = None,
    low_mem: Annotated[
        bool, typer.Option("--low-mem", help="Enable slicing + CPU offload for low memory.")
    ] = False,
    force_size: Annotated[
        bool,
        typer.Option("--force-size", help="Bypass the memory safety check for the requested size."),
    ] = False,
    image: Annotated[
        Path | None,
        typer.Option("--image", "-i", help="Init image for img2img / inpaint."),
    ] = None,
    mask: Annotated[
        Path | None, typer.Option("--mask", help="Mask image for inpaint (white = repaint).")
    ] = None,
    strength: Annotated[
        float | None, typer.Option("--strength", help="img2img/inpaint strength, 0–1.")
    ] = None,
    guidance_scale: Annotated[
        float | None, typer.Option("--guidance-scale", help="Classifier-free guidance scale.")
    ] = None,
    controlnet: Annotated[
        str | None, typer.Option("--controlnet", help="ControlNet model repo id to condition on.")
    ] = None,
    control_image: Annotated[
        Path | None,
        typer.Option("--control-image", help="Pre-processed control map (canny/depth/…)."),
    ] = None,
    sampler: Annotated[
        str | None,
        typer.Option("--sampler", help="Sampler: euler, euler-a, dpm++, ddim, unipc, … "),
    ] = None,
) -> None:
    """Generate an image from a text prompt (text2img, img2img, inpaint, ControlNet)."""
    from diffusion.commands.run import run_generate

    run_generate(
        repo_id=repo_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        output=output,
        seed=seed,
        device=device,
        dtype=dtype,
        low_mem=low_mem,
        force_size=force_size,
        init_image=image,
        mask_image=mask,
        control_image=control_image,
        controlnet=controlnet,
        strength=strength,
        guidance_scale=guidance_scale,
        sampler=sampler,
    )


@app.command()
def chat(
    repo_id: Annotated[str, typer.Argument(help="HuggingFace repo id to run interactively.")],
    steps: Annotated[int, typer.Option("--steps", help="Inference steps per image.")] = 25,
    width: Annotated[int, typer.Option("--width", help="Image width in pixels.")] = 512,
    height: Annotated[int, typer.Option("--height", help="Image height in pixels.")] = 512,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Fixed seed (default: random).")
    ] = None,
    negative_prompt: Annotated[
        str | None, typer.Option("--negative-prompt", help="Default negative prompt.")
    ] = None,
    device: Annotated[str | None, typer.Option("--device", help="Force mps, cuda, or cpu.")] = None,
    dtype: Annotated[
        str | None, typer.Option("--dtype", help="Force float16, bfloat16, float32.")
    ] = None,
    low_mem: Annotated[bool, typer.Option("--low-mem", help="Slicing + CPU offload.")] = False,
    force_size: Annotated[
        bool, typer.Option("--force-size", help="Bypass the memory safety check for the size.")
    ] = False,
    rows: Annotated[int, typer.Option("--rows", help="Preview height in terminal rows.")] = 20,
    outdir: Annotated[
        Path, typer.Option("--outdir", help="Where to save generated images.")
    ] = Path("outputs"),
    sampler: Annotated[
        str | None,
        typer.Option("--sampler", help="Sampler: euler, euler-a, dpm++, ddim, unipc, … "),
    ] = None,
) -> None:
    """Interactive chat: stream the image as it denoises, inline in the terminal."""
    from diffusion.commands.chat import run_chat

    run_chat(
        repo_id,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        negative_prompt=negative_prompt,
        device=device,
        dtype=dtype,
        low_mem=low_mem,
        force_size=force_size,
        rows=rows,
        outdir=outdir,
        sampler=sampler,
    )


def entrypoint() -> None:
    """Console-script entrypoint that renders expected errors cleanly."""
    from diffusion.utils.errors import DiffusionError

    try:
        app()
    except DiffusionError as exc:
        from diffusion.utils.console import err_console

        err_console.print(f"[red]Error:[/red] {exc.message}")
        if exc.hint:
            err_console.print(f"[dim]{exc.hint}[/dim]")
        raise SystemExit(1) from exc
