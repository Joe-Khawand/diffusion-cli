"""`diffusion agent` — iterative image refinement using Claude's vision."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_agent(
    *,
    repo_id: str,
    goal: str,
    max_iterations: int,
    outdir: Path,
    steps: int,
    width: int,
    height: int,
    seed: int | None,
    device: str | None,
    dtype: str | None,
    low_mem: bool,
    force_size: bool,
    sampler: str | None,
) -> None:
    """Run the iterative refinement agent.

    Loads the model once, then loops: plan a prompt with Claude, generate an
    image, critique the result with Claude's vision, refine, and repeat until
    the score is high enough or iterations are exhausted.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id or family slug.
    goal : str
        Natural language description of the desired image.
    max_iterations : int
        Maximum number of plan-generate-critique cycles.
    outdir : Path
        Directory where iteration images and the agent log are saved.
    steps : int
        Number of denoising steps per generation.
    width, height : int
        Output image dimensions in pixels.
    seed : int or None
        Fixed random seed, or None for a random seed per iteration.
    device : str or None
        Device override, or None to autodetect.
    dtype : str or None
        Torch dtype override, or None to autodetect.
    low_mem : bool
        If True, enable memory-saving optimizations.
    force_size : bool
        If True, bypass the pre-flight memory safety check.
    sampler : str or None
        Sampler/scheduler name, or None for the model's default.
    """
    from diffusion.utils.console import console, quiet_diffusion_libraries

    quiet_diffusion_libraries()

    from diffusion.core import generate as gen
    from diffusion.core import registry
    from diffusion.core.generate import write_sidecar
    from diffusion.core.planner import AgentPlanner
    from diffusion.utils import ui
    from diffusion.utils.terminal_image import detect_protocol

    repo_id = registry.resolve_repo(repo_id)
    protocol = detect_protocol()

    # --- Plan the first prompt before loading the heavy model ---
    planner = AgentPlanner(goal)
    with ui.loading_status("Planning initial prompt with Claude …"):
        prompt, negative_prompt = planner.initial_plan()
    console.print(f"[dim]Initial prompt:[/dim] {prompt}")

    # --- Load the pipeline once ---
    with ui.loading_status(f"Loading {repo_id} …"):
        pipe, family, plan = gen.load_pipeline(
            repo_id,
            device_override=device,
            dtype_override=dtype,
            low_mem=low_mem,
            sampler=sampler,
        )
    console.print(ui.model_ready_panel(repo_id, family, plan.device, plan.dtype))
    console.print(f"[bold]Goal:[/bold] {goal}\n")

    outdir.mkdir(parents=True, exist_ok=True)
    iterations: list[dict] = []
    best_score = 0
    best_image = ""

    for i in range(1, max_iterations + 1):
        console.print(f"[bold]Iteration {i}/{max_iterations}[/bold]")
        console.print(f"[dim]Prompt:[/dim] {prompt}")
        if negative_prompt:
            console.print(f"[dim]Negative:[/dim] {negative_prompt}")

        image, elapsed = ui.run_with_preview(
            gen,
            pipe,
            family,
            plan,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            low_mem=low_mem,
            force_size=force_size,
            protocol=protocol,
            rows=20,
        )

        image_name = f"agent_{i:03d}.png"
        image_path = outdir / image_name
        image.save(image_path)
        write_sidecar(
            image_path,
            repo_id=repo_id,
            family=family,
            task="text2img",
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            sampler=type(pipe.scheduler).__name__,
            device=plan.device,
            dtype=plan.dtype,
            elapsed_s=round(elapsed, 2),
        )
        console.print(
            ui.result_panel(image_path, seed, steps, f"{width}×{height}", elapsed)
        )

        # --- Critique ---
        with ui.loading_status("Claude is reviewing the image …"):
            critique, score, new_prompt, new_neg = planner.critique_and_refine(
                image_path, prompt, i
            )

        iterations.append(
            {
                "iteration": i,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "critique": critique,
                "score": score,
                "image": image_name,
                "elapsed_s": round(elapsed, 2),
            }
        )

        if score > best_score:
            best_score = score
            best_image = image_name

        score_color = "green" if score >= 8 else "yellow" if score >= 5 else "red"
        console.print(f"[{score_color}]Score: {score}/10[/]")
        console.print(f"[dim]{critique}[/dim]\n")

        if score >= 8:
            console.print("[green]Goal achieved.[/green]")
            break

        prompt = new_prompt
        negative_prompt = new_neg
        if i < max_iterations:
            console.print(f"[dim]Refined prompt:[/dim] {new_prompt}\n")

    # --- Write agent log ---
    log = {
        "goal": goal,
        "model": repo_id,
        "settings": {
            "steps": steps,
            "width": width,
            "height": height,
            "sampler": type(pipe.scheduler).__name__,
            "device": plan.device,
            "dtype": plan.dtype,
        },
        "iterations": iterations,
        "final_iteration": len(iterations),
        "final_score": iterations[-1]["score"] if iterations else 0,
        "best_score": best_score,
        "best_image": best_image,
        "satisfied": best_score >= 8,
    }
    log_path = outdir / "agent_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    # Copy the best iteration to a stable output path.
    import shutil

    final_path = outdir / "final.png"
    shutil.copy2(outdir / best_image, final_path)

    console.print(
        f"\n[bold]Agent finished.[/bold] Best score: [{score_color}]{best_score}/10[/] "
        f"→ [bold]{final_path}[/bold]\n"
        f"[dim]Log: {log_path}[/dim]"
    )
