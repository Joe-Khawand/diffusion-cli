"""`diffusion serve` — expose a local OpenAI-compatible image API."""

from __future__ import annotations


def run_serve(
    repo_id: str,
    *,
    host: str,
    port: int,
    device: str | None,
    dtype: str | None,
    low_mem: bool,
    force_size: bool,
    sampler: str | None,
) -> None:
    """Run the local HTTP inference server."""
    from diffusion.server import create_app
    from diffusion.utils.console import quiet_diffusion_libraries

    quiet_diffusion_libraries()

    import uvicorn

    app = create_app(
        repo_id,
        device=device,
        dtype=dtype,
        low_mem=low_mem,
        force_size=force_size,
        sampler=sampler,
    )
    uvicorn.run(app, host=host, port=port)
