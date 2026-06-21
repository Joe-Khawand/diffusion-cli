"""Fast latent -> RGB preview for live denoising display.

Decoding the VAE every step is expensive, so we use the cheap linear projection
from latent channels to RGB (the same approximation ComfyUI uses for previews).
This only works for 4-channel SD-style latents (SD 1.5 / SDXL); other families
return None and previews are skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from diffusion.core.models import PipelineKind

if TYPE_CHECKING:
    from PIL.Image import Image

# Each row maps one latent channel to an (R, G, B) contribution.
_LATENT_RGB_FACTORS: dict[PipelineKind, list[list[float]]] = {
    PipelineKind.SD15: [
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ],
    PipelineKind.SDXL: [
        [0.3816, 0.4930, 0.5320],
        [-0.3753, 0.1631, 0.1739],
        [0.1770, 0.3588, -0.2048],
        [-0.4350, -0.2644, -0.4289],
    ],
}


def latents_to_preview(latents, kind: PipelineKind) -> Image | None:
    """Project the first item of a latent batch to a small RGB preview image."""
    factors = _LATENT_RGB_FACTORS.get(kind)
    if factors is None:
        return None

    import torch
    from PIL import Image as PILImage

    weight = torch.tensor(factors, dtype=torch.float32)  # (C, 3)
    sample = latents[0].detach().to("cpu", dtype=torch.float32)  # (C, H, W)
    if sample.shape[0] != weight.shape[0]:
        return None

    rgb = torch.einsum("chw,cr->hwr", sample, weight)  # (H, W, 3)
    rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return PILImage.fromarray(rgb.numpy(), mode="RGB")
