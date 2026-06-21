"""Fast latent -> RGB preview for live denoising display.

Decoding the VAE every step is expensive, so we use the cheap linear projection
from latent channels to RGB (the same approximation ComfyUI uses for previews).
Each family carries its own :class:`PreviewSpec` (factors + bias); families
without one (or pixel-space models) return None and previews are skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

    from diffusion.core.models import ModelFamily


def latents_to_preview(latents, family: ModelFamily) -> Image | None:
    """Project the first item of a latent batch to a small RGB preview image."""
    spec = family.preview
    if spec is None:
        return None

    import torch
    from PIL import Image as PILImage

    weight = torch.tensor(spec.factors, dtype=torch.float32)  # (C, 3)
    bias = torch.tensor(spec.bias, dtype=torch.float32)  # (3,)
    sample = latents[0].detach().to("cpu", dtype=torch.float32)  # (C, H, W)
    if sample.shape[0] != weight.shape[0]:
        return None

    rgb = torch.einsum("chw,cr->hwr", sample, weight) + bias  # (H, W, 3)
    rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return PILImage.fromarray(rgb.numpy(), mode="RGB")
