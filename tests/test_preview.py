"""Tests for latent preview decoding and Kitty escape encoding."""

from __future__ import annotations

from diffusion.core.models import PipelineKind


def test_latents_to_preview_sd15_shape() -> None:
    import torch

    from diffusion.core.preview import latents_to_preview

    latents = torch.randn(1, 4, 64, 96)  # (B, C, H, W)
    img = latents_to_preview(latents, PipelineKind.SD15)
    assert img is not None
    assert img.size == (96, 64)  # PIL is (W, H)
    assert img.mode == "RGB"


def test_latents_to_preview_unsupported_kind() -> None:
    import torch

    from diffusion.core.preview import latents_to_preview

    # FLUX latents (16 ch) have no factor table -> skipped.
    assert latents_to_preview(torch.randn(1, 16, 64, 64), PipelineKind.FLUX) is None


def test_latents_to_preview_channel_mismatch() -> None:
    import torch

    from diffusion.core.preview import latents_to_preview

    # SD15 expects 4 channels; mismatched input returns None rather than crashing.
    assert latents_to_preview(torch.randn(1, 8, 64, 64), PipelineKind.SD15) is None


def test_detect_protocol_env(monkeypatch) -> None:
    from diffusion.utils import terminal_image

    monkeypatch.setenv("DIFFUSION_FORCE_KITTY", "1")
    assert terminal_image.detect_protocol() == "kitty"

    monkeypatch.delenv("DIFFUSION_FORCE_KITTY")
    monkeypatch.setenv("DIFFUSION_NO_IMAGES", "1")
    assert terminal_image.detect_protocol() == "none"

    monkeypatch.delenv("DIFFUSION_NO_IMAGES")
    monkeypatch.setenv("TERM", "xterm-ghostty")
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "")
    assert terminal_image.detect_protocol() == "kitty"


def test_kitty_transmit_framing() -> None:
    from PIL import Image

    from diffusion.utils.terminal_image import _kitty_transmit

    img = Image.new("RGB", (32, 32), (10, 20, 30))
    seq = _kitty_transmit(img, rows=10, cols=20)

    assert seq.startswith("\x1b_G")
    assert seq.endswith("\x1b\\")
    assert "a=T" in seq and "f=100" in seq and "r=10" in seq and "c=20" in seq
    # Last chunk must signal completion with m=0.
    assert "m=0" in seq


def test_kitty_transmit_chunks_large_image() -> None:
    from PIL import Image

    from diffusion.utils import terminal_image

    # Noise compresses poorly -> base64 well over one 4 KB chunk -> multiple frames.
    img = Image.effect_noise((256, 256), 100).convert("RGB")
    seq = terminal_image._kitty_transmit(img, rows=20, cols=40)
    assert seq.count("\x1b_G") > 1  # multiple chunks
    assert "m=1" in seq  # at least one "more" marker


def test_kitty_renderer_anchors_and_redraws(capsys) -> None:
    from PIL import Image

    from diffusion.utils.terminal_image import KittyRenderer

    r = KittyRenderer(rows=8)
    img = Image.new("RGB", (16, 16), (0, 0, 0))

    r.show(img, status="step 1/8")
    first = capsys.readouterr().out
    assert "\x1b[?25l" in first  # hides cursor on first frame
    assert "\x1b[9A" in first  # reserves rows+1 lines and moves back up
    assert "\x1b7" in first  # saves the anchor
    assert "C=1" in first  # image transmitted without moving the cursor

    r.show(img, status="step 2/8")
    second = capsys.readouterr().out
    assert "\x1b8" in second  # restores to the anchor
    assert "a=d" in second  # deletes the previous frame
    assert "\x1b[?25l" not in second  # cursor already hidden; not re-reserved

    r.finish()
    end = capsys.readouterr().out
    assert "\x1b[?25h" in end  # restores the cursor


def test_each_renderer_uses_a_unique_image_id(capsys) -> None:
    from PIL import Image

    from diffusion.utils.terminal_image import KittyRenderer

    img = Image.new("RGB", (16, 16), (0, 0, 0))
    r1 = KittyRenderer(rows=4)
    r2 = KittyRenderer(rows=4)
    assert r1.image_id != r2.image_id  # so a new generation doesn't wipe the previous

    r2.show(img)
    out = capsys.readouterr().out
    # The new renderer only ever deletes/draws its own id, never r1's.
    assert f"i={r2.image_id}" in out
    # Delete must be scoped with d=i (else d defaults to 'a' = delete ALL images).
    assert f"a=d,d=i,i={r2.image_id}" in out
    assert f"i={r1.image_id}" not in out
