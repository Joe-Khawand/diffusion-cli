# diffusion

A unified local diffusion runner — an `ollama run`-style CLI for HuggingFace diffusion models.

```bash
diffusion pull stabilityai/sdxl-turbo
diffusion run stabilityai/sdxl-turbo --prompt "a robot in a forest" -o robot.png
diffusion chat stabilityai/sdxl-turbo          # interactive REPL with live previews
diffusion list
diffusion info stabilityai/sdxl-turbo
diffusion remove stabilityai/sdxl-turbo
```

## Interactive mode

`diffusion chat <repo_id>` loads the model once and opens a prompt loop. In a
terminal that supports the Kitty graphics protocol (e.g. **Ghostty**, Kitty), it
renders the image **inline and redraws it after every denoising step** — so you
watch it emerge from noise. The final crisp frame is drawn when decoding finishes,
and each image is saved to `outputs/`.

```
› a robot in a forest
‹live preview animates here, then the final image›
✓ outputs/chat_001.png (4.2s)

› /steps 30
› /size 768x768
› /seed 42
› /neg blurry, low quality
› /help        /exit
```

Previews use a fast linear latent→RGB approximation (no extra VAE decode), so they
add negligible time. If no inline-image terminal is detected, generation still
works and images are saved; force rendering with `DIFFUSION_FORCE_KITTY=1` or
disable it with `DIFFUSION_NO_IMAGES=1`.

Phase 1 (Image Foundation) supports text-to-image generation for SD 1.5 and SDXL
(first-class), with FLUX and SD3 as best-effort. Hardware backends: Apple Silicon
(MPS), CUDA, and CPU, with automatic dtype and memory-optimization selection.

Each `run` writes the image plus a JSON sidecar (`<output>.json`) recording the
prompt, seed, steps, device, dtype, and elapsed time.

## Performance (M3 Max, SD 1.5, 512×512, fp16/MPS)

Measured steady-state ≈ **0.44 s/step + ~1.5 s fixed overhead**:

| Steps | Time | Notes |
| ----: | ---: | ----- |
| 8     | ~5 s | meets the <5 s target |
| 20    | ~10 s | |
| 25    | ~19 s (cold) | first run pays one-time MPS kernel warmup (~5 s) |

The <5 s target is reachable at low step counts; for fast high-quality output use
a distilled model (e.g. SDXL-Turbo / LCM, 1–4 steps). The default is 25 steps for
quality.

## Development

```bash
uv sync
uv run pytest -m "not integration"   # fast offline suite (no downloads)
uv run diffusion --help
```

### Corporate proxy / TLS

`huggingface_hub` uses Python's TLS, which trusts `certifi` (not the macOS
keychain). Behind a TLS-inspecting proxy, point Python at a CA bundle that
includes the proxy's root certificate:

```bash
export SSL_CERT_FILE=/path/to/corporate-ca-bundle.pem
```
