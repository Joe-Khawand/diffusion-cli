# diffusion cli

A unified local diffusion runner: an `ollama run`-style CLI for Hugging Face
diffusers models.

```bash
diffusion catalog
diffusion variants sdxl
diffusion pull sdxl
diffusion run sdxl --prompt "a robot in a forest" --output robot.png
diffusion chat sdxl
diffusion serve sdxl
```

`diffusion` reuses the Hugging Face cache, keeps startup light, downloads lean
weight variants where possible, and runs models locally on Apple Silicon (MPS),
CUDA, or CPU.

![diffusion chat demo](docs/assets/diffusion-chat-demo.gif)

## Install

```bash
uv tool install .
diffusion --help
```

For development:

```bash
uv sync
uv run diffusion --help
```

## First Run

List the supported model families:

```bash
diffusion catalog
```

Inspect the downloadable precision variants and estimated memory use:

```bash
diffusion variants sdxl
```

Download a model. Family slugs such as `sdxl`, `sd1.5`, `flux`, and `sana` map
to curated example repos; full Hugging Face repo ids also work.

```bash
diffusion pull sdxl
diffusion pull stabilityai/stable-diffusion-xl-base-1.0 --variant fp16
```

Generate an image:

```bash
diffusion run sdxl \
  --prompt "a cinematic photo of a robot walking through a pine forest" \
  --steps 8 \
  --width 512 \
  --height 512 \
  --seed 42 \
  --output robot.png
```

Each `run` writes the image plus a JSON sidecar (`<output>.json`) with the prompt,
seed, steps, sampler, device, dtype, elapsed time, and model family.

## Commands

| Command | Purpose |
| --- | --- |
| `diffusion catalog` | Show supported model families and example repos. |
| `diffusion variants <repo-or-slug>` | Show precision variants, download sizes, and memory estimates. |
| `diffusion pull <repo-or-slug>` | Download a model into the Hugging Face cache. |
| `diffusion run <repo-or-slug>` | Generate one image. |
| `diffusion chat <repo-or-slug>` | Load once, then generate repeatedly in an interactive prompt loop. |
| `diffusion serve <repo-or-slug>` | Load once, then serve an OpenAI-compatible local image API. |
| `diffusion list` | List cached diffusion models. |
| `diffusion info <repo>` | Show cached model metadata. |
| `diffusion remove <repo>` | Delete a cached model. |

## Generation Modes

Text-to-image is the default:

```bash
diffusion run sdxl --prompt "a watercolor city skyline" -o skyline.png
```

Image-to-image:

```bash
diffusion run sdxl \
  --prompt "turn this into a pencil sketch" \
  --image input.png \
  --strength 0.65 \
  -o sketch.png
```

Inpainting:

```bash
diffusion run sdxl \
  --prompt "replace the masked area with wildflowers" \
  --image input.png \
  --mask mask.png \
  -o inpainted.png
```

ControlNet:

```bash
diffusion run sdxl \
  --prompt "architectural render of a glass house" \
  --controlnet diffusers/controlnet-canny-sdxl-1.0 \
  --control-image canny.png \
  -o controlled.png
```

Sampler selection is available for classic SD-style schedulers:

```bash
diffusion run sdxl --prompt "a neon street scene" --sampler dpm++-karras
```

## Interactive Mode

`diffusion chat <repo-or-slug>` loads the model once and opens a prompt loop. In
a terminal that supports the Kitty graphics protocol, such as Ghostty or Kitty,
it renders the image inline and redraws it after each denoising step. If inline
images are not available, generation still works and images are saved to
`outputs/`.

```text
› a robot in a forest
‹live preview animates here, then the final image›
✓ outputs/chat_001.png

› /steps 30
› /size 768x768
› /seed 42
› /neg blurry, low quality
› /cfg 7.5
› /sampler euler-a
› /help
› /exit
```

Preview controls:

```bash
DIFFUSION_FORCE_KITTY=1 diffusion chat sdxl
DIFFUSION_NO_IMAGES=1 diffusion chat sdxl
DIFFUSION_NO_BORDER=1 diffusion chat sdxl
```

Previews use a fast latent-to-RGB approximation when the model family supports
one. Families without preview projections fall back to a progress bar.

## Local HTTP Server

`diffusion serve <repo-or-slug>` loads one text-to-image model and exposes
OpenAI-compatible image generation on `http://127.0.0.1:8000/v1`.

```bash
diffusion serve sdxl --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl",
    "prompt": "a cinematic robot in a pine forest",
    "size": "512x512",
    "steps": 8,
    "seed": 42,
    "response_format": "b64_json"
  }'
```

The v1 server supports `GET /health`, `GET /v1/models`, and
`POST /v1/images/generations`. It returns base64 PNG images and serializes
requests through the one loaded pipeline.

## Memory Safeguards

Before text-to-image generation, `diffusion` validates dimensions and estimates
whether the requested size is likely to fit in available RAM/VRAM. Oversized
requests fail early with a clear error instead of pushing the machine into swap
or crashing deep inside PyTorch.

Useful options:

```bash
diffusion run sdxl --prompt "..." --low-mem
diffusion run sdxl --prompt "..." --force-size
DIFFUSION_SKIP_MEM_CHECK=1 diffusion run sdxl --prompt "..."
```

`--low-mem` enables slicing and CPU offload. It lowers peak memory use at the
cost of speed. `--force-size` and `DIFFUSION_SKIP_MEM_CHECK=1` bypass only the
pre-flight estimate; real out-of-memory errors are still caught and reported
cleanly when possible.

## Model Support

First-class coverage is strongest for SD 1.5/2.x and SDXL. The catalog also
includes newer diffusers families such as FLUX, SD3/3.5, Sana, PixArt, Qwen-Image,
Z-Image, and others when the installed `diffusers` version exposes their pipeline
classes.

Unknown text-to-image diffusers pipelines may work through `AutoPipeline`. Video,
audio, speech, and music pipelines are intentionally not supported in this image
runner.

## Downloads

`diffusion pull` inspects the repo before downloading and selects one runnable
weight variant per component. It prefers safetensors and lean precision variants
such as fp16/bf16 when available, skipping top-level checkpoint files and
training-only weights that diffusers will not load for inference.

If repo file listing is unavailable, downloads fall back to a broader safe
allow-list so offline or proxy-limited environments still degrade gracefully.

## Performance

Performance depends heavily on model, device, resolution, sampler, and step
count. On an M3 Max with SD 1.5 at 512x512, steady-state generation was measured
around 0.44 seconds per step plus roughly 1.5 seconds fixed overhead.

For fast iteration, use distilled models such as SDXL-Turbo or LCM and low step
counts. For larger models or resolutions, use `--low-mem`.

## Development

```bash
uv sync
uv run ruff check .
uv run mypy src
uv run pyright
uv run pytest -m "not integration"
uv build
```

The offline suite mocks Hugging Face, diffusers, and torch boundaries. Real model
smoke tests should be run before a release when cached models or network access
are available.

## Corporate Proxy / TLS

`huggingface_hub` uses Python's TLS, which trusts `certifi` rather than the macOS
keychain. Behind a TLS-inspecting proxy, point Python at a CA bundle that includes
the proxy root certificate:

```bash
export SSL_CERT_FILE=/path/to/corporate-ca-bundle.pem
```
