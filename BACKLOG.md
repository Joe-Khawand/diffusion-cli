# Backlog

Deferred work and improvements. Not scheduled — grab from here when picking up
follow-on tasks.

## Tighten download patterns to avoid disk bloat

**Problem:** `pull` currently downloads every weight variant a repo ships. For
`stable-diffusion-v1-5` this pulled **~33 GB** for a model whose runtime
footprint is only **~2–2.5 GB** in fp16. The repo contains ~6 redundant copies
of the same weights: fp32 + fp16, `.bin` + `.safetensors`, EMA + non-EMA, plus
two full single-file checkpoints (`v1-5-pruned.safetensors`,
`v1-5-pruned-emaonly.safetensors`).

The current allow-list in `src/diffusion/core/cache.py` (`_ALLOW_PATTERNS`)
includes `*.safetensors`, `*.bin`, and `*.ckpt`, so it grabs all of them.

**Fix:** prefer a single lean variant and skip the rest:
- Prefer `*.fp16.safetensors`; fall back to `*.safetensors` only when no fp16
  variant exists.
- Skip `*.bin` when a `.safetensors` equivalent is present (avoid duplicate
  formats; safetensors is safer/faster).
- Skip non-diffusers single-file checkpoints (`v1-5-pruned*.safetensors`,
  top-level `*.ckpt`) — we load the diffusers component layout, not single-file.
- Skip `.non_ema` UNet weights (inference uses the EMA/standard weights).

Likely needs a two-step approach: list the repo files
(`HfApi().list_repo_files`) to decide whether an fp16 variant exists, then build
`allow_patterns` accordingly, rather than a static list. Keep `*.json`, `*.txt`,
and tokenizer/`*.model` files.

Expected impact: SD 1.5 download drops from ~33 GB to ~2–4 GB.

**Also:** add a way to clean up the existing bloated cache entry (the current
33 GB SD 1.5 download can be removed with `diffusion remove
stable-diffusion-v1-5/stable-diffusion-v1-5` and re-pulled lean once fixed).
