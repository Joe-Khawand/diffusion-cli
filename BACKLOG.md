# Backlog

Deferred work and improvements. Not scheduled — grab from here when picking up
follow-on tasks.

## ~~Tighten download patterns to avoid disk bloat~~ — DONE

Resolved in `src/diffusion/core/cache.py`: `pull` now lists the repo's files
(`HfApi().list_repo_files`) and selects one lean weight variant per component
(`_select_download_files` / `_select_component_weights`): prefers
`*.fp16.safetensors`, falls back to `*.safetensors` then `*.bin`, and skips
`.non_ema` weights and top-level single-file checkpoints. `load_pipeline` calls
`cache.detect_variant` and passes `variant="fp16"` so diffusers loads the lean
files (falling back to fp32 per-component). Falls back to the old static
allow-list if the file listing fails (offline). Verified live: SD 1.5 selects 4
fp16 safetensors out of 36 files (~2.5 GB vs ~35 GB).

**To reclaim the existing bloated cache entry:** the 35 GB SD 1.5 download can be
removed and re-pulled lean:

```
diffusion remove stable-diffusion-v1-5/stable-diffusion-v1-5
diffusion pull stable-diffusion-v1-5/stable-diffusion-v1-5
```
