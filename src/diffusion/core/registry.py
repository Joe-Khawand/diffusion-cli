"""Registry of known diffusion model families.

Pure data: imports only :mod:`diffusion.core.models` (no torch/diffusers), so the
CLI can read it cheaply. Each :class:`ModelFamily` lists the diffusers
``_class_name`` values (across text2img/img2img/inpaint/controlnet variants) that
map to it, plus the knobs the runner needs (memory profile, latent channels,
dtype hint, preview projection) and example HuggingFace repos for ``catalog``.

Anything not listed here but still a recognizable ``*Pipeline`` is handled by
:data:`GENERIC` — we trust diffusers' ``AutoPipeline`` to route it. Truly
unrecognizable repos map to :data:`UNKNOWN`.
"""

from __future__ import annotations

from diffusion.core.models import ModelFamily, PreviewSpec

# --- Latent→RGB preview projections (factors + bias), from ComfyUI's
# comfy/latent_formats.py. 4-channel for SD-VAE families, 16-channel for the
# rectified-flow MMDiT families (SD3/FLUX). ---

_SD15_PREVIEW = PreviewSpec(
    factors=[
        [0.3512, 0.2297, 0.3227],
        [0.3250, 0.4974, 0.2350],
        [-0.2829, 0.1762, 0.2721],
        [-0.2120, -0.2616, -0.7177],
    ],
    bias=[0.0, 0.0, 0.0],
    channels=4,
)

_SDXL_PREVIEW = PreviewSpec(
    factors=[
        [0.3651, 0.4232, 0.4341],
        [-0.2533, -0.0042, 0.1068],
        [0.1076, 0.1111, -0.0362],
        [-0.3165, -0.2492, -0.2188],
    ],
    bias=[0.1084, -0.0175, -0.0011],
    channels=4,
)

_SD3_PREVIEW = PreviewSpec(
    factors=[
        [-0.0922, -0.0175, 0.0749],
        [0.0311, 0.0633, 0.0954],
        [0.1994, 0.0927, 0.0458],
        [0.0856, 0.0339, 0.0902],
        [0.0587, 0.0272, -0.0496],
        [-0.0006, 0.1104, 0.0309],
        [0.0978, 0.0306, 0.0427],
        [-0.0042, 0.1038, 0.1358],
        [-0.0194, 0.0020, 0.0669],
        [-0.0488, 0.0130, -0.0268],
        [0.0922, 0.0988, 0.0951],
        [-0.0278, 0.0524, -0.0542],
        [0.0332, 0.0456, 0.0895],
        [-0.0069, -0.0030, -0.0810],
        [-0.0596, -0.0465, -0.0293],
        [-0.1448, -0.1463, -0.1189],
    ],
    bias=[0.2394, 0.2135, 0.1925],
    channels=16,
)

_FLUX_PREVIEW = PreviewSpec(
    factors=[
        [-0.0346, 0.0244, 0.0681],
        [0.0034, 0.0210, 0.0687],
        [0.0275, -0.0668, -0.0433],
        [-0.0174, 0.0160, 0.0617],
        [0.0859, 0.0721, 0.0329],
        [0.0004, 0.0383, 0.0115],
        [0.0405, 0.0861, 0.0915],
        [-0.0236, -0.0185, -0.0259],
        [-0.0245, 0.0250, 0.1180],
        [0.1008, 0.0755, -0.0421],
        [-0.0515, 0.0201, 0.0011],
        [0.0428, -0.0012, -0.0036],
        [0.0817, 0.0765, 0.0749],
        [-0.1264, -0.0522, -0.1103],
        [-0.0280, -0.0881, -0.0499],
        [-0.1262, -0.0982, -0.0778],
    ],
    bias=[-0.0329, -0.0718, -0.0851],
    channels=16,
)


# --- The curated families. Ordered roughly by lineage for the `catalog` view. ---
FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        id="sd1.5",
        label="Stable Diffusion 1.5 / 2.x",
        class_names=(
            "StableDiffusionPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
        ),
        latent_channels=4,
        cuda_dtype="float16",
        preview=_SD15_PREVIEW,
        example_repos=(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
        ),
    ),
    ModelFamily(
        id="sdxl",
        label="Stable Diffusion XL",
        class_names=(
            "StableDiffusionXLPipeline",
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLControlNetPipeline",
            "StableDiffusionXLControlNetImg2ImgPipeline",
            "StableDiffusionXLControlNetInpaintPipeline",
            "StableDiffusionXLControlNetUnionPipeline",
        ),
        latent_channels=4,
        cuda_dtype="float16",
        preview=_SDXL_PREVIEW,
        example_repos=("stabilityai/sdxl-turbo", "stabilityai/stable-diffusion-xl-base-1.0"),
    ),
    ModelFamily(
        id="lcm",
        label="Latent Consistency Model",
        class_names=(
            "LatentConsistencyModelPipeline",
            "LatentConsistencyModelImg2ImgPipeline",
        ),
        latent_channels=4,
        cuda_dtype="float16",
        preview=_SD15_PREVIEW,
        example_repos=("SimianLuo/LCM_Dreamshaper_v7",),
    ),
    ModelFamily(
        id="kolors",
        label="Kolors",
        class_names=("KolorsPipeline", "KolorsImg2ImgPipeline"),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="float16",
        preview=_SDXL_PREVIEW,
        example_repos=("Kwai-Kolors/Kolors-diffusers",),
    ),
    ModelFamily(
        id="sd3",
        label="Stable Diffusion 3 / 3.5",
        class_names=(
            "StableDiffusion3Pipeline",
            "StableDiffusion3Img2ImgPipeline",
            "StableDiffusion3InpaintPipeline",
            "StableDiffusion3ControlNetPipeline",
        ),
        memory_heavy=True,
        latent_channels=16,
        cuda_dtype="bfloat16",
        preview=_SD3_PREVIEW,
        example_repos=(
            "stabilityai/stable-diffusion-3.5-medium",
            "stabilityai/stable-diffusion-3-medium-diffusers",
        ),
    ),
    ModelFamily(
        id="flux",
        label="FLUX.1",
        class_names=(
            "FluxPipeline",
            "FluxImg2ImgPipeline",
            "FluxInpaintPipeline",
            "FluxControlPipeline",
            "FluxControlNetPipeline",
            "FluxKontextPipeline",
        ),
        memory_heavy=True,
        latent_channels=16,
        supports_negative_prompt=False,  # no real CFG path without true_cfg_scale>1
        cuda_dtype="bfloat16",
        preview=_FLUX_PREVIEW,
        example_repos=("black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"),
    ),
    ModelFamily(
        id="flux2",
        label="FLUX.2",
        class_names=("Flux2Pipeline", "Flux2KleinPipeline"),
        memory_heavy=True,
        latent_channels=16,
        supports_negative_prompt=False,
        cuda_dtype="bfloat16",
        example_repos=("black-forest-labs/FLUX.2-dev",),
    ),
    ModelFamily(
        id="chroma",
        label="Chroma",
        class_names=("ChromaPipeline",),
        memory_heavy=True,
        latent_channels=16,
        cuda_dtype="bfloat16",
        preview=_FLUX_PREVIEW,
        example_repos=("lodestones/Chroma",),
    ),
    ModelFamily(
        id="pixart-alpha",
        label="PixArt-α",
        class_names=("PixArtAlphaPipeline",),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="bfloat16",
        example_repos=("PixArt-alpha/PixArt-XL-2-1024-MS",),
    ),
    ModelFamily(
        id="pixart-sigma",
        label="PixArt-Σ",
        class_names=("PixArtSigmaPipeline",),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="bfloat16",
        example_repos=("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",),
    ),
    ModelFamily(
        id="sana",
        label="Sana",
        class_names=("SanaPipeline", "SanaSprintPipeline"),
        memory_heavy=True,
        latent_channels=32,
        cuda_dtype="bfloat16",
        example_repos=("Efficient-Large-Model/Sana_1600M_1024px_diffusers",),
    ),
    ModelFamily(
        id="auraflow",
        label="AuraFlow",
        class_names=("AuraFlowPipeline",),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="bfloat16",
        example_repos=("fal/AuraFlow-v0.3",),
    ),
    ModelFamily(
        id="lumina",
        label="Lumina-Next",
        class_names=("LuminaPipeline", "LuminaText2ImgPipeline"),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="bfloat16",
        example_repos=("Alpha-VLLM/Lumina-Next-SFT-diffusers",),
    ),
    ModelFamily(
        id="lumina2",
        label="Lumina Image 2.0",
        class_names=("Lumina2Pipeline", "Lumina2Text2ImgPipeline"),
        memory_heavy=True,
        latent_channels=16,
        cuda_dtype="bfloat16",
        example_repos=("Alpha-VLLM/Lumina-Image-2.0",),
    ),
    ModelFamily(
        id="hunyuan-dit",
        label="Hunyuan-DiT",
        class_names=("HunyuanDiTPipeline",),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="float16",
        example_repos=("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",),
    ),
    ModelFamily(
        id="kandinsky2.2",
        label="Kandinsky 2.2",
        class_names=(
            "KandinskyV22CombinedPipeline",
            "KandinskyV22Img2ImgCombinedPipeline",
            "KandinskyV22InpaintCombinedPipeline",
            "KandinskyV22Pipeline",
        ),
        latent_channels=4,
        cuda_dtype="float16",
        example_repos=("kandinsky-community/kandinsky-2-2-decoder",),
    ),
    ModelFamily(
        id="kandinsky3",
        label="Kandinsky 3",
        class_names=("Kandinsky3Pipeline", "Kandinsky3Img2ImgPipeline"),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="float16",
        example_repos=("kandinsky-community/kandinsky-3",),
    ),
    ModelFamily(
        id="wuerstchen",
        label="Würstchen",
        class_names=("WuerstchenCombinedPipeline",),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="float16",
        example_repos=("warp-ai/wuerstchen",),
    ),
    ModelFamily(
        id="stable-cascade",
        label="Stable Cascade",
        class_names=("StableCascadeCombinedPipeline",),
        memory_heavy=True,
        latent_channels=4,
        cuda_dtype="bfloat16",
        example_repos=("stabilityai/stable-cascade",),
    ),
    ModelFamily(
        id="cogview3",
        label="CogView3-Plus",
        class_names=("CogView3PlusPipeline",),
        memory_heavy=True,
        latent_channels=16,
        cuda_dtype="bfloat16",
        example_repos=("THUDM/CogView3-Plus-3B",),
    ),
    ModelFamily(
        id="cogview4",
        label="CogView4",
        class_names=("CogView4Pipeline", "CogView4ControlPipeline"),
        memory_heavy=True,
        latent_channels=16,
        cuda_dtype="bfloat16",
        example_repos=("THUDM/CogView4-6B",),
    ),
    ModelFamily(
        id="qwen-image",
        label="Qwen-Image",
        class_names=(
            "QwenImagePipeline",
            "QwenImageImg2ImgPipeline",
            "QwenImageInpaintPipeline",
            "QwenImageEditPipeline",
            "QwenImageControlNetPipeline",
        ),
        memory_heavy=True,
        latent_channels=16,
        cuda_dtype="bfloat16",
        example_repos=("Qwen/Qwen-Image",),
    ),
    ModelFamily(
        id="z-image",
        label="Z-Image",
        class_names=(
            "ZImagePipeline",
            "ZImageImg2ImgPipeline",
            "ZImageInpaintPipeline",
            "ZImageControlNetPipeline",
        ),
        memory_heavy=True,
        latent_channels=16,
        cuda_dtype="bfloat16",
        preview=_FLUX_PREVIEW,  # shares the FLUX.1 VAE
        example_repos=("Tongyi-MAI/Z-Image-Turbo",),
    ),
    ModelFamily(
        id="deepfloyd-if",
        label="DeepFloyd IF",
        class_names=("IFPipeline", "IFImg2ImgPipeline", "IFInpaintingPipeline"),
        memory_heavy=True,
        latent_channels=0,  # pixel-space cascade; no latent preview
        cuda_dtype="float16",
        example_repos=("DeepFloyd/IF-I-XL-v1.0",),
    ),
)

# Class name -> family, built from each family's declared variants.
BY_CLASS: dict[str, ModelFamily] = {cls: fam for fam in FAMILIES for cls in fam.class_names}

# Permissive fallback: a recognizable diffusers `*Pipeline` we don't have curated
# metadata for. Conservative defaults (offload, no preview, no negative prompt);
# `AutoPipeline` does the real routing at load time.
GENERIC = ModelFamily(
    id="generic",
    label="Generic diffusers pipeline",
    memory_heavy=True,
    latent_channels=0,
    supports_negative_prompt=False,
    cuda_dtype="bfloat16",
)

# Not a runnable image pipeline (no model_index.json / unrecognized).
UNKNOWN = ModelFamily(
    id="unknown",
    label="Unknown",
    latent_channels=0,
    supported=False,
)

# `_class_name` substrings that mark non-image pipelines, so `list`/`info` can
# label them honestly instead of optimistically routing to text-to-image.
NON_IMAGE_SUFFIXES: tuple[str, ...] = ("Video", "Audio", "Speech", "Music")


# Rough peak memory to run inference at fp16, single image at the model's native
# resolution, BEFORE any --low-mem/offload. Offload trades this down for speed.
# Dominated by the text encoder for some families (T5-XXL, ChatGLM, GLM-4).
# Approximate, for picking hardware — not a hard limit.
_VRAM_GB: dict[str, float] = {
    "sd1.5": 4,
    "sdxl": 10,
    "lcm": 4,
    "kolors": 16,
    "sd3": 14,
    "flux": 24,
    "flux2": 32,
    "chroma": 18,
    "pixart-alpha": 12,
    "pixart-sigma": 12,
    "sana": 9,
    "auraflow": 16,
    "lumina": 12,
    "lumina2": 16,
    "hunyuan-dit": 14,
    "kandinsky2.2": 8,
    "kandinsky3": 24,
    "wuerstchen": 10,
    "stable-cascade": 16,
    "cogview3": 16,
    "cogview4": 24,
    "qwen-image": 40,
    "z-image": 12,
    "deepfloyd-if": 16,
}


def by_class_name(class_name: str | None) -> ModelFamily | None:
    """Return the curated family for a diffusers ``_class_name``, if known."""
    if not class_name:
        return None
    return BY_CLASS.get(class_name)


def vram_hint(family: ModelFamily) -> str:
    """Return an approximate fp16 inference-memory string (e.g. ``"~10 GB"``).

    Returns ``"—"`` when no estimate is curated (e.g. the GENERIC fallback).
    """
    gb = _VRAM_GB.get(family.id)
    return f"~{gb:g} GB" if gb is not None else "—"


def family_by_id(slug: str) -> ModelFamily | None:
    """Return the curated family with id ``slug`` (e.g. ``"sdxl"``), if any."""
    return next((fam for fam in FAMILIES if fam.id == slug), None)


def resolve_repo(arg: str) -> str:
    """Map a family slug to its example HuggingFace repo; pass repo ids through.

    Lets users ``diffusion pull sdxl`` instead of copying a full repo id. Family
    slugs never contain ``/`` (real repo ids always do), so this is unambiguous.
    """
    if "/" in arg:
        return arg
    fam = family_by_id(arg)
    if fam is not None and fam.example_repos:
        return fam.example_repos[0]
    return arg


def require(class_name: str) -> ModelFamily:
    """Return the curated family for ``class_name`` or raise (registry invariant)."""
    fam = BY_CLASS.get(class_name)
    if fam is None:
        raise KeyError(f"no registered family for {class_name!r}")
    return fam


def is_available(family: ModelFamily) -> bool:
    """Return True if installed diffusers exposes a pipeline class for ``family``.

    Imports diffusers (heavy) — only call from commands, never at startup. Guards
    against advertising families the installed diffusers version cannot run.
    """
    import diffusers

    return any(hasattr(diffusers, cls) for cls in family.class_names)


def available_families() -> list[ModelFamily]:
    """Curated families whose pipeline classes exist in the installed diffusers."""
    import diffusers

    return [fam for fam in FAMILIES if any(hasattr(diffusers, cls) for cls in fam.class_names)]


def is_non_image(class_name: str | None) -> bool:
    """Return True if the class name looks like a non-image (video/audio) pipeline."""
    if not class_name:
        return False
    return any(suffix in class_name for suffix in NON_IMAGE_SUFFIXES)
