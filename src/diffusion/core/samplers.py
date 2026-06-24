"""Sampler (scheduler) selection.

Maps friendly sampler names to diffusers scheduler classes and swaps them onto a
loaded pipeline via ``Scheduler.from_config(pipe.scheduler.config)`` — the
mechanism the diffusers docs recommend for interchanging samplers. ``diffusers``
is imported lazily so importing this module stays cheap.

Note: classic samplers here target the epsilon/v-prediction families (SD 1.5,
SDXL, etc.). Flow-matching families (FLUX, SD3) ship a
``FlowMatchEulerDiscreteScheduler`` and ignore these; swapping their sampler is a
no-op at best, so the runner only applies a sampler when the user asks for one.
"""

from __future__ import annotations

from diffusion.utils.errors import InvalidSamplerError

# Friendly name -> (diffusers scheduler class name, extra from_config overrides).
# The class is resolved lazily from the ``diffusers`` namespace so this module
# stays import-light.
_SAMPLERS: dict[str, tuple[str, dict]] = {
    "euler": ("EulerDiscreteScheduler", {}),
    "euler-a": ("EulerAncestralDiscreteScheduler", {}),
    "dpm++": ("DPMSolverMultistepScheduler", {}),
    "dpm++-karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpm++-sde": ("DPMSolverSDEScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "ddpm": ("DDPMScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "deis": ("DEISMultistepScheduler", {}),
}


def available_samplers() -> list[str]:
    """Return the friendly sampler names accepted by :func:`apply_sampler`."""
    return list(_SAMPLERS)


def current_sampler(pipe) -> str:
    """Return the diffusers scheduler class name currently set on ``pipe``."""
    scheduler = getattr(pipe, "scheduler", None)
    return type(scheduler).__name__ if scheduler is not None else "unknown"


def apply_sampler(pipe, name: str) -> str:
    """Swap ``pipe``'s scheduler to the sampler named ``name`` (case-insensitive).

    Returns the resolved diffusers scheduler class name. Raises
    :class:`~diffusion.utils.errors.InvalidSamplerError` for an unknown name.
    """
    key = name.strip().lower()
    if key not in _SAMPLERS:
        raise InvalidSamplerError(name, available_samplers())

    import diffusers

    class_name, overrides = _SAMPLERS[key]
    scheduler_cls = getattr(diffusers, class_name)
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **overrides)
    return class_name
