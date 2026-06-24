"""Sampler selection: name resolution and scheduler swapping."""

from __future__ import annotations

import pytest

from diffusion.core import samplers
from diffusion.utils.errors import InvalidSamplerError


class _FakePipe:
    """Minimal stand-in exposing the ``scheduler.config`` swap surface."""

    def __init__(self, scheduler):
        self.scheduler = scheduler


def test_available_samplers_includes_common_names():
    names = samplers.available_samplers()
    assert {"euler", "euler-a", "dpm++", "ddim", "unipc"} <= set(names)


def test_current_sampler_reports_scheduler_class():
    from diffusers import EulerDiscreteScheduler

    pipe = _FakePipe(EulerDiscreteScheduler())
    assert samplers.current_sampler(pipe) == "EulerDiscreteScheduler"


def test_apply_sampler_swaps_scheduler_case_insensitively():
    from diffusers import EulerDiscreteScheduler

    pipe = _FakePipe(EulerDiscreteScheduler())
    resolved = samplers.apply_sampler(pipe, "DDIM")
    assert resolved == "DDIMScheduler"
    assert samplers.current_sampler(pipe) == "DDIMScheduler"


def test_apply_sampler_passes_config_overrides():
    from diffusers import EulerDiscreteScheduler

    pipe = _FakePipe(EulerDiscreteScheduler())
    samplers.apply_sampler(pipe, "dpm++-karras")
    assert samplers.current_sampler(pipe) == "DPMSolverMultistepScheduler"
    assert pipe.scheduler.config.get("use_karras_sigmas") is True


def test_apply_sampler_rejects_unknown_name():
    from diffusers import EulerDiscreteScheduler

    pipe = _FakePipe(EulerDiscreteScheduler())
    with pytest.raises(InvalidSamplerError):
        samplers.apply_sampler(pipe, "totally-not-a-sampler")
