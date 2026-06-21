"""diffusion — a unified local diffusion runner CLI.

Keep this module import-light: no torch/diffusers here so that `diffusion --help`
stays fast. Heavy imports live inside command/core function bodies.
"""

__version__ = "0.1.0"
