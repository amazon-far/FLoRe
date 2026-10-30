"""1D action decoders for DP-Diffusion.

``ConditionalUnet1D`` (FiLM) is the default (``diffusion_net="unet"``, UMI's choice);
``ConditionalDiT1D`` (AdaLN + RoPE, ``diffusion_net="dit"``) also accepts a per-position time.
"""

from flore.s1_policy.model.dit.conditional_dit1d import ConditionalDiT1D
from flore.s1_policy.model.dit.conditional_unet1d import ConditionalUnet1D

__all__ = ["ConditionalDiT1D", "ConditionalUnet1D"]
