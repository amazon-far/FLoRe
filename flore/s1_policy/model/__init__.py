"""S1 VLA models: the lazily exported Diffusion policy.

Keeping these imports lazy means ``import flore.s1_policy.model`` stays cheap: the model
modules (and their timm/diffusers imports) load only when a name is first used. The public
``from flore.s1_policy.model import ...`` API is unchanged.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DPDiffusionConfig",
    "DPDiffusionModel",
    "DiffusionVLABase",
]


def __getattr__(name: str) -> Any:
    if name == "DiffusionVLABase":
        from flore.s1_policy.model.diffusion_base import DiffusionVLABase

        value = DiffusionVLABase
    elif name in {"DPDiffusionConfig", "DPDiffusionModel"}:
        from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel

        value = {
            "DPDiffusionConfig": DPDiffusionConfig,
            "DPDiffusionModel": DPDiffusionModel,
        }[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
