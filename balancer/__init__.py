from .base import BaseBalancer
from .gradient_annealing import GradientAnnealingBalancer
from .softadapt import SoftAdaptBalancer


def build_balancer(config):
    """Instantiate the loss balancer selected by ``config.balancer``."""
    if config.balancer == "softadapt":
        return SoftAdaptBalancer(**config.softadapt_params)
    if config.balancer == "gradient_annealing":
        return GradientAnnealingBalancer(**config.gradient_annealing_params)
    raise ValueError(
        f"Unknown balancer {config.balancer!r}. "
        "Use 'softadapt' or 'gradient_annealing'."
    )


__all__ = [
    "BaseBalancer",
    "SoftAdaptBalancer",
    "GradientAnnealingBalancer",
    "build_balancer",
]
