from .base import BaseBalancer
from .gradient_annealing import GradientAnnealing
from .softadapt import SoftAdapt


def build_balancer(config):
    """Instantiate the loss balancer selected by ``config.balancer``."""
    if config.balancer == "softadapt":
        return SoftAdapt(**config.softadapt_params)
    if config.balancer == "gradient_annealing":
        return GradientAnnealing(**config.gradient_annealing_params)
    raise ValueError(
        f"Unknown balancer {config.balancer!r}. "
        "Use 'softadapt' or 'gradient_annealing'."
    )


__all__ = [
    "BaseBalancer",
    "SoftAdapt",
    "GradientAnnealing",
    "build_balancer",
]
