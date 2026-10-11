from .base import BaseBalancer
from .registry import register_balancer


@register_balancer("uniform")
class UniformBalancer(BaseBalancer):
    """Give every loss component the same weight."""

    def __init__(self):
        super().__init__()

    def __call__(self, loss_dict):
        names = [key.removesuffix("_loss") for key in loss_dict]
        if set(names) != set(self.weights):
            weight = 1.0 / len(names)
            self.weights = {name: weight for name in names}
        return sum(
            self.weights[name] * loss_dict[f"{name}_loss"] for name in names
        )
