"""Gradient-based loss balancing with the learning-rate annealing rule.

Reference: Wang, Teng & Perdikaris (2021), "Understanding and mitigating
gradient flow pathologies in physics-informed neural networks".
"""

from .base import BaseBalancer


class GradientAnnealingBalancer(BaseBalancer):
    """Balance auxiliary losses against the PDE residual by gradient scale.

    The residual is the reference term and keeps weight 1. Every other term is
    driven toward

        lambda_hat_i = max|grad L_res| / mean|grad (lambda_i * L_i)|

    where the max and mean are taken over individual elements of the flattened
    gradient vector of all trainable parameters. The auxiliary gradient is
    measured on the weighted contribution that enters the total loss, matching
    the reference implementation, so the update is

        lambda_i <- (1 - alpha) * lambda_i + alpha * R_i / lambda_i

    with R_i = max|grad L_res| / mean|grad L_i|. This converges to
    lambda_i = sqrt(R_i) instead of chasing R_i without bound.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        reference: str = "res",
        initial_weight: float = 1.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Gradient annealing alpha must be in (0, 1].")
        if initial_weight <= 0.0:
            raise ValueError("Gradient annealing initial_weight must be positive.")

        self.alpha = alpha
        self.reference = reference
        self.initial_weight = initial_weight
        self.epsilon = epsilon
        self._names = set()

    def _ensure_components(self, names):
        names = set(names)
        if names != self._names:
            self._names = names
            self.weights = {
                name: 1.0 if name == self.reference else self.initial_weight
                for name in names
            }

    def __call__(self, loss_dict):
        names = [key.removesuffix("_loss") for key in loss_dict]
        self._ensure_components(names)
        if self.reference not in self.weights:
            raise ValueError(
                f"Gradient annealing requires the {self.reference!r} loss term."
            )
        return sum(
            self.weights[name] * loss_dict[f"{name}_loss"] for name in names
        )

    def update(self, loss_dict, *, gradients=None, step=None):
        if gradients is None:
            raise ValueError(
                "GradientAnnealingBalancer.update requires gradient statistics."
            )
        names = [key.removesuffix("_loss") for key in loss_dict]
        self._ensure_components(names)

        reference = float(gradients.max_abs[self.reference])
        for name in names:
            if name == self.reference:
                self.weights[name] = 1.0
                continue
            # The statistics are measured on the raw component loss, so scale
            # by the current weight to obtain grad(lambda_i * L_i).
            weighted_mean_abs = (
                float(gradients.mean_abs[name]) * self.weights[name]
            )
            target = reference / max(weighted_mean_abs, self.epsilon)
            self.weights[name] = (
                (1.0 - self.alpha) * self.weights[name] + self.alpha * target
            )
