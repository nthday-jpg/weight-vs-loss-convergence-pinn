"""SoftAdapt loss balancing from the algorithm described in the paper."""

from collections import deque
import math

import numpy as np

from .base import BaseBalancer


class SoftAdaptBalancer(BaseBalancer):
    def __init__(
        self,
        beta: float = 0.1,
        history_length: int = 5,
        normalized: bool = False,
        loss_weighted: bool = False,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if history_length < 2:
            raise ValueError("SoftAdapt history_length must be at least 2.")

        self.beta = beta
        self.history_length = history_length
        self.normalized = normalized
        self.loss_weighted = loss_weighted
        self.epsilon = epsilon
        self.weights = {}
        self.history = {}

        # Least-squares slope weights: slope = w . history is the slope of the
        # best-fit line through the equally spaced loss history. Centering makes
        # the estimator well-conditioned, unlike a high-order finite difference.
        points = np.arange(history_length, dtype=np.float64)
        centered = points - points.mean()
        self._slope_weights = centered / np.dot(centered, centered)

    @staticmethod
    def _current_losses(loss_dict):
        current = {}
        for key, value in loss_dict.items():
            if hasattr(value, "detach"):
                value = value.detach().item()
            current[key.removesuffix("_loss")] = float(value)
        return current

    def _ensure_components(self, names):
        if set(names) != set(self.history):
            weight = 1.0 / len(names)
            self.weights = {name: weight for name in names}
            self.history = {
                name: deque(maxlen=self.history_length) for name in names
            }

    def _record(self, current):
        self._ensure_components(current)
        for name, value in current.items():
            self.history[name].append(value)

    def _history_is_full(self):
        return all(len(values) == self.history_length for values in self.history.values())

    def _update_weights(self):
        slopes = {
            name: float(np.dot(self._slope_weights, np.asarray(values)))
            for name, values in self.history.items()
        }
        averages = {
            name: float(np.mean(values)) for name, values in self.history.items()
        }

        scores = slopes
        if self.normalized:
            scale = sum(abs(value) for value in slopes.values()) + self.epsilon
            scores = {name: value / scale for name, value in slopes.items()}

        max_score = max(scores.values())
        exponentials = {
            name: math.exp(self.beta * (value - max_score))
            for name, value in scores.items()
        }
        denominator = sum(exponentials.values())
        weights = {
            name: value / denominator for name, value in exponentials.items()
        }

        if self.loss_weighted:
            weighted = {
                name: averages[name] * weights[name] for name in weights
            }
            denominator = sum(weighted.values()) + self.epsilon
            weights = {name: value / denominator for name, value in weighted.items()}

        self.weights = weights

    def __call__(self, loss_dict):
        names = [key.removesuffix("_loss") for key in loss_dict]
        self._ensure_components(names)
        return sum(
            self.weights[name] * loss_dict[f"{name}_loss"]
            for name in self.weights
        )

    def update(self, loss_dict):
        """Update once from losses averaged over an adaptation interval."""
        current = self._current_losses(loss_dict)
        self._record(current)
        # n complete adaptation intervals are required before weights change.
        # New weights are then held fixed until the next interval ends.
        if self._history_is_full():
            self._update_weights()
