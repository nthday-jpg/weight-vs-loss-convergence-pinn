"""SoftAdapt loss balancing from the algorithm described in the paper."""

from collections import deque
import math

import numpy as np

from .base import BaseBalancer
from .registry import register_balancer


@register_balancer("softadapt")
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
        self.weights = {"ics": 1.0 / 3.0, "bcs": 1.0 / 3.0, "res": 1.0 / 3.0}
        self.history = {
            name: deque(maxlen=history_length) for name in self.weights
        }

        # Weights for the derivative at the newest point of an equally spaced
        # interpolation polynomial. This is an (n - 1)-order finite difference.
        points = np.arange(-(history_length - 1), 1, dtype=np.float64)
        vandermonde = np.vstack([points**order for order in range(history_length)])
        derivative = np.zeros(history_length, dtype=np.float64)
        derivative[1] = 1.0
        self._slope_weights = np.linalg.solve(vandermonde, derivative)

    @staticmethod
    def _current_losses(loss_dict):
        return {
            "ics": float(loss_dict["ics_loss"].detach()),
            "bcs": float(loss_dict["bcs_loss"].detach()),
            "res": float(loss_dict["res_loss"].detach()),
        }

    def _record(self, current):
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
        denominator = sum(exponentials.values()) + self.epsilon
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
        current = self._current_losses(loss_dict)
        self._record(current)
        # The algorithm requires n stored observations before adapting weights.
        if self._history_is_full():
            self._update_weights()

        return sum(
            self.weights[name] * loss_dict[f"{name}_loss"]
            for name in self.weights
        )
