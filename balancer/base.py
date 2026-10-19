from abc import ABC, abstractmethod

class BaseBalancer(ABC):
    # True when update() needs per-term gradient statistics, not just losses.
    requires_gradients = False

    def __init__(self):
        self.weights = {}

    @abstractmethod
    def __call__(self, loss_dict):
        """Compute the weighted optimizer loss without changing weights."""
        pass

    def update(self, loss_dict, *, gradients=None, step=None):
        """Update state after one adaptation interval.

        args:
            loss_dict: detached interval-mean scalars keyed "<term>_loss".
            gradients: GradientStats, present only when requires_gradients is
                True.
            step: optimizer step at the end of the interval.
        """
        return None
