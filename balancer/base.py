from abc import ABC, abstractmethod

class BaseBalancer(ABC):
    def __init__(self):
        self.weights = {}

    @abstractmethod
    def __call__(self, loss_dict):
        """Compute the weighted optimizer loss without changing weights."""
        pass

    def update(self, loss_dict):
        """Update state from aggregated losses; fixed balancers do nothing."""
        return None
