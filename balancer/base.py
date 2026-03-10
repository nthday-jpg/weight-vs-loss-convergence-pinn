from abc import ABC, abstractmethod

class BaseBalancer(ABC):
    def __init__(self):
        self.weights = {}

    @abstractmethod
    def __call__(self, loss_dict):
        """ Compute weighted total loss
        """
        pass
