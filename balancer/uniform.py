from .base import BaseBalancer
from .registry import register_balancer

@register_balancer('uniform')
class UniformBalancer(BaseBalancer):
    def __init__(self):
        super().__init__()
        self.weights = {
            'ics': 1.0/3.0,
            'bcs': 1.0/3.0,
            'res': 1.0/3.0
        }
    
    def __call__(self, loss_dict):
        total_loss = (
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )
        return total_loss