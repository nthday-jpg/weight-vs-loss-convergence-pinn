from .base import BaseBalancer
import math

class SoftAdaptBalancer(BaseBalancer):
    def __init__(self, beta=1.0):  # beta is temperature parameter
        super().__init__()
        self.beta = beta
        self.weights = {'ics': 1.0/3.0, 'bcs': 1.0/3.0, 'res': 1.0/3.0}
        self.prev_losses = {}  
        
    def _compute_rate_of_change(self, loss_dict):
        current_losses = {
            'ics': loss_dict['ics_loss'].detach().item(),
            'bcs': loss_dict['bcs_loss'].detach().item(),
            'res': loss_dict['res_loss'].detach().item(),
        }
        
        prev = self.prev_losses or current_losses
        
        rates = {
            k: (current_losses[k] - prev[k]) / (prev[k] + 1e-8)
            for k in current_losses
        }
        
        self.prev_losses = current_losses
        return rates
    
    def __call__(self, loss_dict):
        rates = self._compute_rate_of_change(loss_dict)
        
        exp_rates = {k: math.exp(v * self.beta) for k, v in rates.items()}
        sum_exp = sum(exp_rates.values())
        
        self.weights = {k: v / sum_exp for k, v in exp_rates.items()}
        
        return (
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )
