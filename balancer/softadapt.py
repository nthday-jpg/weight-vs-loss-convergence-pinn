from .registry import register_balancer
from .base import BaseBalancer
import math

@register_balancer('softadapt')
class SoftAdaptBalancer(BaseBalancer):
    def __init__(self, beta=0.1, ema_alpha=0.05):  
        super().__init__()
        self.beta = beta
        self.ema_alpha = ema_alpha
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
        
        self.prev_losses = current_losses.copy()
        return rates

    def _update_weights(self, rates):
        max_rate = max(rates.values())
        exp_rates = {k: math.exp((v-max_rate) * self.beta) for k, v in rates.items()}
        sum_exp = sum(exp_rates.values())

        target_weights = {k: v / sum_exp for k, v in exp_rates.items()}
        self.weights = {
            k: self.ema_alpha * target_weights[k] + (1 - self.ema_alpha) * self.weights[k]
            for k in self.weights
        }
    
    def __call__(self, loss_dict):
        rates = self._compute_rate_of_change(loss_dict)
        self._update_weights(rates)
            
        return (
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )

