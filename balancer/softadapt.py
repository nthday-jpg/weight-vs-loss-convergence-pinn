from .registry import register_balancer
from .base import BaseBalancer
import math

@register_balancer('softadapt')
class SoftAdaptBalancer(BaseBalancer):
    def __init__(self, beta=0.1, ema_alpha=0.05, update_interval=1):  # beta is temperature parameter
        super().__init__()
        self.beta = beta
        self.update_interval = update_interval
        self.ema_alpha = ema_alpha  
        self.weights = {'ics': 1.0/3.0, 'bcs': 1.0/3.0, 'res': 1.0/3.0}
        self.prev_losses = {}
        self.ema_losses = {}  
        self.step = 0 
        
    def _compute_rate_of_change(self, loss_dict):
        current_losses = {
            'ics': loss_dict['ics_loss'].detach().item(),
            'bcs': loss_dict['bcs_loss'].detach().item(),
            'res': loss_dict['res_loss'].detach().item(),
        }
        
        if not self.ema_losses:
            self.ema_losses = current_losses.copy()
        else:
            self.ema_losses = {
                k: self.ema_alpha * current_losses[k] + (1 - self.ema_alpha) * self.ema_losses[k]
                for k in current_losses
            }
        
        prev = self.prev_losses or self.ema_losses
        
        rates = {
            k: (self.ema_losses[k] - prev[k]) / (prev[k] + 1e-8)
            for k in self.ema_losses
        }
        
        self.prev_losses = self.ema_losses.copy()
        return rates

    def _update_weights(self, rates):
        max_rate = max(rates.values())
        exp_rates = {k: math.exp((v-max_rate) * self.beta) for k, v in rates.items()}
        sum_exp = sum(exp_rates.values())
            
        self.weights = {k: v / sum_exp for k, v in exp_rates.items()}
    
    def __call__(self, loss_dict):
        rates = self._compute_rate_of_change(loss_dict)

        self.step += 1
        if self.step % self.update_interval == 0:
            self._update_weights(rates)
            
        return (
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )

