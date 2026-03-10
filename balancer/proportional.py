from .base import BaseBalancer
class ProportionalBalancer(BaseBalancer):
    def __init__(self, alpha=0.1):
        """Simple balancer with EMA smoothing.
        
        Args:
            alpha: EMA coefficient (0-1). Higher = more responsive to recent losses.
                   0.1 means 10% new weight, 90% old weight.
        """
        super().__init__()
        self.alpha = alpha
        self.weights = {
            'ics': 1.0/3.0,
            'bcs': 1.0/3.0,
            'res': 1.0/3.0
        }

    def _balance(self, loss_dict):
        """Adjust weights to equalize loss contributions with EMA smoothing."""  
        # Detach and convert to scalars
        ics_val = loss_dict['ics_loss'].detach().item()
        bcs_val = loss_dict['bcs_loss'].detach().item()
        res_val = loss_dict['res_loss'].detach().item()
        
        # Compute raw weights (proportional to losses)
        sum_losses = ics_val + bcs_val + res_val + 1e-8  
        weight_ics_raw = ics_val / sum_losses
        weight_bcs_raw = bcs_val / sum_losses
        weight_res_raw = res_val / sum_losses
        
        # Apply EMA smoothing
        self.weights['ics'] = self.alpha * weight_ics_raw + (1 - self.alpha) * self.weights['ics']
        self.weights['bcs'] = self.alpha * weight_bcs_raw + (1 - self.alpha) * self.weights['bcs']
        self.weights['res'] = self.alpha * weight_res_raw + (1 - self.alpha) * self.weights['res']
        
        # Clamp to minimum weight
        min_weight = 0.15
        self.weights['ics'] = max(self.weights['ics'], min_weight)
        self.weights['bcs'] = max(self.weights['bcs'], min_weight)
        self.weights['res'] = max(self.weights['res'], min_weight)
        
        # Renormalize to sum to 1
        weight_sum = self.weights['ics'] + self.weights['bcs'] + self.weights['res']
        self.weights['ics'] /= weight_sum
        self.weights['bcs'] /= weight_sum
        self.weights['res'] /= weight_sum
    
    def __call__(self, loss_dict):
        self._balance(loss_dict)
        return (    
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )
