class Balancer:
    def __init__(self, alpha=0.1):
        """Inverse balancer with EMA smoothing.
        
        Args:
            alpha: EMA coefficient (0-1). Higher = more responsive to recent losses.
                   0.1 means 10% new weight, 90% old weight.
        """
        self.alpha = alpha
        self.weights = {
            'ics': 1.0/3.0,
            'bcs': 1.0/3.0,
            'res': 1.0/3.0,
        }
    
    def balance(self, loss_dict):
        """Adjust weights inversely to losses with max-normalization and clamping."""
        ics_val = loss_dict['ics_loss'].detach().item()
        bcs_val = loss_dict['bcs_loss'].detach().item()
        res_val = loss_dict['res_loss'].detach().item()
        
        # Normalize by max to keep values in [0, 1]
        max_loss = max(ics_val, bcs_val, res_val) + 1e-8
        
        # Inverse of normalized losses
        inv_ics = 1.0 / (ics_val / max_loss + 1e-2)
        inv_bcs = 1.0 / (bcs_val / max_loss + 1e-2)
        inv_res = 1.0 / (res_val / max_loss + 1e-2)
        
        inv_sum = inv_ics + inv_bcs + inv_res
        weight_ics_raw = inv_ics / inv_sum
        weight_bcs_raw = inv_bcs / inv_sum
        weight_res_raw = inv_res / inv_sum
        
        # Apply EMA smoothing
        self.weights['ics'] = self.alpha * weight_ics_raw + (1 - self.alpha) * self.weights['ics']
        self.weights['bcs'] = self.alpha * weight_bcs_raw + (1 - self.alpha) * self.weights['bcs']
        self.weights['res'] = self.alpha * weight_res_raw + (1 - self.alpha) * self.weights['res']
        
        # Clamp weights to minimum threshold
        min_weight = 0.15
        self.weights['ics'] = max(self.weights['ics'], min_weight)
        self.weights['bcs'] = max(self.weights['bcs'], min_weight)
        self.weights['res'] = max(self.weights['res'], min_weight)
        
        # Renormalize to sum to 1.0
        weight_sum = self.weights['ics'] + self.weights['bcs'] + self.weights['res']
        self.weights['ics'] /= weight_sum
        self.weights['bcs'] /= weight_sum
        self.weights['res'] /= weight_sum

    def __call__(self, loss_dict):
        self.balance(loss_dict)
        return (
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )