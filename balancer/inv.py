class Balancer:
    def __init__(self, alpha=1):
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
        """Adjust weights inversely to losses with EMA smoothing."""
        # Detach and convert to scalars
        ics_val = loss_dict['ics_loss'].detach().item()
        bcs_val = loss_dict['bcs_loss'].detach().item()
        res_val = loss_dict['res_loss'].detach().item()
        
        # Compute raw inverse weights
        inv_sum = 1.0 / (ics_val + 1e-8) + 1.0 / (bcs_val + 1e-8) + 1.0 / (res_val + 1e-8)
        weight_ics_raw = (1.0 / (ics_val + 1e-8)) / inv_sum
        weight_bcs_raw = (1.0 / (bcs_val + 1e-8)) / inv_sum
        weight_res_raw = (1.0 / (res_val + 1e-8)) / inv_sum
        
        # Apply EMA smoothing
        self.weights['ics'] = self.alpha * weight_ics_raw + (1 - self.alpha) * self.weights['ics']
        self.weights['bcs'] = self.alpha * weight_bcs_raw + (1 - self.alpha) * self.weights['bcs']
        self.weights['res'] = self.alpha * weight_res_raw + (1 - self.alpha) * self.weights['res']

    def __call__(self, loss_dict):
        self.balance(loss_dict)
        return (
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )