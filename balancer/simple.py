class Balancer():
    def __init__(self):
        self.weights = {
            'ics': 1.0,
            'bcs': 1.0,
            'res': 1.0
        }

    def balance(self, loss_dict):
        """Adjust weights to equalize loss contributions."""  
        # Detach and convert to scalars
        ics_val = loss_dict['ics_loss'].detach().item()
        bcs_val = loss_dict['bcs_loss'].detach().item()
        res_val = loss_dict['res_loss'].detach().item()
        
        sum_losses = ics_val + bcs_val + res_val + 1e-8  
        self.weights['ics'] = ics_val / sum_losses
        self.weights['bcs'] = bcs_val / sum_losses
        self.weights['res'] = res_val / sum_losses
    
    def __call__(self, loss_dict):
        self.balance(loss_dict)
        return (    
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )
