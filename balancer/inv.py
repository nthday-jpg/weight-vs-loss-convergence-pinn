class Balancer:
    def __init__(self):
        self.weights = {
            'ics': 1.0,
            'bcs': 1.0,
            'res': 1.0,
        }
    
    def balance(self, loss_dict):
        # Detach and convert to scalars
        ics_val = loss_dict['ics_loss'].detach().item()
        bcs_val = loss_dict['bcs_loss'].detach().item()
        res_val = loss_dict['res_loss'].detach().item()
        
        inv_sum = 1.0 / ics_val + 1.0 / bcs_val + 1.0 / res_val + 1e-8
        self.weights['ics'] = (1.0 / ics_val) / inv_sum
        self.weights['bcs'] = (1.0 / bcs_val) / inv_sum
        self.weights['res'] = (1.0 / res_val) / inv_sum

    def __call__(self, loss_dict):
        total_loss = (
            self.weights['ics'] * loss_dict['ics_loss']
            + self.weights['bcs'] * loss_dict['bcs_loss']
            + self.weights['res'] * loss_dict['res_loss']
        )
        return total_loss