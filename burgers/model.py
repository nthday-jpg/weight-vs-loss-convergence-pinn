"""
PyTorch implementation of Burgers equation PINN model.
Converted from JAXpi implementation.
"""

import torch
import torch.nn as nn
from share import MLP

class BurgersPINN(nn.Module):
    def __init__(self, layers: list, nu=0.01/3.141592653589793):
        super().__init__()
        
        self.network = MLP(layers)
        self.nu = nu

    def u_net(self, t, x):
        """Forward pass through network."""
        # Ensure inputs require grad for autodiff
        if not t.requires_grad:
            t = t.requires_grad_(True)
        if not x.requires_grad:
            x = x.requires_grad_(True)
            
        z = torch.stack([t, x], dim=-1)
        u = self.network(z)
        return u.squeeze(-1)
    
    def r_net(self, u, t, x):
        """PDE residual: u_t + u*u_x - nu*u_xx."""
        u_t = torch.autograd.grad(u, t, 
                                  torch.ones_like(u), 
                                  create_graph=True,    
                                  retain_graph=True)[0]   
        u_x = torch.autograd.grad(u, x, 
                                  torch.ones_like(u), 
                                  create_graph=True,    
                                  retain_graph=True)[0]
      
        u_xx = torch.autograd.grad(u_x, x, 
                               torch.ones_like(u_x),
                               create_graph=True)[0]  
      
        # Burgers equation residual
        residual = u_t + u * u_x - self.nu * u_xx
        return residual
    
    def compute_loss(self, batch, ics, bcs):
        t_int = batch['t'].requires_grad_(True)
        x_int = batch['x'].requires_grad_(True)

        u_pred = self.u_net(t_int, x_int)
        residual = self.r_net(u_pred, t_int, x_int)
        loss_res = (residual ** 2).mean()

        u_ics_pred = self.u_net(ics['t'], ics['x'])
        loss_ics = ((u_ics_pred - ics['u']) ** 2).mean()

        u_bc_pred = self.u_net(bcs['t'], bcs['x'])
        loss_bcs = ((u_bc_pred - bcs['u']) ** 2).mean()

        loss_dict = {
            'ics_loss': loss_ics,
            'bcs_loss': loss_bcs,
            'res_loss': loss_res,
        }

        return loss_dict
        
    def predict_solution(self, t, x):
        """Predict solution on grid."""
        self.eval()
        with torch.no_grad():
            # Create meshgrid
            T, X = torch.meshgrid(t, x, indexing='ij')
            t_flat = T.flatten()
            x_flat = X.flatten()
            
            u_flat = self.u_net(t_flat, x_flat)
            u = u_flat.reshape(T.shape)
        
        return u