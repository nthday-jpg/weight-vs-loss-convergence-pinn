import torch.nn as nn
import torch
import torch.nn.functional as F

from share import MLP

"""
    Compressible 1D navier stokes equations
"""

class NavierStokesPINN(nn.Module):
    def __init__(self, layers, mu=0.01/3.141592653589793):
        super().__init__()
        self.network = MLP(layers)
        self.mu = mu

    def r_net(self, rho, u, p, t, x):
        """PDE residuals for compressible Navier-Stokes."""
        gamma = 1.4
        gas_constant = 1.0
        prandtl = 0.72
        mu = self.mu
        cv = gas_constant / (gamma - 1.0)
        cp = gamma * cv
        kappa = mu * cp / prandtl

        rho_t = torch.autograd.grad(rho, t, 
                          torch.ones_like(rho), 
                          retain_graph=True) [0]

        u_x = torch.autograd.grad(u, x, 
                                torch.ones_like(u), 
                                create_graph=True,    
                                retain_graph=True)[0]

        momentum = rho * u
        # Mass residual d(rho)/dt + d(rho*u)/dx = 0
        momentum_x = torch.autograd.grad(
            momentum,
            x,
            torch.ones_like(momentum),
            create_graph=False,
            retain_graph=True
        )[0]
        r1 = rho_t + momentum_x

        # Momentum residual: d(rho*u)/dt + d(rho*u^2 + p)/dx - d(tau)/dx = 0
        momentum_t = torch.autograd.grad(
            momentum,
            t,
            torch.ones_like(momentum),
            create_graph=True,
            retain_graph=True,
        )[0]
        momentum_flux = rho * u**2 + p
        momentum_flux_x = torch.autograd.grad(
            momentum_flux,
            x,
            torch.ones_like(momentum_flux),
            create_graph=False,
            retain_graph=True,
        )[0]
        tau = (4.0 / 3.0) * mu * u_x
        tau_x = torch.autograd.grad(
            tau,
            x,
            torch.ones_like(tau),
            create_graph=False,
            retain_graph=True,
        )[0]
        r2 = momentum_t + momentum_flux_x - tau_x

        # Total-energy residual:
        # dE/dt + d(u(E+p))/dx - d(u*tau - q)/dx = 0, q = -kappa*dT/dx
        rho_safe = torch.clamp(rho, min=1e-6)
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        E_t = torch.autograd.grad(
            E,
            t,
            torch.ones_like(E),
            create_graph=False,
            retain_graph=True,
        )[0]
        energy_flux = u * (E + p)
        energy_flux_x = torch.autograd.grad(
            energy_flux,
            x,
            torch.ones_like(energy_flux),
            create_graph=False,
            retain_graph=True,
        )[0]
        T = p / (rho_safe * gas_constant)
        T_x = torch.autograd.grad(
            T,
            x,
            torch.ones_like(T),
            create_graph=False,
            retain_graph=True,
        )[0]
        q = -kappa * T_x
        viscous_heat_flux = u * tau - q
        viscous_heat_flux_x = torch.autograd.grad(
            viscous_heat_flux,
            x,
            torch.ones_like(viscous_heat_flux),
            create_graph=False,
            retain_graph=True,
        )[0]
        r3 = E_t + energy_flux_x - viscous_heat_flux_x

        return r1, r2, r3

    def u_net(self, t, x):
        """Predict [rho, u, p] from (t, x)"""
        z = torch.stack([t, x], dim=-1) # (batch_size, 2)
        output = self.network(z)
        rho_raw, u_raw, p_raw = output.split(1, dim=-1)
        rho = F.softplus(rho_raw).squeeze(-1) + 1e-6
        u = u_raw.squeeze(-1)
        p = F.softplus(p_raw).squeeze(-1) + 1e-6
        return rho, u, p

    def compute_loss(self, batch, ics, bcs):
        t = batch['t'].requires_grad_(True)
        x = batch['x'].requires_grad_(True)

        rho, u, p = self.u_net(t, x)

        # Residual loss
        r1, r2, r3 = self.r_net(rho, u, p, t, x)
        residual_loss = torch.mean(r1**2 + r2**2 + r3**2)

        def _supervised_loss(data, rho_pred, u_pred, p_pred):
            losses = []

            if 'rho' in data:
                losses.append(torch.mean((rho_pred - data['rho'].reshape(-1))**2))
            if 'u' in data:
                losses.append(torch.mean((u_pred - data['u'].reshape(-1))**2))
            if 'p' in data:
                losses.append(torch.mean((p_pred - data['p'].reshape(-1))**2))

            # Support stacked targets with conservative ordering [rho, rho*u, E]
            # or primitive ordering [rho, u, p] depending on provided key.
            if not losses:
                if 'primitive' in data:
                    target = data['primitive']
                    losses.append(torch.mean((rho_pred - target[:, 0].reshape(-1))**2))
                    losses.append(torch.mean((u_pred - target[:, 1].reshape(-1))**2))
                    losses.append(torch.mean((p_pred - target[:, 2].reshape(-1))**2))
                elif 'q' in data:
                    target = data['q']
                    losses.append(torch.mean((rho_pred - target[:, 0].reshape(-1))**2))
                    losses.append(torch.mean((u_pred - target[:, 1].reshape(-1))**2))
                    losses.append(torch.mean((p_pred - target[:, 2].reshape(-1))**2))

            if not losses:
                return torch.tensor(0.0, device=rho_pred.device, dtype=rho_pred.dtype)

            return sum(losses) / len(losses)

        rho_ics, u_ics, p_ics = self.u_net(ics['t'], ics['x'])
        rho_bcs, u_bcs, p_bcs = self.u_net(bcs['t'], bcs['x'])

        loss_ics = _supervised_loss(ics, rho_ics, u_ics, p_ics)
        loss_bcs = _supervised_loss(bcs, rho_bcs, u_bcs, p_bcs)

        loss_dict = {
            'ics_loss': loss_ics,
            'bcs_loss': loss_bcs,
            'res_loss': residual_loss,
        }

        return loss_dict

        
        