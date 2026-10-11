"""PINN for the periodic one-dimensional compressible Navier-Stokes equations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from share import MLP


class NavierStokesPINN(nn.Module):
    def __init__(
        self,
        layers: list[int],
        mu: float,
        gamma: float = 1.4,
        prandtl: float = 0.72,
        gas_constant: float = 1.0,
    ):
        super().__init__()
        if layers[0] != 2 or layers[-1] != 3:
            raise ValueError(
                "NavierStokesPINN requires two inputs (t, x) and three outputs "
                f"(rho, u, p); received layers={layers}."
            )

        self.network = MLP(layers)
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.prandtl = float(prandtl)
        self.gas_constant = float(gas_constant)

    @staticmethod
    def _derivative(output, input_):
        # create_graph=True is required so the PDE loss can train network parameters.
        return torch.autograd.grad(
            output,
            input_,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]

    def u_net(self, t, x):
        """Predict primitive variables [rho, u, p] from (t, x)."""
        output = self.network(torch.stack([t, x], dim=-1))
        rho_raw, u_raw, p_raw = output.unbind(dim=-1)
        rho = F.softplus(rho_raw) + 1e-6
        p = F.softplus(p_raw) + 1e-6
        return rho, u_raw, p

    def r_net(self, rho, u, p, t, x):
        """Return mass, momentum, and total-energy residuals."""
        cv = self.gas_constant / (self.gamma - 1.0)
        cp = self.gamma * cv
        kappa = self.mu * cp / self.prandtl

        momentum = rho * u
        rho_t = self._derivative(rho, t)
        momentum_x = self._derivative(momentum, x)
        r_mass = rho_t + momentum_x

        u_x = self._derivative(u, x)
        tau = (4.0 / 3.0) * self.mu * u_x
        momentum_t = self._derivative(momentum, t)
        momentum_flux_x = self._derivative(rho * u.square() + p, x)
        tau_x = self._derivative(tau, x)
        r_momentum = momentum_t + momentum_flux_x - tau_x

        energy = p / (self.gamma - 1.0) + 0.5 * rho * u.square()
        energy_t = self._derivative(energy, t)
        energy_flux_x = self._derivative(u * (energy + p), x)
        temperature = p / (rho * self.gas_constant)
        heat_flux = -kappa * self._derivative(temperature, x)
        viscous_heat_flux_x = self._derivative(u * tau - heat_flux, x)
        r_energy = energy_t + energy_flux_x - viscous_heat_flux_x

        return r_mass, r_momentum, r_energy

    @staticmethod
    def _primitive_mse(data, predictions):
        return sum(
            torch.mean((prediction - data[name].reshape(-1)).square())
            for name, prediction in zip(("rho", "u", "p"), predictions)
        ) / 3.0

    def _periodic_boundary_loss(self, bcs):
        x_left = bcs["x_left"].requires_grad_(True)
        x_right = bcs["x_right"].requires_grad_(True)
        left = self.u_net(bcs["t"], x_left)
        right = self.u_net(bcs["t"], x_right)

        value_loss = sum(
            torch.mean((left_value - right_value).square())
            for left_value, right_value in zip(left, right)
        ) / 3.0
        derivative_loss = sum(
            torch.mean(
                (self._derivative(left_value, x_left) - self._derivative(right_value, x_right)).square()
            )
            for left_value, right_value in zip(left, right)
        ) / 3.0
        return 0.5 * (value_loss + derivative_loss)

    def forward(self, batch, ics, bcs):
        t = batch["t"].requires_grad_(True)
        x = batch["x"].requires_grad_(True)
        primitive = self.u_net(t, x)
        residuals = self.r_net(*primitive, t, x)

        return {
            "ics_loss": self._primitive_mse(
                ics, self.u_net(ics["t"], ics["x"])
            ),
            "bcs_loss": self._periodic_boundary_loss(bcs),
            "res_loss": sum(torch.mean(residual.square()) for residual in residuals) / 3.0,
        }

    def compute_loss(self, batch, ics, bcs):
        """Backward-compatible alias for direct, non-distributed use."""
        return self.forward(batch, ics, bcs)
