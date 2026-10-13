"""PyTorch PINN for the one-dimensional viscous Burgers equation."""

import torch
import torch.nn as nn

from share import MLP


class BurgersPINN(nn.Module):
    def __init__(self, layers: list[int], nu: float):
        super().__init__()
        if layers[0] != 2 or layers[-1] != 1:
            raise ValueError(
                "BurgersPINN requires two inputs (t, x) and one output u; "
                f"received layers={layers}."
            )
        self.network = MLP(layers)
        self.nu = float(nu)

    def u_net(self, t, x):
        return self.network(torch.stack([t, x], dim=-1)).squeeze(-1)

    @staticmethod
    def _derivative(output, input_):
        return torch.autograd.grad(
            output,
            input_,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]

    def r_net(self, u, t, x):
        u_t = self._derivative(u, t)
        u_x = self._derivative(u, x)
        u_xx = self._derivative(u_x, x)
        return u_t + u * u_x - self.nu * u_xx

    def forward(self, batch, ics, bcs, data=None, loss_names=None):
        requested = set(loss_names or ("ics", "bcs", "res", "data"))
        losses = {}
        if "ics" in requested:
            losses["ics_loss"] = torch.mean(
                (self.u_net(ics["t"], ics["x"]) - ics["u"]).square()
            )
        if "bcs" in requested:
            losses["bcs_loss"] = torch.mean(
                (self.u_net(bcs["t"], bcs["x"]) - bcs["u"]).square()
            )
        if "res" in requested:
            t = batch["t"].requires_grad_(True)
            x = batch["x"].requires_grad_(True)
            u = self.u_net(t, x)
            losses["res_loss"] = torch.mean(self.r_net(u, t, x).square())
        if "data" in requested and data is not None:
            losses["data_loss"] = torch.mean(
                (self.u_net(data["t"], data["x"]) - data["u"]).square()
            )
        return losses

    def compute_loss(self, batch, ics, bcs, data=None, loss_names=None):
        """Backward-compatible alias for direct, non-distributed use."""
        return self.forward(batch, ics, bcs, data, loss_names)

    def predict_solution(self, t, x):
        self.eval()
        with torch.no_grad():
            grid_t, grid_x = torch.meshgrid(t, x, indexing="ij")
            return self.u_net(grid_t.flatten(), grid_x.flatten()).reshape_as(grid_t)
