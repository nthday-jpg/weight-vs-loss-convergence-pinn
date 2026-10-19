"""PyTorch PINN for the two-dimensional Helmholtz equation."""

import torch
import torch.nn as nn

from share import MLP


class HelmholtzPINN(nn.Module):
    """Solve Laplacian(u) + k^2 u = q on a rectangular domain.

    The network maps spatial coordinates (x, y) to u. The source term q is
    evaluated analytically from the manufactured solution parameters a1 and a2,
    which must match the parameters used to generate the reference dataset.
    """

    def __init__(self, layers: list[int], k: float = 1.0, a1: float = 1.0, a2: float = 4.0):
        super().__init__()
        if layers[0] != 2 or layers[-1] != 1:
            raise ValueError(
                "HelmholtzPINN requires two inputs (x, y) and one output u; "
                f"received layers={layers}."
            )
        self.network = MLP(layers)
        self.k = float(k)
        self.a1 = float(a1)
        self.a2 = float(a2)

    def u_net(self, x, y):
        return self.network(torch.stack([x, y], dim=-1)).squeeze(-1)

    @staticmethod
    def _derivative(output, input_):
        return torch.autograd.grad(
            output,
            input_,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]

    def source_term(self, x, y):
        """Return the manufactured source q for the exact Helmholtz solution."""
        pi = torch.pi
        return (
            self.k**2 - (self.a1 * pi) ** 2 - (self.a2 * pi) ** 2
        ) * torch.sin(self.a1 * pi * x) * torch.sin(self.a2 * pi * y)

    def r_net(self, u, x, y):
        u_x = self._derivative(u, x)
        u_y = self._derivative(u, y)
        u_xx = self._derivative(u_x, x)
        u_yy = self._derivative(u_y, y)
        return u_xx + u_yy + self.k**2 * u - self.source_term(x, y)

    def forward(self, batch, bcs, data=None, loss_names=None):
        requested = set(loss_names or ("bcs", "res", "data"))
        losses = {}
        if "bcs" in requested:
            losses["bcs_loss"] = torch.mean(
                (self.u_net(bcs["x"], bcs["y"]) - bcs["u"]).square()
            )
        if "res" in requested:
            x = batch["x"].requires_grad_(True)
            y = batch["y"].requires_grad_(True)
            u = self.u_net(x, y)
            losses["res_loss"] = torch.mean(self.r_net(u, x, y).square())
        if "data" in requested and data is not None:
            losses["data_loss"] = torch.mean(
                (self.u_net(data["x"], data["y"]) - data["u"]).square()
            )
        return losses

    def compute_loss(self, batch, bcs, data=None, loss_names=None):
        """Backward-compatible alias for direct, non-distributed use."""
        return self.forward(batch, bcs, data, loss_names)

    def predict_solution(self, x, y):
        self.eval()
        with torch.no_grad():
            grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
            return self.u_net(grid_x.flatten(), grid_y.flatten()).reshape_as(grid_x)
