from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from balancer import create_balancer
from model import NavierStokesPINN


def load_navier_stokes_data(data_path: Union[str, Path]):
    """Load primitive solution fields and physical constants from .pt or .npz."""
    path = Path(data_path)
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
    elif path.suffix == ".npz":
        data = np.load(path)
    else:
        raise ValueError("Unsupported file format. Use .pt or .npz")

    def tensor(name):
        value = data[name]
        return value.float() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)

    def scalar(name, default):
        if name not in data:
            return default
        value = data[name]
        return float(value.item() if hasattr(value, "item") else value)

    return {
        "t": tensor("t"),
        "x": tensor("x"),
        "rho": tensor("rho"),
        "u": tensor("u"),
        "p": tensor("p"),
        "mu": scalar("mu", 1e-2),
        "gamma": scalar("gamma", 1.4),
        "prandtl": scalar("prandtl", 0.72),
        "gas_constant": scalar("gas_constant", 1.0),
    }


class Trainer:
    def __init__(self, config, data_path):
        self.config = config
        self.dataset = load_navier_stokes_data(data_path)
        torch.manual_seed(config.seed)

        self.model = NavierStokesPINN(
            config.layers,
            mu=self.dataset["mu"],
            gamma=self.dataset["gamma"],
            prandtl=self.dataset["prandtl"],
            gas_constant=self.dataset["gas_constant"],
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.balancer = create_balancer(
            config.balancer_type,
            **getattr(config, f"{config.balancer_type}_params", {}),
        )

        self.t_end = self.dataset["t"][-1].item()
        self.x_start = self.dataset["x"][0].item()
        dx = (self.dataset["x"][1] - self.dataset["x"][0]).item()
        # Generated periodic grids omit the repeated right endpoint.
        self.x_end = self.dataset["x"][-1].item() + dx

        nx = len(self.dataset["x"])
        self.ics = {
            "t": torch.zeros(nx),
            "x": self.dataset["x"].clone(),
            "rho": self.dataset["rho"][0].clone(),
            "u": self.dataset["u"][0].clone(),
            "p": self.dataset["p"][0].clone(),
        }

        nt = len(self.dataset["t"])
        self.bcs = {
            "t": self.dataset["t"].clone(),
            "x_left": torch.full((nt,), self.x_start),
            "x_right": torch.full((nt,), self.x_end),
        }

    def sample_batches(self, device):
        for _ in range(self.config.step_per_epoch):
            yield {
                "t": torch.rand(self.config.batch_size, device=device) * self.t_end,
                "x": self.x_start
                + torch.rand(self.config.batch_size, device=device)
                * (self.x_end - self.x_start),
            }

    def train(self):
        accelerator = Accelerator(log_with="wandb")
        device = accelerator.device
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        self.ics = {key: value.to(device) for key, value in self.ics.items()}
        self.bcs = {key: value.to(device) for key, value in self.bcs.items()}

        init_kwargs = {}
        if self.config.wandb_run_name:
            init_kwargs["wandb"] = {"name": self.config.wandb_run_name}
        accelerator.init_trackers(
            self.config.wandb_project,
            config=asdict(self.config),
            init_kwargs=init_kwargs,
        )

        if accelerator.is_main_process:
            print(f"\nTraining on {device} with mu={self.dataset['mu']:.6g}")

        progress = tqdm(
            range(self.config.num_epochs),
            desc="Training",
            disable=not accelerator.is_main_process,
            ncols=120,
        )
        for epoch in progress:
            self.model.train()
            totals = {name: torch.zeros((), device=device) for name in ("total", "ics", "bcs", "res")}

            for batch in self.sample_batches(device):
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.model(batch, self.ics, self.bcs)
                total_loss = self.balancer(losses)
                accelerator.backward(total_loss)
                self.optimizer.step()

                totals["total"] += total_loss.detach()
                for name in ("ics", "bcs", "res"):
                    totals[name] += losses[f"{name}_loss"].detach()

            averages = {
                name: accelerator.reduce(value, reduction="mean").item()
                / self.config.step_per_epoch
                for name, value in totals.items()
            }
            if accelerator.is_main_process:
                progress.set_postfix({name: f"{value:.3e}" for name, value in averages.items()})
                accelerator.log(
                    {
                        "total_loss": averages["total"],
                        "ics_loss": averages["ics"],
                        "bcs_loss": averages["bcs"],
                        "res_loss": averages["res"],
                        **{f"{name}_weight": value for name, value in self.balancer.weights.items()},
                    },
                    step=epoch,
                )

        self.model = accelerator.unwrap_model(self.model)
        metrics = self.evaluate()
        if accelerator.is_main_process:
            print(
                "\nFinal relative L2 | "
                f"rho: {metrics['rho_relative_l2']:.3e}, "
                f"u: {metrics['u_relative_l2']:.3e}, "
                f"p: {metrics['p_relative_l2']:.3e}"
            )
            accelerator.log(metrics, step=self.config.num_epochs)
        accelerator.end_training()
        return metrics

    def evaluate(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        t, x = self.dataset["t"].to(device), self.dataset["x"].to(device)
        with torch.no_grad():
            grid_t, grid_x = torch.meshgrid(t, x, indexing="ij")
            predictions = self.model.u_net(grid_t.flatten(), grid_x.flatten())

        metrics = {}
        for name, prediction in zip(("rho", "u", "p"), predictions):
            target = self.dataset[name].to(device)
            prediction = prediction.reshape_as(target)
            metrics[f"{name}_mse"] = torch.mean((target - prediction).square()).item()
            metrics[f"{name}_relative_l2"] = (
                torch.linalg.vector_norm(target - prediction)
                / torch.linalg.vector_norm(target)
            ).item()
        return metrics
