from dataclasses import asdict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
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

        if not 0.0 < config.train_time_fraction < config.validation_time_fraction < 1.0:
            raise ValueError(
                "Expected 0 < train_time_fraction < validation_time_fraction < 1."
            )
        self.full_t_end = self.dataset["t"][-1].item()
        self.train_t_end = self.full_t_end * config.train_time_fraction
        self.validation_t_end = self.full_t_end * config.validation_time_fraction
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

        boundary_t = self.dataset["t"][
            self.dataset["t"] <= self.train_t_end
        ].clone()
        nt = len(boundary_t)
        self.bcs = {
            "t": boundary_t,
            "x_left": torch.full((nt,), self.x_start),
            "x_right": torch.full((nt,), self.x_end),
        }

    def sample_batches(self, device):
        for _ in range(self.config.step_per_epoch):
            yield {
                "t": torch.rand(self.config.batch_size, device=device) * self.train_t_end,
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
            validation_metrics = self.evaluate(split="validation")
            if accelerator.is_main_process:
                progress.set_postfix(
                    {
                        "loss": f"{averages['total']:.3e}",
                        "interp": f"{validation_metrics['validation_interpolation_relative_l2']:.3e}",
                        "extra": f"{validation_metrics['validation_extrapolation_relative_l2']:.3e}",
                    }
                )
                accelerator.log(
                    {
                        "total_loss": averages["total"],
                        "ics_loss": averages["ics"],
                        "bcs_loss": averages["bcs"],
                        "res_loss": averages["res"],
                        **validation_metrics,
                        **{f"{name}_weight": value for name, value in self.balancer.weights.items()},
                    },
                    step=epoch,
                )

        self.model = accelerator.unwrap_model(self.model)
        metrics = self.evaluate(split="test")
        if accelerator.is_main_process:
            plot_path = self.plot_results()
            print(
                "\nFinal average relative L2 | "
                f"interpolation: {metrics['test_interpolation_relative_l2']:.3e}, "
                f"extrapolation: {metrics['test_extrapolation_relative_l2']:.3e}"
            )
            print(f"Comparison plot: {plot_path}")
            accelerator.log(metrics, step=self.config.num_epochs)
        accelerator.end_training()
        return metrics

    def evaluate(self, split="validation"):
        """Evaluate disjoint interpolation and temporal-extrapolation points."""
        if split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")

        self.model.eval()
        device = next(self.model.parameters()).device
        prediction_model = self.model.module if hasattr(self.model, "module") else self.model
        time = self.dataset["t"].to(device)
        x = self.dataset["x"].to(device)
        with torch.no_grad():
            grid_t, grid_x = torch.meshgrid(time, x, indexing="ij")
            predictions = prediction_model.u_net(
                grid_t.flatten(), grid_x.flatten()
            )

        interpolation = time <= self.train_t_end
        if split == "validation":
            extrapolation = (time > self.train_t_end) & (
                time <= self.validation_t_end
            )
        else:
            extrapolation = time > self.validation_t_end

        x_indices = torch.arange(len(x), device=device)
        space_mask = x_indices.remainder(2) == (0 if split == "validation" else 1)

        metrics = {}
        region_relative_l2 = {"interpolation": [], "extrapolation": []}
        for name, prediction in zip(("rho", "u", "p"), predictions):
            target = self.dataset[name].to(device)
            prediction = prediction.reshape_as(target)
            for region, mask in (
                ("interpolation", interpolation),
                ("extrapolation", extrapolation),
            ):
                region_target = target[mask][:, space_mask]
                error = region_target - prediction[mask][:, space_mask]
                mse = torch.mean(error.square()).item()
                relative_l2 = (
                    torch.linalg.vector_norm(error)
                    / torch.linalg.vector_norm(region_target)
                ).item()
                metrics[f"{split}_{region}_{name}_mse"] = mse
                metrics[f"{split}_{region}_{name}_relative_l2"] = relative_l2
                region_relative_l2[region].append(relative_l2)

        for region, values in region_relative_l2.items():
            metrics[f"{split}_{region}_relative_l2"] = sum(values) / len(values)
        return metrics

    def plot_results(self):
        """Save ground truth, prediction, and absolute error for each field."""
        self.model.eval()
        device = next(self.model.parameters()).device
        t = self.dataset["t"].to(device)
        x = self.dataset["x"].to(device)
        with torch.no_grad():
            grid_t, grid_x = torch.meshgrid(t, x, indexing="ij")
            predictions = self.model.u_net(grid_t.flatten(), grid_x.flatten())

        t_np = t.cpu().numpy()
        x_np = x.cpu().numpy()
        fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
        column_titles = ("Ground truth", "PINN prediction", "Absolute error")

        for row, (name, prediction) in enumerate(zip(("rho", "u", "p"), predictions)):
            truth = self.dataset[name].cpu().numpy()
            prediction = prediction.reshape_as(grid_t).cpu().numpy()
            fields = (truth, prediction, np.abs(truth - prediction))
            value_min = min(truth.min(), prediction.min())
            value_max = max(truth.max(), prediction.max())
            for column, field in enumerate(fields):
                color_limits = (
                    {"vmin": value_min, "vmax": value_max} if column < 2 else {}
                )
                image = axes[row, column].pcolormesh(
                    t_np,
                    x_np,
                    field.T,
                    shading="auto",
                    cmap="viridis" if column < 2 else "magma",
                    **color_limits,
                )
                axes[row, column].set_xlabel("Time (t)")
                axes[row, column].set_ylabel("Space (x)")
                axes[row, column].set_title(f"{name}: {column_titles[column]}")
                axes[row, column].axvline(
                    self.train_t_end, color="white", linestyle="--", linewidth=1.5
                )
                axes[row, column].axvline(
                    self.validation_t_end,
                    color="cyan",
                    linestyle=":",
                    linewidth=1.5,
                )
                fig.colorbar(image, ax=axes[row, column])

        save_path = Path("results/navier_stokes_comparison.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
