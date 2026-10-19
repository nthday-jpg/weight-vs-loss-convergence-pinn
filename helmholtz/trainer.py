from dataclasses import asdict
from pathlib import Path
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from balancer import build_balancer
from model import HelmholtzPINN
from share.artifacts import comparison_plot_path
from share.gradient_metrics import component_gradient_summary


def load_helmholtz_data(data_path: Union[str, Path]):
    """Load fields and physical constants for the Helmholtz problem."""
    path = Path(data_path)
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
    elif path.suffix == ".npz":
        data = np.load(path)
    else:
        raise ValueError("Unsupported file format. Use .pt or .npz")

    def tensor(name):
        value = data[name]
        return (
            value.float()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=torch.float32)
        )

    def scalar(name, default):
        if name not in data:
            return default
        value = data[name]
        return float(value.item() if hasattr(value, "item") else value)

    return {
        "x": tensor("x"),
        "y": tensor("y"),
        "u": tensor("u"),
        "q": tensor("q"),
        "k": scalar("k", 1.0),
        "a1": scalar("a1", 1.0),
        "a2": scalar("a2", 4.0),
    }


class Trainer:
    def __init__(self, config, data_path):
        self.config = config
        self.dataset = load_helmholtz_data(data_path)
        # Identical model initialization and deterministic host-side sampling
        # across processes. Device-specific sampling is configured after the
        # Accelerator has initialized its process index.
        set_seed(config.seed)

        self.model = HelmholtzPINN(
            config.layers,
            k=self.dataset["k"],
            a1=self.dataset["a1"],
            a2=self.dataset["a2"],
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.balancer = build_balancer(config)

        x = self.dataset["x"]
        y = self.dataset["y"]
        if len(x) < 3 or len(y) < 3:
            raise ValueError("Helmholtz data must contain at least 3 points per axis.")
        self.x_start = x[0].item()
        self.x_end = x[-1].item()
        self.y_start = y[0].item()
        self.y_end = y[-1].item()

        # Use every Dirichlet boundary point once. Interior endpoints of the
        # bottom/top edges are omitted because the left/right edges already
        # include the four corners.
        left_x = torch.full_like(y, self.x_start)
        right_x = torch.full_like(y, self.x_end)
        inner_x = x[1:-1]
        bottom_y = torch.full_like(inner_x, self.y_start)
        top_y = torch.full_like(inner_x, self.y_end)
        self.bcs = {
            "x": torch.cat([left_x, right_x, inner_x, inner_x]),
            "y": torch.cat([y, y, bottom_y, top_y]),
            "u": torch.cat(
                [
                    self.dataset["u"][0, :],
                    self.dataset["u"][-1, :],
                    self.dataset["u"][1:-1, 0],
                    self.dataset["u"][1:-1, -1],
                ]
            ),
        }

        self.data = self._sample_interior_data(config.num_data_points)
        available_loss_names = ["bcs", "res"]
        if self.data is not None:
            available_loss_names.append("data")
        if config.loss_terms is None:
            self.loss_names = available_loss_names
        else:
            self.loss_names = list(dict.fromkeys(config.loss_terms))
            unknown = set(self.loss_names) - {"bcs", "res", "data"}
            if unknown:
                raise ValueError(f"Unknown loss terms: {sorted(unknown)}")
            if not self.loss_names:
                raise ValueError("loss_terms must contain at least one loss term")
            if "data" in self.loss_names and self.data is None:
                raise ValueError("The data loss requires num_data_points > 0")

        reference = getattr(self.balancer, "reference", None)
        if reference is not None and reference not in self.loss_names:
            raise ValueError(
                f"Balancer {self.config.balancer!r} requires the {reference!r} "
                "loss term."
            )

    def _sample_interior_data(self, count):
        if count == 0:
            return None
        if count != "all" and (not isinstance(count, int) or count < 0):
            raise ValueError('num_data_points must be zero, a positive integer, or "all"')

        # Data observations must not duplicate the boundary-condition loss.
        x_indices = torch.arange(1, len(self.dataset["x"]) - 1)
        y_indices = torch.arange(1, len(self.dataset["y"]) - 1)
        candidate_count = len(x_indices) * len(y_indices)
        if count == "all":
            flat_indices = torch.arange(candidate_count)
        else:
            if count > candidate_count:
                raise ValueError(
                    f"Requested {count} data points, but only {candidate_count} are available."
                )
            generator = torch.Generator().manual_seed(self.config.seed)
            flat_indices = torch.randperm(candidate_count, generator=generator)[:count]
        selected_x = x_indices[flat_indices // len(y_indices)]
        selected_y = y_indices[flat_indices % len(y_indices)]
        return {
            "x": self.dataset["x"][selected_x],
            "y": self.dataset["y"][selected_y],
            "u": self.dataset["u"][selected_x, selected_y],
        }

    def sample_batch(self, device):
        return {
            "x": self.x_start
            + torch.rand(self.config.batch_size, device=device)
            * (self.x_end - self.x_start),
            "y": self.y_start
            + torch.rand(self.config.batch_size, device=device)
            * (self.y_end - self.y_start),
        }

    def train(self):
        accelerator = Accelerator(log_with="wandb")
        device = accelerator.device
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        self.bcs = {key: value.to(device) for key, value in self.bcs.items()}
        if self.data is not None:
            self.data = {key: value.to(device) for key, value in self.data.items()}

        init_kwargs = {}
        if self.config.wandb_run_name:
            init_kwargs["wandb"] = {"name": self.config.wandb_run_name}
        accelerator.init_trackers(
            self.config.wandb_project,
            config=asdict(self.config),
            init_kwargs=init_kwargs,
        )

        # Keep identical initial parameters but sample different collocation
        # points on each distributed worker.
        set_seed(self.config.seed, device_specific=True)

        if accelerator.is_main_process:
            print(
                "\nTraining on "
                f"{device} with k={self.dataset['k']:.6g}, "
                f"a1={self.dataset['a1']:.6g}, a2={self.dataset['a2']:.6g}"
            )

        if self.config.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.config.adapt_every_steps < 1:
            raise ValueError("adapt_every_steps must be positive")
        if self.config.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")
        progress = tqdm(
            range(1, self.config.num_steps + 1),
            desc="Training steps",
            disable=not accelerator.is_main_process,
            ncols=120,
        )

        history = []
        optimization_time = 0.0
        wall_start = time.perf_counter()
        loss_names = self.loss_names
        totals = {
            name: torch.zeros((), device=device)
            for name in ["total", *loss_names]
        }
        interval_step = 0

        self.model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_start = time.perf_counter()

        try:
            for optimizer_step in progress:
                should_log = (
                    optimizer_step % self.config.log_every_steps == 0
                    or optimizer_step == self.config.num_steps
                )
                should_adapt = (
                    optimizer_step % self.config.adapt_every_steps == 0
                    or optimizer_step == self.config.num_steps
                )
                needs_gradient_statistics = (
                    should_adapt and self.balancer.requires_gradients
                )
                needs_gradient_metrics = (
                    should_log and self.config.track_gradient_metrics
                )
                batch = self.sample_batch(device)
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.model(batch, self.bcs, self.data, loss_names=loss_names)
                total_loss = self.balancer(losses)
                # Gradient statistics must be taken before the optimizer step,
                # while the autograd graph still matches the current parameters.
                gradient_metrics = {}
                gradient_statistics = None
                if needs_gradient_metrics or needs_gradient_statistics:
                    gradient_metrics, gradient_statistics = (
                        component_gradient_summary(
                            losses, self.model.parameters(), self.balancer.weights
                        )
                    )
                    if not needs_gradient_metrics:
                        gradient_metrics = {}
                    if needs_gradient_statistics:
                        gradient_statistics = gradient_statistics.reduced(
                            accelerator
                        )
                    else:
                        gradient_statistics = None
                accelerator.backward(total_loss)
                self.optimizer.step()

                totals["total"] += total_loss.detach()
                for name in loss_names:
                    totals[name] += losses[f"{name}_loss"].detach()
                interval_step += 1

                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    optimization_time += time.perf_counter() - optimization_start
                    averages = {
                        name: accelerator.reduce(value, reduction="mean").item()
                        / interval_step
                        for name, value in totals.items()
                    }
                    gradient_metrics = {
                        name: accelerator.reduce(value, reduction="mean").item()
                        for name, value in gradient_metrics.items()
                    }
                    validation_metrics = self.evaluate(split="validation")
                    wall_time = time.perf_counter() - wall_start
                    record = {
                        "optimizer_step": optimizer_step,
                        "optimization_time_seconds": optimization_time,
                        "wall_time_seconds": wall_time,
                        **{f"{name}_loss": averages[name] for name in loss_names},
                        **gradient_metrics,
                        **validation_metrics,
                        **{
                            f"{name}_weight": value
                            for name, value in self.balancer.weights.items()
                        },
                    }
                    if accelerator.is_main_process:
                        history.append(record)
                        progress.set_postfix(
                            {
                                "loss": f"{averages['total']:.3e}",
                                "l2": f"{validation_metrics['validation_relative_l2']:.3e}",
                            }
                        )
                        accelerator.log(
                            {
                                "train/total_loss": averages["total"],
                                **{
                                    f"train/{name}_loss": averages[name]
                                    for name in loss_names
                                },
                                **{
                                    f"weights/{name}": value
                                    for name, value in self.balancer.weights.items()
                                },
                                **{
                                    f"gradient_diagnostics/{name}": value
                                    for name, value in gradient_metrics.items()
                                },
                                "validation/relative_l2": validation_metrics[
                                    "validation_relative_l2"
                                ],
                                "validation/relative_linf": validation_metrics[
                                    "validation_relative_linf"
                                ],
                            },
                            step=optimizer_step,
                        )
                    self.model.train()
                    optimization_start = time.perf_counter()

                if should_adapt:
                    if not should_log:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        optimization_time += time.perf_counter() - optimization_start
                    interval_averages = {
                        name: accelerator.reduce(value, reduction="mean").item()
                        / interval_step
                        for name, value in totals.items()
                    }
                    adaptation_start = time.perf_counter()
                    self.balancer.update(
                        {f"{name}_loss": interval_averages[name] for name in loss_names},
                        gradients=gradient_statistics,
                        step=optimizer_step,
                    )
                    optimization_time += time.perf_counter() - adaptation_start
                    totals = {
                        name: torch.zeros((), device=device)
                        for name in ["total", *loss_names]
                    }
                    interval_step = 0
                    optimization_start = time.perf_counter()
        except KeyboardInterrupt:
            self.model = accelerator.unwrap_model(self.model)
            metrics = self.evaluate(split="test")
            if accelerator.is_main_process:
                progress.close()
                plot_path = self.plot_results()
                print(
                    "\nTraining interrupted - final evaluation saved: "
                    f"{plot_path}"
                )
            accelerator.end_training()
            return metrics

        self.model = accelerator.unwrap_model(self.model)
        metrics = self.evaluate(split="test")
        if accelerator.is_main_process:
            plot_path = self.plot_results()
            self.log_convergence_to_wandb(
                accelerator, history, metrics, plot_path
            )
            print(
                "\nFinal relative L2: "
                f"{metrics['test_relative_l2']:.3e}, "
                f"relative L-infinity: {metrics['test_relative_linf']:.3e}"
            )
            print(f"Comparison plot: {plot_path}")
            print("Convergence history and plots logged to W&B")
            accelerator.log(
                {
                    "test/relative_l2": metrics["test_relative_l2"],
                    "test/relative_linf": metrics["test_relative_linf"],
                },
                step=optimizer_step + 1,
            )
        accelerator.end_training()
        return metrics

    def evaluate(self, split="validation"):
        """Compare the PINN prediction with the exact solution on the full grid."""
        if split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")

        self.model.eval()
        device = next(self.model.parameters()).device
        prediction_model = self.model.module if hasattr(self.model, "module") else self.model
        x = self.dataset["x"].to(device)
        y = self.dataset["y"].to(device)
        with torch.no_grad():
            grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
            prediction = prediction_model.u_net(
                grid_x.flatten(), grid_y.flatten()
            ).reshape_as(grid_x)
        target = self.dataset["u"].to(device)
        error = prediction - target
        return {
            f"{split}_mse": torch.mean(error.square()).item(),
            f"{split}_relative_l2": (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(target)
            ).item(),
            f"{split}_relative_linf": (
                torch.max(torch.abs(error))
                / torch.max(torch.abs(target))
            ).item(),
        }

    @staticmethod
    def log_convergence_to_wandb(accelerator, history, test_metrics, comparison_path):
        """Log the convergence table, final metrics, and key charts."""
        run = accelerator.get_tracker("wandb", unwrap=True)
        columns = list(history[0])
        history_table = wandb.Table(
            columns=columns,
            data=[[row[column] for column in columns] for row in history],
        )
        test_table = wandb.Table(
            columns=["metric", "value"],
            data=[[name, value] for name, value in sorted(test_metrics.items())],
        )
        artifacts = {
            "tables/convergence_history": history_table,
            "tables/test_metrics": test_table,
            "evaluation/comparison": wandb.Image(comparison_path),
        }
        x_axes = (
            ("optimization_time_seconds", "Optimization time (seconds)", "time"),
            ("optimizer_step", "Optimizer updates", "steps"),
        )
        for x_key, x_label, suffix in x_axes:
            artifacts[f"convergence/relative_l2_vs_{suffix}"] = wandb.plot.line_series(
                xs=[row[x_key] for row in history],
                ys=[[row["validation_relative_l2"] for row in history]],
                keys=["relative L2"],
                title=f"Relative L2 vs {x_label}",
                xname=x_label,
            )

        gradient_norm_keys = [
            key
            for key in history[0]
            if key.endswith("_gradient_norm")
            and not key.endswith("_weighted_gradient_norm")
        ]
        weighted_norm_keys = [
            key for key in history[0] if key.endswith("_weighted_gradient_norm")
        ]
        cosine_keys = [
            key for key in history[0] if key.startswith("gradient_cosine_")
        ]
        steps = [row["optimizer_step"] for row in history]
        for artifact_name, keys, title in (
            ("gradient_norms", gradient_norm_keys, "Component gradient norms"),
            (
                "weighted_gradient_norms",
                weighted_norm_keys,
                "Weighted component gradient norms",
            ),
            ("gradient_cosines", cosine_keys, "Pairwise gradient cosine similarities"),
        ):
            if keys:
                artifacts[f"gradient_diagnostics/{artifact_name}_vs_steps"] = (
                    wandb.plot.line_series(
                        xs=steps,
                        ys=[[row[key] for row in history] for key in keys],
                        keys=keys,
                        title=title,
                        xname="Optimizer updates",
                    )
                )
        run.log(artifacts)

    def plot_results(self):
        """Save ground truth, prediction, and absolute error heatmaps."""
        self.model.eval()
        device = next(self.model.parameters()).device
        x = self.dataset["x"].to(device)
        y = self.dataset["y"].to(device)
        with torch.no_grad():
            grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
            prediction = self.model.u_net(
                grid_x.flatten(), grid_y.flatten()
            ).reshape_as(grid_x)

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        truth = self.dataset["u"].cpu().numpy()
        prediction = prediction.cpu().numpy()
        error = np.abs(truth - prediction)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        value_min = min(truth.min(), prediction.min())
        value_max = max(truth.max(), prediction.max())
        for axis, field, title, cmap, limits in (
            (axes[0], truth, "Exact solution", "viridis", {"vmin": value_min, "vmax": value_max}),
            (
                axes[1],
                prediction,
                "PINN prediction",
                "viridis",
                {"vmin": value_min, "vmax": value_max},
            ),
            (axes[2], error, "Absolute error", "magma", {}),
        ):
            image = axis.pcolormesh(
                x_np, y_np, field.T, shading="auto", cmap=cmap, **limits
            )
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_title(title)
            axis.set_aspect("equal")
            fig.colorbar(image, ax=axis)

        run_name = self.config.wandb_run_name or "helmholtz"
        results_dir = Path(__file__).resolve().parent.parent / "results"
        save_path = comparison_plot_path(run_name, results_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
