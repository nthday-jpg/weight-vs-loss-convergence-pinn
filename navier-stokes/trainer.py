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
from model import NavierStokesPINN
from share.artifacts import comparison_plot_path
from share.gradient_metrics import (
    component_gradient_metrics,
    component_gradient_statistics,
    component_gradient_summary,
)


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
        # Identical model initialization and deterministic host-side sampling
        # across processes. Device-specific sampling is configured after the
        # Accelerator has initialized its process index.
        set_seed(config.seed)

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
        self.balancer = build_balancer(config)

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
        if config.loss_terms is None:
            self.loss_names = ["ics", "bcs", "res"]
        else:
            self.loss_names = list(dict.fromkeys(config.loss_terms))
            unknown = set(self.loss_names) - {"ics", "bcs", "res"}
            if unknown:
                raise ValueError(f"Unknown loss terms: {sorted(unknown)}")
            if not self.loss_names:
                raise ValueError("loss_terms must contain at least one loss term")

        reference = getattr(self.balancer, "reference", None)
        if reference is not None and reference not in self.loss_names:
            raise ValueError(
                f"Balancer {self.config.balancer!r} requires the {reference!r} "
                "loss term."
            )

    def sample_batch(self, device):
        return {
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
            # Keep everything in run history/plots only; do not populate the
            # run summary with tables, charts, or scalar metrics.
            run = accelerator.get_tracker("wandb", unwrap=True)
            run.define_metric("*", summary="none")

        # Keep identical initial parameters but sample different collocation
        # points on each distributed worker.
        set_seed(self.config.seed, device_specific=True)

        if accelerator.is_main_process:
            print(f"\nTraining on {device} with mu={self.dataset['mu']:.6g}")

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
                # Gradient diagnostics are always tracked so the W&B run
                # records norms, cosine similarities, and histograms.
                needs_gradient_metrics = should_log
                batch = self.sample_batch(device)
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.model(
                    batch, self.ics, self.bcs, loss_names=loss_names
                )
                total_loss = self.balancer(losses)
                # Gradient statistics must be taken before the optimizer step,
                # while the autograd graph still matches the current parameters.
                gradient_metrics = {}
                gradient_histograms = {}
                gradient_statistics = None
                if needs_gradient_statistics:
                    if needs_gradient_metrics:
                        (
                            gradient_metrics,
                            gradient_statistics,
                            gradient_histograms,
                        ) = component_gradient_summary(
                            losses,
                            self.model.parameters(),
                            self.balancer.weights,
                            return_histograms=True,
                        )
                    else:
                        gradient_statistics = component_gradient_statistics(
                            losses, self.model.parameters()
                        )
                    gradient_statistics = gradient_statistics.reduced(accelerator)
                elif needs_gradient_metrics:
                    gradient_metrics, gradient_histograms = component_gradient_metrics(
                        losses,
                        self.model.parameters(),
                        self.balancer.weights,
                        return_histograms=True,
                    )
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
                                "interp": f"{validation_metrics['validation_interpolation_relative_l2']:.3e}",
                                "extra": f"{validation_metrics['validation_extrapolation_relative_l2']:.3e}",
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
                                    f"gradient/{name}": value
                                    for name, value in gradient_metrics.items()
                                },
                                **{
                                    f"gradient/{name}_gradient_histogram": (
                                        wandb.Histogram(
                                            values.float().cpu().numpy(), num_bins=64
                                        )
                                    )
                                    for name, values in gradient_histograms.items()
                                },
                                "validation/interpolation_relative_l2": validation_metrics[
                                    "validation_interpolation_relative_l2"
                                ],
                                "validation/extrapolation_relative_l2": validation_metrics[
                                    "validation_extrapolation_relative_l2"
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
                "\nFinal average relative L2 | "
                f"interpolation: {metrics['test_interpolation_relative_l2']:.3e}, "
                f"extrapolation: {metrics['test_extrapolation_relative_l2']:.3e}"
            )
            print(f"Comparison plot: {plot_path}")
            print("Convergence history and plots logged to W&B")
            accelerator.log(
                {
                    "test/interpolation_relative_l2": metrics[
                        "test_interpolation_relative_l2"
                    ],
                    "test/extrapolation_relative_l2": metrics[
                        "test_extrapolation_relative_l2"
                    ],
                },
                step=optimizer_step + 1,
            )
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

        targets = {
            name: self.dataset[name].to(device) for name in ("rho", "u", "p")
        }
        prediction_fields = {
            name: prediction.reshape_as(targets[name])
            for name, prediction in zip(("rho", "u", "p"), predictions)
        }
        evaluation_dx = (self.x_end - self.x_start) / int(space_mask.sum())

        metrics = {}
        region_field_metrics = {"interpolation": {}, "extrapolation": {}}
        for region, time_mask in (
            ("interpolation", interpolation),
            ("extrapolation", extrapolation),
        ):
            region_targets = {
                name: value[time_mask][:, space_mask]
                for name, value in targets.items()
            }
            region_predictions = {
                name: value[time_mask][:, space_mask]
                for name, value in prediction_fields.items()
            }

            for name in ("rho", "u", "p"):
                field_metrics = self._field_metrics(
                    region_targets[name], region_predictions[name], evaluation_dx
                )
                metrics.update(
                    {
                        f"{split}_{region}_{name}_{metric}": value
                        for metric, value in field_metrics.items()
                    }
                )
                for metric, value in field_metrics.items():
                    region_field_metrics[region].setdefault(metric, []).append(value)

            conservative_metrics = self._conservative_metrics(
                region_targets, region_predictions, evaluation_dx
            )
            metrics.update(
                {
                    f"{split}_{region}_{metric}": value
                    for metric, value in conservative_metrics.items()
                }
            )

        for region, field_metrics in region_field_metrics.items():
            for metric, values in field_metrics.items():
                metrics[f"{split}_{region}_{metric}"] = sum(values) / len(values)
        return metrics

    @staticmethod
    def _field_metrics(target, prediction, dx):
        """Metrics sensitive to amplitudes, sharp gradients, and frequencies."""
        epsilon = torch.finfo(target.dtype).eps
        error = prediction - target

        target_gradient = (
            torch.roll(target, -1, dims=-1) - torch.roll(target, 1, dims=-1)
        ) / (2.0 * dx)
        prediction_gradient = (
            torch.roll(prediction, -1, dims=-1)
            - torch.roll(prediction, 1, dims=-1)
        ) / (2.0 * dx)
        gradient_error = prediction_gradient - target_gradient

        target_spectrum = torch.fft.rfft(target, dim=-1)
        prediction_spectrum = torch.fft.rfft(prediction, dim=-1)
        spectral_error = prediction_spectrum - target_spectrum
        high_frequency_start = max(1, (2 * target_spectrum.shape[-1]) // 3)

        target_tv = torch.sum(
            torch.abs(torch.roll(target, -1, dims=-1) - target), dim=-1
        )
        prediction_tv = torch.sum(
            torch.abs(torch.roll(prediction, -1, dims=-1) - prediction), dim=-1
        )

        centered_target = target - torch.mean(target)
        return {
            "mse": torch.mean(error.square()).item(),
            "relative_l1": (
                torch.sum(torch.abs(error))
                / (torch.sum(torch.abs(target)) + epsilon)
            ).item(),
            "relative_l2": (
                torch.linalg.vector_norm(error)
                / (torch.linalg.vector_norm(target) + epsilon)
            ).item(),
            "relative_linf": (
                torch.max(torch.abs(error))
                / (torch.max(torch.abs(target)) + epsilon)
            ).item(),
            "r2": (
                1.0
                - torch.sum(error.square())
                / (torch.sum(centered_target.square()) + epsilon)
            ).item(),
            "gradient_relative_l2": (
                torch.linalg.vector_norm(gradient_error)
                / (torch.linalg.vector_norm(target_gradient) + epsilon)
            ).item(),
            "gradient_linf": torch.max(torch.abs(gradient_error)).item(),
            "spectral_relative_l2": (
                torch.linalg.vector_norm(spectral_error)
                / (torch.linalg.vector_norm(target_spectrum) + epsilon)
            ).item(),
            "high_frequency_error_ratio": (
                torch.linalg.vector_norm(spectral_error[..., high_frequency_start:])
                / (torch.linalg.vector_norm(target_spectrum) + epsilon)
            ).item(),
            "high_frequency_energy_fraction_error": torch.abs(
                torch.linalg.vector_norm(
                    prediction_spectrum[..., high_frequency_start:]
                )
                / (torch.linalg.vector_norm(prediction_spectrum) + epsilon)
                - torch.linalg.vector_norm(
                    target_spectrum[..., high_frequency_start:]
                )
                / (torch.linalg.vector_norm(target_spectrum) + epsilon)
            ).item(),
            "total_variation_relative_error": (
                torch.sum(torch.abs(prediction_tv - target_tv))
                / (torch.sum(torch.abs(target_tv)) + epsilon)
            ).item(),
        }

    def _conservative_metrics(self, target, prediction, dx):
        """Compare spatial integrals of mass, momentum, and total energy."""
        epsilon = torch.finfo(target["rho"].dtype).eps

        def conserved(fields):
            rho, velocity, pressure = fields["rho"], fields["u"], fields["p"]
            return {
                "mass": rho,
                "momentum": rho * velocity,
                "energy": pressure / (self.dataset["gamma"] - 1.0)
                + 0.5 * rho * velocity.square(),
            }

        target_conserved = conserved(target)
        prediction_conserved = conserved(prediction)
        metrics = {}
        for name in ("mass", "momentum", "energy"):
            target_integral = target_conserved[name].sum(dim=-1) * dx
            prediction_integral = prediction_conserved[name].sum(dim=-1) * dx
            integral_error = prediction_integral - target_integral
            # Normalize by the integral of |q| rather than the signed integral;
            # total momentum can legitimately be close to zero.
            target_scale = target_conserved[name].abs().sum(dim=-1) * dx
            metrics[f"{name}_integral_rmse"] = torch.sqrt(
                torch.mean(integral_error.square())
            ).item()
            metrics[f"{name}_integral_normalized_l2"] = (
                torch.linalg.vector_norm(integral_error)
                / (torch.linalg.vector_norm(target_scale) + epsilon)
            ).item()
        return metrics

    @staticmethod
    def log_convergence_to_wandb(accelerator, history, test_metrics, comparison_path):
        """Log full data as tables but expose only four readable charts."""
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

        def two_region_chart(metric, x_key, x_label, title):
            return wandb.plot.line_series(
                xs=[row[x_key] for row in history],
                ys=[
                    [row[f"validation_interpolation_{metric}"] for row in history],
                    [row[f"validation_extrapolation_{metric}"] for row in history],
                ],
                keys=["interpolation", "extrapolation"],
                title=title,
                xname=x_label,
            )

        artifacts["convergence/relative_l2_vs_time"] = two_region_chart(
            "relative_l2",
            "optimization_time_seconds",
            "Optimization time (seconds)",
            "Relative L2 vs optimization time",
        )
        artifacts["convergence/relative_l2_vs_steps"] = two_region_chart(
            "relative_l2", "optimizer_step", "Optimizer updates", "Relative L2 vs steps"
        )
        artifacts["convergence/r2_vs_steps"] = two_region_chart(
            "r2", "optimizer_step", "Optimizer updates", "R2 vs steps"
        )

        shock_metrics = (
            ("gradient_relative_l2", "gradient"),
            ("high_frequency_error_ratio", "high-frequency"),
            ("total_variation_relative_error", "total-variation"),
        )
        artifacts["convergence/shock_metrics_vs_steps"] = wandb.plot.line_series(
            xs=[row["optimizer_step"] for row in history],
            ys=[
                [row[f"validation_{region}_{metric}"] for row in history]
                for metric, _ in shock_metrics
                for region in ("interpolation", "extrapolation")
            ],
            keys=[
                f"{region}-{label}"
                for _, label in shock_metrics
                for region in ("interpolation", "extrapolation")
            ],
            title="Shock-sensitive metrics vs optimizer updates",
            xname="Optimizer updates",
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
                artifacts[f"gradient/{artifact_name}_vs_steps"] = (
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

        run_name = self.config.wandb_run_name or "navier-stokes"
        results_dir = Path(__file__).resolve().parent.parent / "results"
        save_path = comparison_plot_path(run_name, results_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
