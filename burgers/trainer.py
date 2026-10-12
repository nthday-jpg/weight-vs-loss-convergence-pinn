from dataclasses import asdict
import time

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from balancer import create_balancer
from model import BurgersPINN
from utils import load_burgers_data, plot_comparison


class Trainer:
    def __init__(self, config, data_path):
        self.config = config
        self.dataset = load_burgers_data(data_path, is_torch=True)
        self.dataset = {
            key: value.float() if isinstance(value, torch.Tensor) else value
            for key, value in self.dataset.items()
        }
        # Identical model initialization and deterministic host-side sampling
        # across processes. Device-specific sampling is configured after the
        # Accelerator has initialized its process index.
        set_seed(config.seed)

        nu = self.dataset["nu"]
        nu = float(nu.item() if isinstance(nu, torch.Tensor) else nu)
        self.model = BurgersPINN(config.layers, nu=nu)
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
        self.x_end = self.dataset["x"][-1].item()

        nx = len(self.dataset["x"])
        self.ics = {
            "t": torch.zeros(nx),
            "x": self.dataset["x"].clone(),
            "u": self.dataset["usol"][0].clone(),
        }
        # Boundary targets beyond the training horizon must not leak into
        # extrapolation validation or test regions.
        train_time_mask = self.dataset["t"] <= self.train_t_end
        boundary_t = self.dataset["t"][train_time_mask]
        boundary_u = self.dataset["usol"][train_time_mask]
        nt = len(boundary_t)
        self.bcs = {
            "t": boundary_t.repeat(2),
            "x": torch.cat(
                [
                    torch.full((nt,), self.x_start),
                    torch.full((nt,), self.x_end),
                ]
            ),
            "u": torch.cat([boundary_u[:, 0], boundary_u[:, -1]]),
        }
        self.data = self._sample_interior_data(config.num_data_points)

    def _sample_interior_data(self, count):
        if count == 0:
            return None
        if count < 0:
            raise ValueError("num_data_points cannot be negative")

        time_indices = torch.where(
            (self.dataset["t"] > 0) & (self.dataset["t"] <= self.train_t_end)
        )[0]
        # Exclude the two supervised boundary coordinates.
        interior_x = torch.arange(1, len(self.dataset["x"]) - 1)
        candidate_count = len(time_indices) * len(interior_x)
        if count > candidate_count:
            raise ValueError(
                f"Requested {count} data points, but only {candidate_count} are available."
            )

        generator = torch.Generator().manual_seed(self.config.seed)
        flat_indices = torch.randperm(candidate_count, generator=generator)[:count]
        selected_t = time_indices[flat_indices // len(interior_x)]
        selected_x = interior_x[flat_indices % len(interior_x)]
        return {
            "t": self.dataset["t"][selected_t],
            "x": self.dataset["x"][selected_x],
            "u": self.dataset["usol"][selected_t, selected_x],
        }

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
            print(f"\nTraining on {device} with nu={self.model.module.nu if hasattr(self.model, 'module') else self.model.nu:.6g}")

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
        loss_names = ["ics", "bcs", "res"]
        if self.data is not None:
            loss_names.append("data")
        totals = {
            name: torch.zeros((), device=device)
            for name in ["total", *loss_names]
        }
        interval_step = 0

        self.model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_start = time.perf_counter()

        for optimizer_step in progress:
            batch = self.sample_batch(device)
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.model(batch, self.ics, self.bcs, self.data)
            total_loss = self.balancer(losses)
            accelerator.backward(total_loss)
            self.optimizer.step()

            totals["total"] += total_loss.detach()
            for name in loss_names:
                totals[name] += losses[f"{name}_loss"].detach()
            interval_step += 1

            should_log = (
                optimizer_step % self.config.log_every_steps == 0
                or optimizer_step == self.config.num_steps
            )
            should_adapt = (
                optimizer_step % self.config.adapt_every_steps == 0
                or optimizer_step == self.config.num_steps
            )

            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                optimization_time += time.perf_counter() - optimization_start
                averages = {
                    name: accelerator.reduce(value, reduction="mean").item()
                    / interval_step
                    for name, value in totals.items()
                }
                validation_metrics = self.evaluate(split="validation")
                wall_time = time.perf_counter() - wall_start
                record = {
                    "optimizer_step": optimizer_step,
                    "optimization_time_seconds": optimization_time,
                    "wall_time_seconds": wall_time,
                    **{f"{name}_loss": averages[name] for name in loss_names},
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
                        {"total_loss": averages["total"], **record},
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
                    {f"{name}_loss": interval_averages[name] for name in loss_names}
                )
                optimization_time += time.perf_counter() - adaptation_start
                totals = {
                    name: torch.zeros((), device=device)
                    for name in ["total", *loss_names]
                }
                interval_step = 0
                optimization_start = time.perf_counter()

        self.model = accelerator.unwrap_model(self.model)
        metrics = self.evaluate(split="test")
        if accelerator.is_main_process:
            plot_path = self.plot_results()
            self.log_convergence_to_wandb(accelerator, history, plot_path)
            print(
                "\nFinal relative L2 | "
                f"interpolation: {metrics['test_interpolation_relative_l2']:.3e}, "
                f"extrapolation: {metrics['test_extrapolation_relative_l2']:.3e}"
            )
            print(f"Comparison plot: {plot_path}")
            print("Convergence history and plots logged to W&B")
            accelerator.log(metrics, step=optimizer_step + 1)
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
        target = self.dataset["usol"].to(device)
        prediction = prediction_model.predict_solution(
            time, self.dataset["x"].to(device)
        )

        interpolation = time <= self.train_t_end
        if split == "validation":
            extrapolation = (time > self.train_t_end) & (time <= self.validation_t_end)
        else:
            extrapolation = time > self.validation_t_end

        # Keep reference coordinates used for training-time validation disjoint
        # from those used in the final test comparison.
        x_indices = torch.arange(target.shape[1], device=device)
        space_mask = x_indices.remainder(2) == (0 if split == "validation" else 1)

        metrics = {}
        for region, mask in (("interpolation", interpolation), ("extrapolation", extrapolation)):
            region_target = target[mask][:, space_mask]
            region_prediction = prediction[mask][:, space_mask]
            error = region_target - region_prediction
            metrics[f"{split}_{region}_mse"] = torch.mean(error.square()).item()
            metrics[f"{split}_{region}_relative_l2"] = (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(region_target)
            ).item()
        return metrics

    @staticmethod
    def log_convergence_to_wandb(accelerator, history, comparison_path):
        run = accelerator.get_tracker("wandb", unwrap=True)
        columns = list(history[0])
        table = wandb.Table(
            columns=columns,
            data=[[row[column] for column in columns] for row in history],
        )
        x_axes = (
            ("optimization_time_seconds", "Optimization time (seconds)", "time"),
            ("optimizer_step", "Optimizer updates", "steps"),
        )
        artifacts = {
            "convergence/history": table,
            "evaluation/comparison": wandb.Image(comparison_path),
        }
        for x_key, x_label, suffix in x_axes:
            artifacts[f"convergence/relative_l2_vs_{suffix}"] = wandb.plot.line_series(
                xs=[row[x_key] for row in history],
                ys=[
                    [row["validation_interpolation_relative_l2"] for row in history],
                    [row["validation_extrapolation_relative_l2"] for row in history],
                ],
                keys=["interpolation", "extrapolation"],
                title=f"Relative L2 vs {x_label}",
                xname=x_label,
            )
        run.log(artifacts)

    def plot_results(self):
        """Save ground truth, prediction, and absolute error heatmaps."""
        device = next(self.model.parameters()).device
        prediction = self.model.predict_solution(
            self.dataset["t"].to(device), self.dataset["x"].to(device)
        )
        save_path = "results/burgers_comparison.png"
        plot_comparison(
            self.dataset["t"].cpu().numpy(),
            self.dataset["x"].cpu().numpy(),
            self.dataset["usol"].cpu().numpy(),
            prediction.detach().cpu().numpy(),
            save_path=save_path,
            train_t_end=self.train_t_end,
            validation_t_end=self.validation_t_end,
        )
        return save_path
