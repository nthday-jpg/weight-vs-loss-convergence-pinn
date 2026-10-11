from dataclasses import asdict

import torch
from accelerate import Accelerator
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
        torch.manual_seed(config.seed)

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
            print(f"\nTraining on {device} with nu={self.model.module.nu if hasattr(self.model, 'module') else self.model.nu:.6g}")

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
                "\nFinal relative L2 | "
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
        target = self.dataset["usol"].to(device)
        prediction = prediction_model.predict_solution(
            time, self.dataset["x"].to(device)
        )

        interpolation = time <= self.train_t_end
        if split == "validation":
            extrapolation = (time > self.train_t_end) & (time <= self.validation_t_end)
        else:
            extrapolation = time > self.validation_t_end

        # Keep reference coordinates used for epoch-level validation disjoint
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
