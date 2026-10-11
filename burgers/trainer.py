from dataclasses import asdict

import torch
from accelerate import Accelerator
from tqdm import tqdm

from balancer import create_balancer
from model import BurgersPINN
from utils import load_burgers_data


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

        self.t_end = self.dataset["t"][-1].item()
        self.x_start = self.dataset["x"][0].item()
        self.x_end = self.dataset["x"][-1].item()

        nx = len(self.dataset["x"])
        self.ics = {
            "t": torch.zeros(nx),
            "x": self.dataset["x"].clone(),
            "u": self.dataset["usol"][0].clone(),
        }
        nt = len(self.dataset["t"])
        self.bcs = {
            "t": self.dataset["t"].repeat(2),
            "x": torch.cat(
                [
                    torch.full((nt,), self.x_start),
                    torch.full((nt,), self.x_end),
                ]
            ),
            "u": torch.cat(
                [self.dataset["usol"][:, 0], self.dataset["usol"][:, -1]]
            ),
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
            print(f"\nFinal relative L2 error: {metrics['relative_l2']:.3e}")
            accelerator.log(metrics, step=self.config.num_epochs)
        accelerator.end_training()
        return metrics

    def evaluate(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        target = self.dataset["usol"].to(device)
        prediction = self.model.predict_solution(
            self.dataset["t"].to(device), self.dataset["x"].to(device)
        )
        return {
            "mse": torch.mean((target - prediction).square()).item(),
            "relative_l2": (
                torch.linalg.vector_norm(target - prediction)
                / torch.linalg.vector_norm(target)
            ).item(),
        }
