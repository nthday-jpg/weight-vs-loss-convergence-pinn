"""Diagnose gradient-annealing weights on a PINN benchmark.

Runs a physics-only training loop (no W&B, no Accelerate) and records, at every
adaptation step, the per-term gradient statistics that drive the
learning-rate-annealing rule

    lambda_i_hat = max|grad L_res| / mean|grad (lambda_i * L_i)|

together with the resulting weights, the losses, and the parameter magnitudes.

Supported problems:
    navier_stokes  periodic 1D compressible Navier-Stokes (losses: ics, bcs, res)
    helmholtz      2D Helmholtz equation (losses: bcs, res)

Usage:
    python scripts/diagnose_gradient_annealing.py --problem helmholtz \
        --steps 5000 --out results/diag_ga_helmholtz.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balancer.gradient_annealing import GradientAnnealing  # noqa: E402
from share.gradient_metrics import component_gradient_statistics  # noqa: E402


def load_model_class(problem_file: Path, module_name: str):
    """Load a problem's model module without shadowing the other problem's model."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, problem_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model module from {problem_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scalar(data, name, default):
    if name not in data:
        return default
    value = data[name]
    return float(value.item() if hasattr(value, "item") else value)


class NavierStokesProblem:
    """Periodic 1D compressible Navier-Stokes, physics-only."""

    loss_names = ["ics", "bcs", "res"]
    reference = "res"

    def __init__(
        self, dataset, seed=42, train_time_fraction=0.7, num_boundary_points=0
    ):
        module = load_model_class(
            ROOT / "navier-stokes" / "model.py", "diagnostic_navier_stokes_model"
        )

        self.dataset = dataset
        self.full_t_end = dataset["t"][-1].item()
        self.train_t_end = self.full_t_end * train_time_fraction
        self.x_start = dataset["x"][0].item()
        dx = (dataset["x"][1] - dataset["x"][0]).item()
        self.x_end = dataset["x"][-1].item() + dx

        nx = len(dataset["x"])
        self.ics = {
            "t": torch.zeros(nx),
            "x": dataset["x"].clone(),
            "rho": dataset["rho"][0].clone(),
            "u": dataset["u"][0].clone(),
            "p": dataset["p"][0].clone(),
        }
        boundary_t = dataset["t"][dataset["t"] <= self.train_t_end].clone()
        nt = len(boundary_t)
        self.bcs = {
            "t": boundary_t,
            "x_left": torch.full((nt,), self.x_start),
            "x_right": torch.full((nt,), self.x_end),
        }
        self.generator = torch.Generator().manual_seed(seed)
        self.model_kwargs = {
            "mu": dataset["mu"],
            "gamma": dataset["gamma"],
            "prandtl": dataset["prandtl"],
            "gas_constant": dataset["gas_constant"],
        }
        self.model_class = module.NavierStokesPINN

    def sample_batch(self, batch_size):
        t = torch.rand(batch_size, generator=self.generator) * self.train_t_end
        x = self.x_start + torch.rand(batch_size, generator=self.generator) * (
            self.x_end - self.x_start
        )
        return {"t": t, "x": x}

    def forward_kwargs(self, step=None):
        return {"ics": self.ics, "bcs": self.bcs, "loss_names": self.loss_names}


class HelmholtzProblem:
    """2D Helmholtz equation on (-1, 1)^2, physics-only (BC + residual)."""

    loss_names = ["bcs", "res"]
    reference = "res"

    def __init__(
        self,
        dataset,
        seed=42,
        train_time_fraction=None,
        num_boundary_points=0,
    ):
        module = load_model_class(
            ROOT / "helmholtz" / "model.py", "diagnostic_helmholtz_model"
        )

        self.dataset = dataset
        x = dataset["x"]
        y = dataset["y"]
        self.x_start, self.x_end = x[0].item(), x[-1].item()
        self.y_start, self.y_end = y[0].item(), y[-1].item()
        self.a1 = float(dataset["a1"])
        self.a2 = float(dataset["a2"])
        self.num_boundary_points = num_boundary_points

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
                    dataset["u"][0, :],
                    dataset["u"][-1, :],
                    dataset["u"][1:-1, 0],
                    dataset["u"][1:-1, -1],
                ]
            ),
        }
        self.generator = torch.Generator().manual_seed(seed)
        self.model_kwargs = {
            "k": dataset["k"],
            "a1": dataset["a1"],
            "a2": dataset["a2"],
        }
        self.model_class = module.HelmholtzPINN

    def exact_solution(self, x, y):
        return torch.sin(self.a1 * torch.pi * x) * torch.sin(
            self.a2 * torch.pi * y
        )

    def sample_batch(self, batch_size):
        x = self.x_start + torch.rand(batch_size, generator=self.generator) * (
            self.x_end - self.x_start
        )
        y = self.y_start + torch.rand(batch_size, generator=self.generator) * (
            self.y_end - self.y_start
        )
        return {"x": x, "y": y}

    def sample_boundary(self, count):
        """Resample boundary points uniformly from the four edges of the square.

        The paper samples N_b boundary points per iteration. Keeping the
        boundary set fixed (as the production trainer does) lets the network
        memorize it exactly, which drives the boundary gradient to zero and
        makes the annealing weight unbounded.
        """
        count = int(count)
        edge = torch.randint(0, 4, (count,), generator=self.generator)
        u01 = torch.rand(count, generator=self.generator)
        v01 = torch.rand(count, generator=self.generator)

        x = torch.where(
            edge == 0,
            torch.full((count,), self.x_start),
            torch.where(
                edge == 1,
                torch.full((count,), self.x_end),
                self.x_start + u01 * (self.x_end - self.x_start),
            ),
        )
        y = torch.where(
            edge == 2,
            torch.full((count,), self.y_start),
            torch.where(
                edge == 3,
                torch.full((count,), self.y_end),
                self.y_start + v01 * (self.y_end - self.y_start),
            ),
        )
        return {"x": x, "y": y, "u": self.exact_solution(x, y)}

    def forward_kwargs(self, step=None):
        if self.num_boundary_points > 0:
            bcs = self.sample_boundary(self.num_boundary_points)
        else:
            bcs = self.bcs
        return {"bcs": bcs, "data": None, "loss_names": self.loss_names}



def load_data(path: Path, problem: str):
    data = torch.load(path, map_location="cpu")
    keys = {
        "navier_stokes": ("t", "x", "rho", "u", "p"),
        "helmholtz": ("x", "y", "u", "q"),
    }[problem]
    dataset = {
        name: data[name].float() for name in keys if name in data
    }
    for name, default in (
        ("mu", 1e-2),
        ("gamma", 1.4),
        ("prandtl", 0.72),
        ("gas_constant", 1.0),
        ("k", 1.0),
        ("a1", 1.0),
        ("a2", 4.0),
    ):
        dataset[name] = _scalar(data, name, default)
    return dataset


def parameter_stats(model):
    maximum = 0.0
    squared = 0.0
    for parameter in model.parameters():
        maximum = max(maximum, parameter.detach().abs().max().item())
        squared += parameter.detach().square().sum().item()
    return maximum, squared**0.5


def layerwise_gradient_statistics(loss_dict, model):
    """Per-layer gradient statistics over weight matrices only.

    The reference implementation takes the max over per-layer maxima and the
    mean over per-layer means of the weight-matrix gradients, excluding biases.
    """
    from types import SimpleNamespace

    names = [key.removesuffix("_loss") for key in loss_dict]
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".weight") and parameter.requires_grad
    ]
    max_abs = {}
    mean_abs = {}
    for name in names:
        gradients = torch.autograd.grad(
            loss_dict[f"{name}_loss"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        available = [gradient for gradient in gradients if gradient is not None]
        max_abs[name] = torch.stack(
            [gradient.abs().max() for gradient in available]
        ).max().item()
        mean_abs[name] = torch.stack(
            [gradient.abs().mean() for gradient in available]
        ).mean().item()
    return SimpleNamespace(max_abs=max_abs, mean_abs=mean_abs)


def paper_initialize(model, generator, input_sigma):
    """Reinitialize as in the reference TF code (Xavier normal, zero bias).

    The reference code also normalizes its inputs to zero mean / unit variance,
    which for x, y ~ U(-1, 1) means dividing by sigma = 1/sqrt(3). Feeding raw
    coordinates to a network whose first layer is scaled by 1/sigma is exactly
    equivalent.
    """
    import torch.nn as nn

    for index, layer in enumerate(model.network.layers):
        if not isinstance(layer, nn.Linear):
            continue
        std = (2.0 / (layer.in_features + layer.out_features)) ** 0.5
        with torch.no_grad():
            layer.weight.normal_(0.0, std, generator=generator)
            layer.bias.zero_()
            if index == 0 and input_sigma:
                layer.weight.div_(input_sigma)


def run(args):
    torch.manual_seed(args.seed)
    dataset = load_data(Path(args.data), args.problem)
    problem_class = {
        "navier_stokes": NavierStokesProblem,
        "helmholtz": HelmholtzProblem,
    }[args.problem]
    problem = problem_class(dataset, seed=args.seed, num_boundary_points=args.num_boundary_points)
    loss_names = problem.loss_names
    reference = problem.reference

    model = problem.model_class(args.layers, **problem.model_kwargs)
    if args.paper_init:
        paper_initialize(
            model,
            generator=torch.Generator().manual_seed(args.seed),
            input_sigma=1.0 / math.sqrt(3.0),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    balancer = GradientAnnealing(
        alpha=args.alpha,
        reference=reference,
        initial_weight=args.initial_weight,
        epsilon=args.epsilon,
    )

    rows = []
    interval = {name: 0.0 for name in ["total", *loss_names]}
    interval_steps = 0
    started = time.perf_counter()

    progress = tqdm(
        range(1, args.steps + 1),
        desc=f"Training steps ({args.problem})",
        ncols=120,
        unit="step",
    )
    for step in progress:
        batch = problem.sample_batch(args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        losses = model(batch, **problem.forward_kwargs(step=step))
        if args.bcs_scale != 1.0 and "bcs_loss" in losses:
            # The reference implementation sums one mean per boundary edge.
            losses["bcs_loss"] = args.bcs_scale * losses["bcs_loss"]
        total_loss = balancer(losses)

        is_adapt = step % args.adapt_every_steps == 0 or step == args.steps
        is_log = is_adapt and (
            args.log_every_steps <= 0
            or step % args.log_every_steps == 0
            or step == args.steps
        )
        statistics = None
        if is_adapt:
            # Statistics are taken before the optimizer step, on the same graph
            # that produced the losses, exactly as the production trainer does.
            if args.paper_stats:
                statistics = layerwise_gradient_statistics(losses, model)
            else:
                statistics = component_gradient_statistics(losses, model.parameters())
                statistics.max_abs = {k: v.item() for k, v in statistics.max_abs.items()}
                statistics.mean_abs = {k: v.item() for k, v in statistics.mean_abs.items()}

        # Weights that were active for this optimizer step.
        weights_used = dict(balancer.weights)

        total_loss.backward()
        optimizer.step()
        if args.lr_decay < 1.0 and step % args.lr_decay_every == 0:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay

        interval["total"] += total_loss.detach().item()
        for name in loss_names:
            interval[name] += losses[f"{name}_loss"].detach().item()
        interval_steps += 1

        if is_adapt:
            row = {
                "step": step,
                "elapsed_s": time.perf_counter() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_loss": interval["total"] / interval_steps,
                "param_max_abs": 0.0,
                "param_l2": 0.0,
            }
            for name in loss_names:
                row[f"{name}_loss"] = interval[name] / interval_steps
                row[f"{name}_max_abs_grad"] = statistics.max_abs[name]
                row[f"{name}_mean_abs_grad"] = statistics.mean_abs[name]
            for name in loss_names:
                if name == reference:
                    row[f"{name}_weight"] = 1.0
                    row[f"{name}_target_ratio"] = float("nan")
                    continue
                row[f"{name}_weight"] = weights_used[name]
                denominator = statistics.mean_abs[name] * weights_used[name]
                row[f"{name}_target_ratio"] = statistics.max_abs[reference] / max(
                    denominator, args.epsilon
                )
            row["param_max_abs"], row["param_l2"] = parameter_stats(model)

            # The adaptive weights are refreshed after the optimizer step, so
            # they take effect on the following steps (as in the trainer).
            balancer.update(
                {f"{name}_loss": row[f"{name}_loss"] for name in loss_names},
                gradients=statistics,
                step=step,
            )
            for name in loss_names:
                row[f"{name}_weight_new"] = balancer.weights[name]
            if is_log:
                rows.append(row)

            adaptive = [name for name in loss_names if name != reference]
            progress.set_postfix(
                {
                    "res_max|g|": f"{row[f'{reference}_max_abs_grad']:.2e}",
                    **{
                        f"w_{name}": f"{row[f'{name}_weight_new']:.2e}"
                        for name in adaptive
                    },
                    "loss": f"{row['total_loss']:.2e}",
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    meta = {
        "problem": args.problem,
        "layers": args.layers,
        "steps": args.steps,
        "adapt_every_steps": args.adapt_every_steps,
        "log_every_steps": args.log_every_steps,
        "batch_size": args.batch_size,
        "num_boundary_points": args.num_boundary_points,
        "bcs_scale": args.bcs_scale,
        "learning_rate": args.learning_rate,
        "lr_decay": args.lr_decay,
        "lr_decay_every": args.lr_decay_every,
        "alpha": args.alpha,
        "seed": args.seed,
        "paper_init": args.paper_init,
        "out": str(out_path),
    }
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    adaptive = [name for name in loss_names if name != reference]
    for name in adaptive:
        weights = [row[f"{name}_weight_new"] for row in rows]
        mean_weight = sum(weights) / len(weights)
        print(
            f"{name}: mean weight={mean_weight:.4e} "
            f"final={weights[-1]:.4e} max={max(weights):.4e}"
        )
    print(f"\nwrote {out_path}")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=["navier_stokes", "helmholtz"], default="navier_stokes"
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--adapt-every-steps", type=int, default=100)
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=0,
        help="Record a CSV row every this many steps (0 = every adapt step).",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--num-boundary-points",
        type=int,
        default=0,
        help="Helmholtz: resample this many boundary points per step (0 = fixed grid).",
    )
    parser.add_argument(
        "--bcs-scale",
        type=float,
        default=1.0,
        help="Multiplier on the boundary loss (reference code sums one mean per edge).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=1.0,
        help="Multiplicative learning-rate decay applied every --lr-decay-every steps.",
    )
    parser.add_argument("--lr-decay-every", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--initial-weight", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paper-init",
        action="store_true",
        help="Xavier-normal weights, zero biases, and normalized-input first layer.",
    )
    parser.add_argument(
        "--paper-stats",
        action="store_true",
        help="Reference gradient stats: weight matrices only, equal weight per layer.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.data is None:
        args.data = str(
            ROOT
            / ("navier-stokes" if args.problem == "navier_stokes" else "helmholtz")
            / "data"
            / (
                "navier_stokes_1d.pt"
                if args.problem == "navier_stokes"
                else "helmholtz.pt"
            )
        )
    if args.layers is None:
        args.layers = (
            [2, 30, 30, 3] if args.problem == "navier_stokes" else [2, 50, 50, 50, 1]
        )
    if args.out is None:
        args.out = f"results/diag_ga_{args.problem}.csv"
    return args


if __name__ == "__main__":
    run(parse_args())
