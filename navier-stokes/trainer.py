from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from balancer import create_balancer
from model import NavierStokesPINN


def load_navier_stokes_data(data_path: Union[str, Path], is_torch: bool = True):
	"""Load Navier-Stokes data from .pt or .npz."""
	path_obj = Path(data_path)
	if path_obj.suffix == ".pt":
		data = torch.load(path_obj)
	elif path_obj.suffix == ".npz":
		data = np.load(path_obj)
	else:
		raise ValueError("Unsupported file format. Use .pt or .npz")

	t = data["t"]
	x = data["x"]
	rho = data["rho"]
	u = data["u"]
	p = data["p"]

	if is_torch and not isinstance(t, torch.Tensor):
		t = torch.from_numpy(t)
		x = torch.from_numpy(x)
		rho = torch.from_numpy(rho)
		u = torch.from_numpy(u)
		p = torch.from_numpy(p)
	elif not is_torch and isinstance(t, torch.Tensor):
		t = t.numpy()
		x = x.numpy()
		rho = rho.numpy()
		u = u.numpy()
		p = p.numpy()

	return {
		"t": t.float() if isinstance(t, torch.Tensor) else t,
		"x": x.float() if isinstance(x, torch.Tensor) else x,
		"rho": rho.float() if isinstance(rho, torch.Tensor) else rho,
		"u": u.float() if isinstance(u, torch.Tensor) else u,
		"p": p.float() if isinstance(p, torch.Tensor) else p,
	}


class Trainer:
	def __init__(self, config, data_path):
		self.config = config
		self.num_epochs = config.num_epochs
		self.step_per_epoch = config.step_per_epoch
		self.batch_size = config.batch_size

		self.model = NavierStokesPINN(config.layers)
		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=config.learning_rate,
			weight_decay=config.l2_reg,
		)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,
			mode="min",
			factor=0.1,
			patience=100,
		)

		self.balancer = create_balancer(
			config.balancer_type,
			**getattr(config, f"{config.balancer_type}_params", {}),
		)
		self.dataset = load_navier_stokes_data(data_path, is_torch=True)

		self.t_end = self.dataset["t"][-1].item()
		self.x_start = self.dataset["x"][0].item()
		self.x_end = self.dataset["x"][-1].item()

		# Initial conditions: (rho, u, p)(t=0, x) for all x
		nx = len(self.dataset["x"])
		self.ics = {
			"t": torch.zeros(nx),
			"x": self.dataset["x"].clone(),
			"rho": self.dataset["rho"][0, :].clone(),
			"u": self.dataset["u"][0, :].clone(),
			"p": self.dataset["p"][0, :].clone(),
		}

		# Boundary conditions: values at x_left and x_right for all t
		nt = len(self.dataset["t"])
		x_left = self.dataset["x"][0]
		x_right = self.dataset["x"][-1]
		self.bcs = {
			"t": self.dataset["t"].repeat(2),
			"x": torch.cat([
				torch.full((nt,), x_left),
				torch.full((nt,), x_right),
			]),
			"rho": torch.cat([
				self.dataset["rho"][:, 0],
				self.dataset["rho"][:, -1],
			]),
			"u": torch.cat([
				self.dataset["u"][:, 0],
				self.dataset["u"][:, -1],
			]),
			"p": torch.cat([
				self.dataset["p"][:, 0],
				self.dataset["p"][:, -1],
			]),
		}

	def sample_batches(self, device):
		for _ in range(self.step_per_epoch):
			t = torch.rand(self.batch_size, device=device) * self.t_end
			x = torch.rand(self.batch_size, device=device) * (self.x_end - self.x_start) + self.x_start
			yield {"t": t, "x": x}

	def train(self):
		def _reduced_scalar(value):
			reduced = accelerator.reduce(value, reduction="sum")
			if isinstance(reduced, torch.Tensor):
				return reduced.item()
			if isinstance(reduced, (int, float)):
				return float(reduced)
			raise TypeError(f"Unexpected reduced value type: {type(reduced)}")

		accelerator = Accelerator(log_with="wandb")
		device = accelerator.device
		is_main_process = accelerator.is_main_process

		self.model, self.optimizer, self.scheduler = accelerator.prepare(
			self.model, self.optimizer, self.scheduler
		)

		self.ics = {k: v.to(device).requires_grad_(False) for k, v in self.ics.items()}
		self.bcs = {k: v.to(device).requires_grad_(False) for k, v in self.bcs.items()}

		init_kwargs = {}
		wandb_init = {}
		if self.config.wandb_run_name is not None:
			wandb_init["name"] = self.config.wandb_run_name
		if self.config.wandb_tags is not None:
			wandb_init["tags"] = self.config.wandb_tags
		if wandb_init:
			init_kwargs["wandb"] = wandb_init

		accelerator.init_trackers(
			project_name=self.config.wandb_project,
			config=asdict(self.config),
			init_kwargs=init_kwargs,
		)

		if is_main_process:
			print(f"\nStarting training for {self.config.num_epochs} epochs...")
			print(f"Device: {device}")
			print(f"Learning rate: {self.config.learning_rate}")
			print(f"Using Accelerate with {accelerator.num_processes} process(es)")

		pbar = tqdm(
			range(self.config.num_epochs),
			desc="Training",
			disable=not is_main_process,
			ncols=120,
		)
		for epoch in pbar:
			self.model.train()

			total_loss_epoch = torch.tensor(0.0, device=device)
			ics_loss_epoch = torch.tensor(0.0, device=device)
			bcs_loss_epoch = torch.tensor(0.0, device=device)
			res_loss_epoch = torch.tensor(0.0, device=device)
			total_samples = torch.tensor(0, device=device)

			for batch in self.sample_batches(device):
				self.optimizer.zero_grad()

				loss_dict = accelerator.unwrap_model(self.model).compute_loss(
					batch, self.ics, self.bcs
				)
				total_loss = self.balancer(loss_dict)

				total_loss_epoch += total_loss.detach() * self.batch_size
				ics_loss_epoch += loss_dict["ics_loss"].detach() * self.batch_size
				bcs_loss_epoch += loss_dict["bcs_loss"].detach() * self.batch_size
				res_loss_epoch += loss_dict["res_loss"].detach() * self.batch_size
				total_samples += self.batch_size

				accelerator.backward(total_loss)
				if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
					accelerator.clip_grad_norm_(
						self.model.parameters(),
						self.config.max_grad_norm,
					)
				self.optimizer.step()

			total_loss_epoch = _reduced_scalar(total_loss_epoch)
			total_samples = _reduced_scalar(total_samples)
			total_loss_avg = total_loss_epoch / total_samples
			self.scheduler.step(total_loss_avg)

			if epoch % self.config.log_interval == 0:
				ics_loss_epoch = _reduced_scalar(ics_loss_epoch)
				bcs_loss_epoch = _reduced_scalar(bcs_loss_epoch)
				res_loss_epoch = _reduced_scalar(res_loss_epoch)

				ics_loss_avg = ics_loss_epoch / total_samples
				bcs_loss_avg = bcs_loss_epoch / total_samples
				res_loss_avg = res_loss_epoch / total_samples
				unweighted_total = ics_loss_avg + bcs_loss_avg + res_loss_avg

				if is_main_process:
					pbar.set_postfix(
						{
							"total": f"{total_loss_avg:.3e}",
							"ics": f"{ics_loss_avg:.3e}",
							"bcs": f"{bcs_loss_avg:.3e}",
							"res": f"{res_loss_avg:.3e}",
						}
					)
					accelerator.log(
						{
							"total_loss": total_loss_avg,
							"total_loss_unweighted": unweighted_total,
							"ics_loss": ics_loss_avg,
							"bcs_loss": bcs_loss_avg,
							"res_loss": res_loss_avg,
							"ics_weight": self.balancer.weights["ics"],
							"bcs_weight": self.balancer.weights["bcs"],
							"res_weight": self.balancer.weights["res"],
						}
					)

		self.model = accelerator.unwrap_model(self.model)
		metrics = self.evaluate()
		if is_main_process:
			print(
				"\nFinal grid MSE | "
				f"rho: {metrics['rho_mse']:.3e}, "
				f"u: {metrics['u_mse']:.3e}, "
				f"p: {metrics['p_mse']:.3e}, "
				f"avg: {metrics['avg_mse']:.3e}"
			)
			accelerator.log(metrics)

		accelerator.end_training()

	def evaluate(self):
		self.model.eval()
		device = next(self.model.parameters()).device

		t = self.dataset["t"].to(device)
		x = self.dataset["x"].to(device)
		rho_true = self.dataset["rho"].to(device)
		u_true = self.dataset["u"].to(device)
		p_true = self.dataset["p"].to(device)

		with torch.no_grad():
			T, X = torch.meshgrid(t, x, indexing="ij")
			t_flat = T.flatten()
			x_flat = X.flatten()
			rho_pred, u_pred, p_pred = self.model.u_net(t_flat, x_flat)

			rho_pred = rho_pred.reshape_as(T)
			u_pred = u_pred.reshape_as(T)
			p_pred = p_pred.reshape_as(T)

			rho_mse = torch.mean((rho_true - rho_pred) ** 2).item()
			u_mse = torch.mean((u_true - u_pred) ** 2).item()
			p_mse = torch.mean((p_true - p_pred) ** 2).item()

		return {
			"rho_mse": rho_mse,
			"u_mse": u_mse,
			"p_mse": p_mse,
			"avg_mse": (rho_mse + u_mse + p_mse) / 3.0,
		}
