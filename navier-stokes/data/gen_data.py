import time
import torch
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path
import argparse


def primitive_to_conservative(rho, u, p, gamma=1.4):
	"""Convert primitive variables to conservative form."""
	momentum = rho * u
	energy = p / (gamma - 1.0) + 0.5 * rho * u**2
	return np.stack([rho, momentum, energy], axis=0)


def conservative_to_primitive(U, gamma=1.4, rho_floor=1e-8, p_floor=1e-8):
	"""Convert conservative variables [rho, rho*u, E] to primitive [rho, u, p]."""
	rho = np.maximum(U[0], rho_floor)
	u = U[1] / rho
	kinetic = 0.5 * rho * u**2
	p = (gamma - 1.0) * (U[2] - kinetic)
	p = np.maximum(p, p_floor)
	return rho, u, p


def ddx_periodic(q, dx):
	"""First derivative with periodic central difference."""
	return (np.roll(q, -1, axis=-1) - np.roll(q, 1, axis=-1)) / (2.0 * dx)


def build_initial_condition(x, gamma=1.4):
	"""Smooth periodic initial condition suitable for FFT-friendly domains."""
	rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
	u = 0.5 * np.sin(2.0 * np.pi * x + 0.35)
	p = 1.0 + 0.2 * np.cos(2.0 * np.pi * x)
	return primitive_to_conservative(rho, u, p, gamma=gamma)


def solve_compressible_1d_navier_stokes(
	nx=256,
	steps=200,
	t_final=0.6,
	gamma=1.4,
	mu=1e-2,
	prandtl=0.72,
	gas_constant=1.0,
):
	"""
	Solve periodic 1D compressible Navier-Stokes equations in conservative form.

	Unknowns are U = [rho, rho*u, E] on x in [0, 1).
	"""
	x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float64)
	dx = x[1] - x[0]

	U0 = build_initial_condition(x, gamma=gamma)
	y0 = U0.reshape(-1)

	cv = gas_constant / (gamma - 1.0)
	cp = gamma * cv
	kappa = mu * cp / prandtl

	def rhs(_, y):
		U = y.reshape(3, nx)
		rho, u, p = conservative_to_primitive(U, gamma=gamma)

		# Inviscid flux F(U)
		F1 = rho * u
		F2 = rho * u**2 + p
		F3 = u * (U[2] + p)

		dF1_dx = ddx_periodic(F1, dx)
		dF2_dx = ddx_periodic(F2, dx)
		dF3_dx = ddx_periodic(F3, dx)

		# Viscous/heat terms
		ux = ddx_periodic(u, dx)
		tau = (4.0 / 3.0) * mu * ux

		T = p / (rho * gas_constant)
		Tx = ddx_periodic(T, dx)
		q = -kappa * Tx

		dtau_dx = ddx_periodic(tau, dx)
		d_visc_energy_dx = ddx_periodic(u * tau - q, dx)

		drho_dt = -dF1_dx
		dmom_dt = -dF2_dx + dtau_dx
		dE_dt = -dF3_dx + d_visc_energy_dx

		return np.stack([drho_dt, dmom_dt, dE_dt], axis=0).reshape(-1)

	t_eval = np.linspace(0.0, t_final, steps + 1, dtype=np.float64)

	print(f"Solving periodic 1D compressible Navier-Stokes (nx={nx}, steps={steps})")
	print(
		f"gamma={gamma:.4f}, mu={mu:.3e}, Pr={prandtl:.3f}, R={gas_constant:.3f}, t_final={t_final:.3f}"
	)

	sol = solve_ivp(
		rhs,
		(0.0, t_final),
		y0,
		t_eval=t_eval,
		method="RK45",
		rtol=1e-6,
		atol=1e-8,
	)

	if not sol.success:
		raise RuntimeError(f"Time integration failed: {sol.message}")

	qsol = sol.y.T.reshape(steps + 1, 3, nx).transpose(0, 2, 1)

	rho = np.maximum(qsol[:, :, 0], 1e-8)
	u = qsol[:, :, 1] / rho
	p = np.maximum((gamma - 1.0) * (qsol[:, :, 2] - 0.5 * rho * u**2), 1e-8)

	return {
		"t": sol.t.astype(np.float32),
		"x": x.astype(np.float32),
		"qsol": qsol.astype(np.float32),
		"rho": rho.astype(np.float32),
		"u": u.astype(np.float32),
		"p": p.astype(np.float32),
		"gamma": np.float32(gamma),
		"mu": np.float32(mu),
		"prandtl": np.float32(prandtl),
		"gas_constant": np.float32(gas_constant),
	}


def save_data(data, save_dir="navier-stokes/data", base_name="navier_stokes_1d"):
	"""Save generated data in both .pt and .npz formats."""
	save_path = Path(save_dir)
	save_path.mkdir(parents=True, exist_ok=True)

	npz_path = save_path / f"{base_name}.npz"
	pt_path = save_path / f"{base_name}.pt"

	np.savez(npz_path, **data)
	print(f"Saved NumPy data to {npz_path}")

	pt_data = {
		k: torch.tensor(v) if isinstance(v, np.ndarray) or np.isscalar(v) else v
		for k, v in data.items()
	}
	torch.save(pt_data, pt_path)
	print(f"Saved PyTorch data to {pt_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Generate periodic 1D compressible Navier-Stokes data"
	)
	parser.add_argument("--nx", type=int, default=256, help="Number of spatial points")
	parser.add_argument("--steps", type=int, default=200, help="Number of output time steps")
	parser.add_argument("--t_final", type=float, default=0.6, help="Final simulation time")
	parser.add_argument("--gamma", type=float, default=1.4, help="Heat-capacity ratio")
	parser.add_argument("--mu", type=float, default=1e-2, help="Dynamic viscosity")
	parser.add_argument("--prandtl", type=float, default=0.72, help="Prandtl number")
	parser.add_argument("--gas_constant", type=float, default=1.0, help="Gas constant")
	parser.add_argument(
		"--save_dir", type=str, default="navier-stokes/data", help="Output directory"
	)
	parser.add_argument(
		"--base_name",
		type=str,
		default="navier_stokes_1d",
		help="Base filename without extension",
	)
	args = parser.parse_args()

	start_time = time.time()

	data = solve_compressible_1d_navier_stokes(
		nx=args.nx,
		steps=args.steps,
		t_final=args.t_final,
		gamma=args.gamma,
		mu=args.mu,
		prandtl=args.prandtl,
		gas_constant=args.gas_constant,
	)

	save_data(data, save_dir=args.save_dir, base_name=args.base_name)

	elapsed = time.time() - start_time

	print("\nDone!")
	print(f"qsol shape: {data['qsol'].shape}  # (nt, nx, 3)")
	print(f"rho range: [{data['rho'].min():.6f}, {data['rho'].max():.6f}]")
	print(f"u range:   [{data['u'].min():.6f}, {data['u'].max():.6f}]")
	print(f"p range:   [{data['p'].min():.6f}, {data['p'].max():.6f}]")
	print(f"Total time: {elapsed:.2f} s")


if __name__ == "__main__":
	main()

