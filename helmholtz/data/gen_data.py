"""Generate analytic reference data for the two-dimensional Helmholtz problem.

The manufactured solution is

    u(x, y) = sin(a1 * pi * x) * sin(a2 * pi * y)

on the domain (-1, 1)^2. Substituting it into

    Laplacian(u) + k^2 u = q

gives the source term

    q(x, y) = (k^2 - (a1*pi)^2 - (a2*pi)^2) * u(x, y).

The default parameters match the Helmholtz benchmark from Raissi et al.:
a1 = 1, a2 = 4, with k = 1.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def exact_solution(x, y, a1=1.0, a2=4.0):
    """Return the manufactured Helmholtz solution on the given coordinates."""
    return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)


def source_term(x, y, k=1.0, a1=1.0, a2=4.0):
    """Return q such that Laplacian(u) + k^2 u = q for the exact solution."""
    u = exact_solution(x, y, a1, a2)
    return (k**2 - (a1 * np.pi) ** 2 - (a2 * np.pi) ** 2) * u


def generate_helmholtz_data(
    nx=256,
    ny=256,
    k=1.0,
    a1=1.0,
    a2=4.0,
    x_min=-1.0,
    x_max=1.0,
    y_min=-1.0,
    y_max=1.0,
):
    """Sample the exact solution and source term on a rectangular grid.

    Fields are stored with the x coordinate as the first axis, so ``u[i, j]``
    corresponds to ``(x[i], y[j])``. Plot with ``pcolormesh(x, y, u.T)``.
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be at least 2.")
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("Expected x_min < x_max and y_min < y_max.")

    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    y = np.linspace(y_min, y_max, ny, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")

    u = exact_solution(grid_x, grid_y, a1, a2).astype(np.float32)
    q = source_term(grid_x, grid_y, k, a1, a2).astype(np.float32)

    return {
        "x": x,
        "y": y,
        "u": u,
        "q": q,
        "k": np.float32(k),
        "a1": np.float32(a1),
        "a2": np.float32(a2),
    }


def save_data(data, save_dir="helmholtz/data", base_name="helmholtz"):
    """Save generated data in both .npz and .pt formats."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    npz_path = save_path / f"{base_name}.npz"
    pt_path = save_path / f"{base_name}.pt"

    np.savez(npz_path, **data)
    print(f"Saved NumPy data to {npz_path}")

    torch.save(
        {key: torch.as_tensor(value) for key, value in data.items()}, pt_path
    )
    print(f"Saved PyTorch data to {pt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate analytic data for the 2D Helmholtz equation"
    )
    parser.add_argument("--nx", type=int, default=256, help="Number of x points")
    parser.add_argument("--ny", type=int, default=256, help="Number of y points")
    parser.add_argument("--k", type=float, default=1.0, help="Wavenumber")
    parser.add_argument("--a1", type=float, default=1.0, help="x-mode frequency")
    parser.add_argument("--a2", type=float, default=4.0, help="y-mode frequency")
    parser.add_argument(
        "--save_dir", type=str, default="helmholtz/data", help="Output directory"
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default="helmholtz",
        help="Base filename without extension",
    )
    args = parser.parse_args()

    start_time = time.time()
    data = generate_helmholtz_data(
        nx=args.nx,
        ny=args.ny,
        k=args.k,
        a1=args.a1,
        a2=args.a2,
    )
    save_data(data, save_dir=args.save_dir, base_name=args.base_name)

    print("\nDone!")
    print(f"u shape: {data['u'].shape}  # (nx, ny)")
    print(f"k={args.k:g}, a1={args.a1:g}, a2={args.a2:g}")
    print(f"u range: [{data['u'].min():.6f}, {data['u'].max():.6f}]")
    print(f"q range: [{data['q'].min():.6f}, {data['q'].max():.6f}]")
    print(f"Total time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
