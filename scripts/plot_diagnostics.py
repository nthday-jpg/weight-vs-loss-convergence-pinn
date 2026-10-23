"""Plot a gradient-annealing diagnostic CSV (weights, gradient stats, losses).

Usage:
    python scripts/plot_diagnostics.py results/diag_ga_helmholtz.csv \
        --out results/diag_ga_helmholtz.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def column(rows, key):
    return [float(row[key]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    rows = read(args.csv_path)
    steps = column(rows, "step")
    weights = [k for k in rows[0] if k.endswith("_weight_new")]
    terms = [k.removesuffix("_weight_new") for k in weights]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0, 0]
    for term in terms:
        ax.plot(steps, column(rows, f"{term}_weight_new"), label=f"{term}")
    ax.set_yscale("log")
    ax.set_title("Adaptive weights")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[0, 1]
    for term in terms:
        ax.plot(steps, column(rows, f"{term}_mean_abs_grad"), label=f"mean |grad L_{term}|")
    ax.set_yscale("log")
    ax.set_title("Mean absolute component gradients (denominator)")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, column(rows, "res_max_abs_grad"), label="max |grad L_res|")
    ax.set_yscale("log")
    ax.set_title("Residual gradient maximum (numerator)")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[1, 1]
    for term in terms:
        ax.plot(steps, column(rows, f"{term}_loss"), label=f"L_{term}")
    ax.set_yscale("log")
    ax.set_title("Loss terms")
    ax.set_xlabel("optimizer step")
    ax.legend()

    if args.title:
        fig.suptitle(args.title)
    out = Path(args.out or Path(args.csv_path).with_suffix(".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
