"""Summarize a gradient-annealing diagnostic CSV.

Prints the step where each term's weight grew largest, the gradient
statistics that produced it, and the parameter magnitudes, so the mechanism of
the weight blow-up can be inspected directly.

Usage:
    python scripts/summarize_diagnostics.py results/diag_ga_uncapped.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="+")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument(
        "--from-step",
        type=int,
        default=None,
        help="Also report the mean weight over rows at or after this step.",
    )
    args = parser.parse_args()

    for path_text in args.csv_path:
        path = Path(path_text)
        rows = read_rows(path)
        if not rows:
            print(f"{path}: empty")
            continue
        print(f"\n=== {path} ({len(rows)} adaptation records) ===")

        weights = [key for key in rows[0] if key.endswith("_weight")]
        for key in weights:
            name = key.removesuffix("_weight")
            new_key = f"{name}_weight_new"
            values = [
                as_float(row, new_key if new_key in row else key) for row in rows
            ]
            final = values[-1]
            peak = max(values)
            peak_step = rows[values.index(peak)]["step"]
            mean_all = sum(values) / len(values)
            tail = values[len(values) // 2 :]
            mean_tail = sum(tail) / len(tail)
            print(
                f"\nweight[{name}]: mean(all)={mean_all:.4e} "
                f"mean(2nd half)={mean_tail:.4e} final={final:.6e} "
                f"peak={peak:.6e} @step {peak_step}"
            )
            if args.from_step is not None:
                plateau = [
                    as_float(row, new_key if new_key in row else key)
                    for row in rows
                    if int(row["step"]) >= args.from_step
                ]
                if plateau:
                    print(
                        f"weight[{name}]: mean(step>={args.from_step})="
                        f"{sum(plateau) / len(plateau):.4e}"
                    )
            header = (
                f"{'step':>6} {'res_max|g|':>12} {'res_mean|g|':>12} "
                f"{name+'_mean|g|':>12} {'target_ratio':>14} {'weight':>14} "
                f"{'|param|max':>12}"
            )
            print(header)
            show = rows[:3] + rows[-args.rows:]
            for row in show:
                weight_value = as_float(
                    row, f"{name}_weight_new" if f"{name}_weight_new" in row else key
                )
                print(
                    f"{row['step']:>6} "
                    f"{as_float(row, 'res_max_abs_grad'):>12.4e} "
                    f"{as_float(row, 'res_mean_abs_grad'):>12.4e} "
                    f"{as_float(row, f'{name}_mean_abs_grad'):>12.4e} "
                    f"{as_float(row, f'{name}_target_ratio'):>14.4e} "
                    f"{weight_value:>14.4e} "
                    f"{as_float(row, 'param_max_abs'):>12.4e}"
                )


if __name__ == "__main__":
    main()
