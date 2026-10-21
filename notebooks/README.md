# Experiment notebooks

The notebook directory has four distinct entry points. The Navier–Stokes studies share one configurable training notebook; the other notebooks are standalone local diagnostics.

## Notebooks

### `navier_stokes_experiments.ipynb`

Runs one physics-only Navier–Stokes configuration and one W&B run at a time on the shock-forming case. The PINN uses initial-condition, periodic boundary-condition, and PDE-residual losses only; no interior solution observations are used for training. Use the **Run controls** cell to reproduce the related studies:

- **Loss weighting:** compare `SOFTADAPT_BETA = 0.0` (uniform weights) with a positive beta using matched seeds.
- **Gradient diagnostics:** raw and weighted component-gradient norms, pairwise cosine similarities, and per-loss parameter-gradient histograms are always tracked.

Validation and test labels remain held out.

### `softadapt_simple_pde.ipynb`

A standalone, pre-run SoftAdapt diagnostic with no W&B usage:

- Solves the 1-D Poisson problem $u''=-\pi^2\sin(\pi x)$ with homogeneous Dirichlet conditions.
- Tracks interval losses, raw and weighted component-gradient norms, normalized SoftAdapt scores, and selected weights.
- Writes local CSV and plot artifacts to `results/softadapt_simple_pde/` when run.

### `softadapt_compressible_navier_stokes_1d.ipynb`

A standalone, pre-run SoftAdapt diagnostic with no W&B usage:

- Runs locally or as an imported Kaggle notebook; on Kaggle it clones the repository and generates the shock reference dataset when needed.
- Trains the repository's periodic 1-D compressible Navier–Stokes PINN on initial-condition, boundary-condition, and conservation-residual losses.
- Tracks interval losses, raw and weighted gradient norms, normalized SoftAdapt scores, selected weights, and held-out relative errors.
- Writes local CSV and plot artifacts to `results/softadapt_compressible_ns/` when run. Kaggle requires Internet access for the initial clone but needs no secrets or W&B account.

### `tanh_linearization_hypothesis.ipynb`

A standalone local diagnostic that does not use W&B:

- Uses an MLP `x -> u` on `[-1, 1]` and `[10, 20]`.
- Tests a tanh network across initialization scales and seeds.
- Trains with the single residual `u_xx + u = 0` while diagnosing `u`, `u_x`, `u_xx`, tanh linearization, condition-point learning, and per-term parameter-gradient norms.
- Saves CSV diagnostics and plots under `results/tanh_linearization/`.

## Fair comparisons

Keep architecture, optimizer, training steps, temporal split, seed, and generated dataset identical within each comparison. Change one run control at a time, use at least three seeds for final results, and do not use the final test region to select hyperparameters.

“Physics-only” means no interior solution observations; the initial and periodic boundary conditions are still required to define the PDE problem.
