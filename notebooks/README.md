# Experiment notebooks

The notebook directory has two distinct entry points. The Navier–Stokes studies share one configurable training notebook; the tanh-linearization study is a separate local diagnostic.

## Notebooks

### `navier_stokes_experiments.ipynb`

Runs one Navier–Stokes configuration and one W&B run at a time. Use the **Run controls** cell to reproduce all of the related studies:

- **Loss weighting:** compare `SOFTADAPT_BETA = 0.0` (uniform weights) with a positive beta using matched seeds.
- **Flow difficulty:** compare `FLOW_CASE = 'smooth'`, `'steep'`, and `'shock'`.
- **Data ablation:** compare `NUM_DATA_POINTS = 0` with a fixed positive count.
- **Gradient diagnostics:** set `TRACK_GRADIENT_METRICS = True` to track raw and weighted component-gradient norms and pairwise cosine similarities.
- **Data-only baseline:** set `RUN_KIND = 'data_only'` to train on every interior point in the training grid without separate initial-condition, boundary-condition, or PDE-residual losses.

All supervised observations exclude initial-time and spatial-boundary points. Validation and test labels remain held out.

### `tanh_linearization_hypothesis.ipynb`

A standalone local diagnostic that does not use W&B:

- Uses an MLP `x -> u` on `[-1, 1]` and `[10, 20]`.
- Tests a tanh network across initialization scales and seeds.
- Trains with the single residual `u_xx + u = 0` while diagnosing `u`, `u_x`, `u_xx`, tanh linearization, condition-point learning, and per-term parameter-gradient norms.
- Saves CSV diagnostics and plots under `results/tanh_linearization/`.

## Fair comparisons

Keep architecture, optimizer, training steps, temporal split, seed, and generated dataset identical within each comparison. Change one run control at a time, use at least three seeds for final results, and do not use the final test region to select hyperparameters.

A positive `NUM_DATA_POINTS` adds supervised interior-data loss. Zero still uses the initial and boundary conditions required to define the PDE problem; “physics-only” means no interior solution observations.
