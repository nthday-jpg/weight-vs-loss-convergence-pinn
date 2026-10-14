# Experiment notebooks

Notebooks 1–5 launch exactly one configuration and one W&B run. Change only the variables in the experiment-selection cell, then rerun the notebook from the top. Notebook 6 is an independent, local diagnostic and does not use W&B.

## Notebooks

1. `01_weighting_comparison.ipynb`
   - Smooth flow only.
   - Compare `uniform` against `softadapt` with matched seeds.
   - Uses no interior observations.

2. `02_flow_difficulty.ipynb`
   - Compare weighting on `smooth`, `steep`, and `shock` cases.
   - Run every `(flow case, balancer, seed)` combination separately.

3. `03_data_ablation.ipynb`
   - Compare physics-only (`NUM_DATA_POINTS = 0`) against data-assisted PINNs.
   - Sparse interior observations are sampled only from the training time domain.
   - Keep the balancer fixed for this ablation; repeat later for the other balancer if needed.

4. `04_gradient_diagnostics.ipynb`
   - Compare raw and weighted component-gradient norms for `uniform` and `softadapt`.
   - Track pairwise gradient cosine similarities to identify aligned or conflicting objectives.
   - Diagnostics are measured on the current collocation batch every 100 optimizer steps.

5. `05_data_only.ipynb`
   - Train a supervised baseline using `data_loss` over the complete training grid.
   - Do not compute separate initial-condition, boundary-condition, or PDE-residual losses.
   - Keep validation- and test-time labels held out.

6. `06_tanh_linearization_hypothesis.ipynb`
   - Standalone controlled test using an MLP `x -> u` on `[-1, 1]` and `[10, 20]`.
   - Test a tanh network across initialization scales and seeds.
   - Train with the single residual `u_xx + u = 0` while diagnosing `u`, `u_x`, `u_xx`, tanh linearization, and condition-point learning.
   - Save local CSV diagnostics and plots under `results/tanh_linearization/`.

## Fair comparisons

For each pair of runs, keep architecture, optimizer, training steps, temporal split, seed, and generated dataset identical. Use at least three seeds for final results. The final test region must not be used to select hyperparameters.

A positive `NUM_DATA_POINTS` adds a supervised interior-data loss. Zero still uses the initial and boundary conditions required to define the PDE problem; “physics-only” here means no interior solution observations.
