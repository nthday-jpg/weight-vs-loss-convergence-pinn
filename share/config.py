from dataclasses import dataclass, field


@dataclass
class Config:
    """Parameters that define one reproducible training experiment."""

    problem: str = "burgers"
    layers: list[int] = field(default_factory=lambda: [2, 64, 64, 64, 1])
    num_steps: int = 50_000
    adapt_every_steps: int = 100
    log_every_steps: int = 50
    track_gradient_metrics: bool = False
    learning_rate: float = 1e-3
    batch_size: int = 2048
    seed: int = 42

    # Train physics only to this fraction of the available time domain. The
    # following interval is extrapolation validation; the final interval is test.
    train_time_fraction: float = 0.7
    validation_time_fraction: float = 0.85

    # Supervised observations strictly inside the training space-time domain.
    # Zero disables data, a positive integer samples interior points, and "all"
    # uses every interior point. Initial and boundary points are always excluded.
    num_data_points: int | str = 0

    # Select optimizer loss terms from: ics, bcs, res, and data. None uses all
    # available terms; selecting data requires num_data_points > 0.
    loss_terms: list[str] | None = None

    # beta=0 gives fixed uniform weights; beta>0 enables adaptation.
    # weight_ema in [0, 1) smooths weight updates toward the previous weights.
    softadapt_params: dict = field(
        default_factory=lambda: {
            "beta": 0.1,
            "history_length": 5,
            "normalized": False,
            "loss_weighted": False,
            "weight_ema": 0.0,
        }
    )

    # Loss-weighting policy: "softadapt" or "gradient_annealing".
    balancer: str = "softadapt"

    # alpha is the EMA rate for the ratio max|grad L_ref| / mean|grad L_i|.
    # The reference term (PDE residual) keeps a fixed weight of 1.
    gradient_annealing_params: dict = field(
        default_factory=lambda: {
            "alpha": 0.1,
            "reference": "res",
            "initial_weight": 1.0,
        }
    )

    wandb_project: str = "loss-weight-convergence"
    wandb_run_name: str | None = None
