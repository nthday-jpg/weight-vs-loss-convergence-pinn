from dataclasses import dataclass, field


@dataclass
class Config:
    """Parameters that define one reproducible training experiment."""

    problem: str = "burgers"
    layers: list[int] = field(default_factory=lambda: [2, 64, 64, 64, 1])
    num_epochs: int = 500
    step_per_epoch: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 2048
    seed: int = 42

    # Train physics only to this fraction of the available time domain. The
    # following interval is extrapolation validation; the final interval is test.
    train_time_fraction: float = 0.7
    validation_time_fraction: float = 0.85

    # Sparse interior observations inside the training time domain. Zero is a
    # physics-only PINN; a positive value is a data-assisted PINN.
    num_data_points: int = 0

    # Keep uniform as the fixed baseline and SoftAdapt as the adaptive method.
    balancer_type: str = "uniform"
    softadapt_params: dict = field(
        default_factory=lambda: {
            "beta": 0.1,
            "history_length": 5,
            "normalized": False,
            "loss_weighted": False,
        }
    )

    wandb_project: str = "loss-weight-convergence"
    wandb_run_name: str | None = None
