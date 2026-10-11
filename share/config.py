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

    # Keep uniform as the fixed baseline and SoftAdapt as the adaptive method.
    balancer_type: str = "uniform"
    softadapt_params: dict = field(
        default_factory=lambda: {"beta": 0.1, "ema_alpha": 0.05}
    )

    wandb_project: str = "loss-weight-convergence"
    wandb_run_name: str | None = None
