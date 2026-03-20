from dataclasses import dataclass, field

@dataclass
class Config:
    # Problem selection
    problem: str = 'burgers'

    
    # Architecture
    layers: list[int] = field(default_factory=lambda: [2, 32, 1])
    activation: str = 'tanh'
    final_activation: str | None = None

    # Training parameters
    num_epochs: int = 10
    step_per_epoch: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 64
    l2_reg: float = 0.0    
    max_grad_norm: float | None = None
    
    # Logging and checkpointing
    log_interval: int = 1
    save_interval: int = 5000

    # Balancer
    balancer_type: str = 'uniform'
    
    # Balancer-specific parameters (optional dicts)
    proportional_params: dict = field(default_factory=lambda: {'alpha': 0.1})
    inverse_proportional_params: dict = field(default_factory=lambda: {'alpha': 0.1})
    softadapt_params: dict = field(default_factory=lambda: {'beta': 0.1, 'ema_alpha': 0.05})
        
    # Wandb configuration
    wandb_project: str = 'burgers-pinn'
    wandb_run_name: str | None = None
    wandb_tags: list | None = field(default_factory=lambda: None)

