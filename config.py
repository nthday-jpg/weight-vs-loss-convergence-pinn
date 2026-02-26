from dataclasses import dataclass

@dataclass
class Config:
    
    # Architecture
    layers: list[int] = [2, 32, 1]
    activation: str = 'tanh'
    final_activation: str | None = None

    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 64
    l2_reg: float = 0.0
    
    # Logging and checkpointing
    log_interval: int = 1
    save_interval: int = 5000
        
    # Wandb configuration
    wandb_project: str = 'burgers-pinn'
    wandb_run_name: str | None = None  # Auto-generate if None
    wandb_tags: list | None = None
    
