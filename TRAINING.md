# Burgers Equation PINN with WandB and Accelerate

## Installation

Install the required dependencies:

```powershell
# Core dependencies
uv add torch numpy matplotlib tqdm scipy

# Optional: For experiment tracking and distributed training
uv add wandb accelerate
```

Or use the install script:
```powershell
python install_extras.py
```

## Training

### Basic Training

```powershell
python burgers/train.py
```

### With Distributed Training (Accelerate)

First configure accelerate:
```powershell
accelerate config
```

Then run with accelerate:
```powershell
accelerate launch burgers/train.py
```

## Features

### 🔥 Weights & Biases Integration

The training script automatically logs to W&B when available:
- Training losses (total, ICs, residual)
- L2 relative error
- Learning rate
- Prediction visualizations
- Model checkpoints as artifacts

Configure W&B in the config dict:
```python
config = {
    'use_wandb': True,
    'wandb_project': 'burgers-pinn',
    'wandb_run_name': 'my-experiment',
    'wandb_tags': ['pinn', 'burgers', 'custom-tag']
}
```

### ⚡ Accelerate Support

Enables distributed training and mixed precision:
- Multi-GPU training
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Automatic device placement

Configure in config dict:
```python
config = {
    'use_accelerate': True,
    'mixed_precision': 'fp16',  # or 'bf16' or False
    'gradient_accumulation_steps': 1
}
```

## Experiment Tracking

The script logs:
- **Metrics**: Loss components, L2 error, learning rate
- **Visualizations**: Solution comparisons, training history
- **Artifacts**: Model checkpoints, final model
- **System info**: Device, number of parameters, training time

## Configuration

Key hyperparameters in `burgers/train.py`:

```python
config = {
    'layers': [2, 64, 64, 64, 64, 1],  # Network architecture
    'num_epochs': 20000,                # Training epochs
    'learning_rate': 1e-3,              # Learning rate
    'num_res': 10000,                   # Residual collocation points
    'use_causal': False,                # Causal weighting for time-marching
}
```

## Results

Results are saved to `results/`:
- `comparison.png` - True vs predicted solution
- `training_history.png` - Loss curves
- `final_model.pt` - Trained model
- `checkpoints/` - Periodic checkpoints
