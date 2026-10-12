
from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def load_burgers_data(data_path='burgers/data/burgers.npz', is_torch=False):
    """Load data from .npz or .pt file.
        t: shape (steps+1,)
        x: shape (nx,)
        usol: shape (steps+1, nx)
        nu: shape (1,)
    """ 
    if data_path.endswith('.pt'):
        data = torch.load(data_path)
    elif data_path.endswith('.npz'):
        data = np.load(data_path)
    else:
        raise ValueError("Unsupported file format. Use .npz or .pt")

    t = data['t']
    x = data['x']
    usol = data['usol']
    nu = data['nu']

    if is_torch and not isinstance(t, torch.Tensor):
        t = torch.from_numpy(t)
        x = torch.from_numpy(x)
        usol = torch.from_numpy(usol)
        nu = torch.tensor(nu)
    elif not is_torch and isinstance(t, torch.Tensor):
        t = t.numpy()
        x = x.numpy()
        usol = usol.numpy()
        nu = nu.item() if isinstance(nu, torch.Tensor) else nu

    return {
        't': t,
        'x': x,
        'usol': usol,
        'nu': nu
    }

def plot_comparison(
    t,
    x,
    u_true,
    u_pred,
    save_path='results/comparison.png',
    train_t_end=None,
    validation_t_end=None,
):
    """Plot comparison between true and predicted solutions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    value_min = min(np.min(u_true), np.min(u_pred))
    value_max = max(np.max(u_true), np.max(u_pred))
    
    # True solution
    im1 = axes[0].pcolormesh(
        t, x, u_true.T, shading='auto', cmap='jet', vmin=value_min, vmax=value_max
    )
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Space (x)')
    axes[0].set_title('True Solution')
    plt.colorbar(im1, ax=axes[0])
    
    # PINN prediction
    im2 = axes[1].pcolormesh(
        t, x, u_pred.T, shading='auto', cmap='jet', vmin=value_min, vmax=value_max
    )
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Space (x)')
    axes[1].set_title('PINN Prediction')
    plt.colorbar(im2, ax=axes[1])
    
    # Absolute error
    error = np.abs(u_true - u_pred)
    im3 = axes[2].pcolormesh(t, x, error.T, shading='auto', cmap='hot')
    axes[2].set_xlabel('Time (t)')
    axes[2].set_ylabel('Space (x)')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[2])

    for axis in axes:
        if train_t_end is not None:
            axis.axvline(train_t_end, color='white', linestyle='--', linewidth=1.5)
        if validation_t_end is not None:
            axis.axvline(validation_t_end, color='cyan', linestyle=':', linewidth=1.5)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training loss history."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    steps = history['optimizer_step']
    
    # Total loss
    axes[0].semilogy(steps, history['total_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Optimizer step')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].semilogy(steps, history['ics_loss'], 'r-', label='ICs Loss', linewidth=2)
    axes[1].semilogy(steps, history['res_loss'], 'g-', label='Residual Loss', linewidth=2)
    axes[1].set_xlabel('Optimizer step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # L2 error
    if 'l2_error' in history:
        axes[2].semilogy(steps, history['l2_error'], 'm-', linewidth=2)
        axes[2].set_xlabel('Optimizer step')
        axes[2].set_ylabel('Relative L2 Error')
        axes[2].set_title('Test Error')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()