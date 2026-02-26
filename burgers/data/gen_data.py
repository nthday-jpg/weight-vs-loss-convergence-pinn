#Code adapted from https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/examples/burgers/data/gen_burgers.m

import time
import torch
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path


def solve_burgers_equation(nn=511, steps=200, nu=0.01/np.pi, t_final=1.0):
    """
    Solve Burgers equation using pseudo-spectral method with scipy ODE solver.
    
    Args:
        nn: number of spatial points
        steps: number of time steps for output
        nu: viscosity coefficient
        t_final: final time
    
    Returns:
        t: time points (steps+1,)
        x: spatial points (nn,)
        usol: solution array of shape (steps+1, nn)
        nu: viscosity value
    """
    # Spatial domain setup
    x = np.linspace(-1, 1, nn, dtype=np.float32)  # shape: (nn,)
    dx = 2.0 / (nn - 1)
    
    # Wavenumbers for FFT
    k = np.fft.fftfreq(nn, d=dx/(2*np.pi))
    
    # Initial condition: -sin(pi*x)
    u0 = -np.sin(np.pi * x)
    
    # Define RHS function for ODE solver
    def burgers_rhs(t, u):
        """Right-hand side: du/dt = -u*du/dx + nu*d²u/dx²"""
        u_hat = np.fft.fft(u)
        
        # Spectral derivatives
        du_dx = np.fft.ifft(1j * k * u_hat).real
        d2u_dx2 = np.fft.ifft(-k**2 * u_hat).real
        
        return -u * du_dx + nu * d2u_dx2
    
    # Time points for output
    t_eval = np.linspace(0, t_final, steps+1, dtype=np.float32)
    
    print(f"Solving Burgers equation with {nn} spatial points and {steps} time steps...")
    print(f"Viscosity nu = {nu:.6f}")
    
    # Solve using scipy's RK45 method
    sol = solve_ivp(
        burgers_rhs, 
        (0, t_final), 
        u0, 
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    
    print(f"Integration successful: {sol.success}")
    print(f"Number of function evaluations: {sol.nfev}")
    
    # Solution is in sol.y with shape (nn, steps+1)
    usol = sol.y.T  # shape: (steps+1, nn)
    
    return sol.t, x, usol, nu


def save_data(t, x, usol, nu, save_dir='burgers/data'):
    """
    Save solution data in PyTorch and NumPy formats.
    
    Args:
        t: time points
        x: spatial points  
        usol: solution array
        nu: viscosity
        save_dir: directory to save data
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save as PyTorch tensor
    torch.save({
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'usol': torch.from_numpy(usol),
        'nu': nu
    }, save_path / 'burgers.pt')
    print(f"Data saved to {save_path / 'burgers.pt'}")
    
    # Save as NumPy
    np.savez(
        save_path / 'burgers.npz',
        t=t, x=x, usol=usol, nu=nu
    )
    print(f"Data saved to {save_path / 'burgers.npz'}")


def main():
    """Main function to generate Burgers equation data."""
    start_time = time.time()
    
    # Parameters from MATLAB code
    nn = 511
    steps = 200
    nu = 0.01 / np.pi
    t_final = 1.0
    
    # Solve equation
    t, x, usol, nu = solve_burgers_equation(nn, steps, nu, t_final)
    
    # Save data
    save_data(t, x, usol, nu)
    
    elapsed_time = time.time() - start_time
    
    print("\nDone! Data shape:", usol.shape)
    print(f"Time range: [{t[0]:.3f}, {t[-1]:.3f}]")
    print(f"Space range: [{x[0]:.3f}, {x[-1]:.3f}]")
    print(f"Solution range: [{usol.min():.6f}, {usol.max():.6f}]")
    print(f"\nTotal time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
