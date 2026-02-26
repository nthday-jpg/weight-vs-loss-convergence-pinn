from torch.utils.data import Dataset, DataLoader
from burgers.utils import load_burgers_data
import torch


class BurgersDataset(Dataset):
    def __init__(self, data_path='burgers/data/burgers.pt'):
        # Load: t (Nt,), x (Nx,), usol (Nt, Nx), nu (scalar)
        t, x, usol, nu = load_burgers_data(data_path, is_torch=True)
        self.t: torch.Tensor = t  # type: ignore
        self.x: torch.Tensor = x  # type: ignore
        self.usol: torch.Tensor = usol  # type: ignore
        self.nu = nu
    
        # Initial condition: t=0, all x
        self.ics = {
            "t": self.t[0].repeat(len(self.x)),   # (Nx,)
            "x": self.x,                          # (Nx,)
            "u": self.usol[0]                     # (Nx,)
        }

        # Boundary conditions: all t, x=x_min and x=x_max
        t_bc = self.t.repeat_interleave(2)       # (2*Nt,)
        x_bc = torch.stack([self.x[0], self.x[-1]]).repeat(len(self.t))  # (2*Nt,)
        u_bc = torch.stack([self.usol[:, 0], self.usol[:, -1]], dim=1).flatten()  # (2*Nt,)

        self.bcs = {
            "t": t_bc,
            "x": x_bc,
            "u": u_bc
        }

        # Interior points (exclude t=0 and spatial boundaries)
        self.t_interior = self.t[1:]
        self.x_interior = self.x[1:-1]

        self.Nt = len(self.t_interior)
        self.Nx = len(self.x_interior)

    def __len__(self):
        return self.Nt * self.Nx

    def __getitem__(self, idx):
        t_idx = idx // self.Nx
        x_idx = idx % self.Nx

        return {
            "interior": {
                "t": self.t_interior[t_idx],
                "x": self.x_interior[x_idx],
            },
            "ics": self.ics,
            "bcs": self.bcs
        }


def get_dataloader(data_path='burgers/data/burgers.pt',
                   batch_size=64, num_workers=4, pin_memory=True,
                   shuffle=True):
        
        dataset = BurgersDataset(data_path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return dataloader