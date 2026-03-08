from networkx import config
import torch
from dataclasses import asdict
from tqdm import tqdm
from accelerate import Accelerator
from model import BurgersPINN
from utils import load_burgers_data

class Trainer:
    def __init__(self, config, data_path):
        self.config = config
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.step_per_epoch = config.step_per_epoch

        self.model = BurgersPINN(config.layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
        
        self.dataset = load_burgers_data(data_path, is_torch=True)

        self.t_end = self.dataset['t'][-1].item()
        self.x_start = self.dataset['x'][0].item()
        self.x_end = self.dataset['x'][-1].item()

        # Initial conditions: u(0, x) for all x
        nx = len(self.dataset['x'])
        self.ics = {
            't': torch.zeros(nx),  # t=0 for all x points
            'x': self.dataset['x'].clone(),  # all x positions
            'u': self.dataset['usol'][0, :].clone()  # initial condition
        }
        
        # Boundary conditions: u(t, x=0) and u(t, x=L) for all t
        nt = len(self.dataset['t'])
        self.bcs = {
            't': self.dataset['t'].repeat(2),  # repeat for both boundaries
            'x': torch.cat([torch.full((nt,), self.dataset['x'][0]), 
                           torch.full((nt,), self.dataset['x'][-1])]),  # x=0 and x=L
            'u': torch.cat([self.dataset['usol'][:, 0], 
                           self.dataset['usol'][:, -1]])  # boundary values
        }

    def sample_batches(self, device):
        for _ in range(self.step_per_epoch):
            t = torch.rand(self.batch_size, device=device) * self.t_end  # Random time in [0, t_end]
            x = torch.rand(self.batch_size, device=device) * (self.x_end - self.x_start) + self.x_start
            yield {'t': t, 'x': x}

    def train(self):
        accelerator = Accelerator(log_with="wandb")
        device = accelerator.device
        is_main_process = accelerator.is_main_process
                
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)

        # Move ICs/BCs to device once
        self.ics = {k: v.to(device).requires_grad_(False) for k, v in self.ics.items()}
        self.bcs = {k: v.to(device).requires_grad_(False) for k, v in self.bcs.items()}

        history = {
            'epoch': [],
            'total_loss': [],
            'ics_loss': [],
            'bcs_loss': [],
            'res_loss': [],
        }

        accelerator.init_trackers(
            project_name=self.config.wandb_project,
            config=asdict(self.config),
            init_kwargs={"name": self.config.wandb_run_name, "tags": self.config.wandb_tags}
        )
        
        if is_main_process:
            print(f"\nStarting training for {self.config.num_epochs} epochs...")
            print(f"Device: {device}")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"Using Accelerate with {accelerator.num_processes} process(es)")
        
        pbar = tqdm(range(self.config.num_epochs), desc='Training', disable=not is_main_process, ncols=120)
        for epoch in pbar:
            self.model.train()

            total_loss_epoch = torch.tensor(0.0, device=device)
            ics_loss_epoch = torch.tensor(0.0, device=device)
            bcs_loss_epoch = torch.tensor(0.0, device=device)
            res_loss_epoch = torch.tensor(0.0, device=device)
            total_samples = torch.tensor(0, device=device)
            
            for batch in self.sample_batches(device):
                self.optimizer.zero_grad()
                
                loss_dict = accelerator.unwrap_model(self.model).compute_loss(batch, self.ics, self.bcs)
                
                total_loss_epoch += loss_dict['total_loss'].detach() * self.batch_size
                ics_loss_epoch += loss_dict['ics_loss'].detach() * self.batch_size
                bcs_loss_epoch += loss_dict['bcs_loss'].detach() * self.batch_size
                res_loss_epoch += loss_dict['res_loss'].detach() * self.batch_size
                total_samples += self.batch_size
                
                accelerator.backward(loss_dict['total_loss'])
                self.optimizer.step()

            if epoch % self.config.log_interval == 0:
                total_loss_epoch = accelerator.reduce(total_loss_epoch, reduction="sum").item()
                ics_loss_epoch = accelerator.reduce(ics_loss_epoch, reduction="sum").item()
                bcs_loss_epoch = accelerator.reduce(bcs_loss_epoch, reduction="sum").item()
                res_loss_epoch = accelerator.reduce(res_loss_epoch, reduction="sum").item()
                total_samples = accelerator.reduce(total_samples, reduction="sum").item()
                
                total_loss_avg = total_loss_epoch / total_samples
                ics_loss_avg = ics_loss_epoch / total_samples
                bcs_loss_avg = bcs_loss_epoch / total_samples
                res_loss_avg = res_loss_epoch / total_samples
                
                if is_main_process:
                    history['epoch'].append(epoch)
                    history['total_loss'].append(total_loss_avg)
                    history['ics_loss'].append(ics_loss_avg)
                    history['bcs_loss'].append(bcs_loss_avg)
                    history['res_loss'].append(res_loss_avg)
                    
                    pbar.set_postfix({
                        'total': f'{total_loss_avg:.3e}',
                        'ics': f'{ics_loss_avg:.3e}',
                        'bcs': f'{bcs_loss_avg:.3e}',
                        'res': f'{res_loss_avg:.3e}'
                    })
                    accelerator.log({
                        'total_loss': total_loss_avg,
                        'ics_loss': ics_loss_avg,
                        'bcs_loss': bcs_loss_avg,
                        'res_loss': res_loss_avg,
                        'epoch': epoch
                    })
        
        accelerator.end_training()
        return history


