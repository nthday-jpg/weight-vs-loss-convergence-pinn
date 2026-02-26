from networkx import config
import torch
from dataclasses import asdict
from tqdm import tqdm
from accelerate import Accelerator
from model import BurgersPINN
from data.dataset import get_dataloader

class Trainer:
    def __init__(self, config, data_path):
        self.model = BurgersPINN(config.layers)
        self.dataloader = get_dataloader(data_path, batch_size=config.batch_size)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)  

    def train(self):
        accelerator = Accelerator(log_with="wandb")
        device = accelerator.device
        is_main_process = accelerator.is_main_process
                
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        
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
            if accelerator is not None:
                print(f"Using Accelerate with {accelerator.num_processes} process(es)")
        
        pbar = tqdm(range(self.config.num_epochs), desc='Training', disable=not is_main_process)
        for epoch in pbar:
            self.model.train()

            total_loss_epoch = torch.tensor(0.0, device=device)
            ics_loss_epoch = torch.tensor(0.0, device=device)
            bcs_loss_epoch = torch.tensor(0.0, device=device)
            res_loss_epoch = torch.tensor(0.0, device=device)
            total_samples = torch.tensor(0, device=device)
            
            for batch in self.dataloader:
                # Get batch size (may be smaller for last batch)
                batch_size = len(batch['interior']['t'])
                
                self.optimizer.zero_grad()
                loss_dict = self.model.compute_loss(batch)
                
                # Accumulate weighted sum
                total_loss_epoch += loss_dict['total_loss'].detach() * batch_size
                ics_loss_epoch += loss_dict['ics_loss'].detach() * batch_size
                bcs_loss_epoch += loss_dict['bcs_loss'].detach() * batch_size
                res_loss_epoch += loss_dict['res_loss'].detach() * batch_size
                total_samples += batch_size
                
                accelerator.backward(loss_dict['total_loss'])
                self.optimizer.step()

            if epoch % self.config.log_interval == 0:
                # Reduce across processes
                total_loss_epoch = accelerator.reduce(total_loss_epoch, reduction="sum").item()
                ics_loss_epoch = accelerator.reduce(ics_loss_epoch, reduction="sum").item()
                bcs_loss_epoch = accelerator.reduce(bcs_loss_epoch, reduction="sum").item()
                res_loss_epoch = accelerator.reduce(res_loss_epoch, reduction="sum").item()
                total_samples = accelerator.reduce(total_samples, reduction="sum").item()
                
                # Calculate average
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
                        'total_loss': total_loss_avg,
                        'ics_loss': ics_loss_avg,
                        'bcs_loss': bcs_loss_avg,
                        'res_loss': res_loss_avg
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


