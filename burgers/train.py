import torch
from pathlib import Path
from tqdm import tqdm
import wandb    
from accelerate import Accelerator
from burgers.data.dataset import get_dataloader

from burgers.model import BurgersPINN

def train(
    model,
    dataloader,
    config
):
    accelerator = Accelerator(
        mixed_precision='fp16' if config.get('mixed_precision', False) else 'no',
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
    )
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    history = {
        'epoch': [],
        'total_loss': [],
        'ics_loss': [],
        'bcs_loss': [],
        'res_loss': [],
    }
    
    if is_main_process:
        wandb.init(
            project=config.get('wandb_project', 'burgers-pinn'),
            name=config.get('wandb_run_name', None),
            config=config,
            tags=config.get('wandb_tags', ['pinn', 'burgers'])
        )

        print(f"\nStarting training for {config['num_epochs']} epochs...")
        print(f"Device: {device}")
        print(f"Learning rate: {config['learning_rate']}")
        if accelerator is not None:
            print(f"Using Accelerate with {accelerator.num_processes} process(es)")
    
    pbar = tqdm(range(config['num_epochs']), desc='Training', disable=not is_main_process)
    for epoch in pbar:
        model.train()

        total_loss_epoch = torch.tensor(0.0, device=device)
        ics_loss_epoch = torch.tensor(0.0, device=device)
        bcs_loss_epoch = torch.tensor(0.0, device=device)
        res_loss_epoch = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0, device=device)
        
        for batch in dataloader:
            # Get batch size (may be smaller for last batch)
            batch_size = len(batch['interior']['t'])
            
            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch)
            
            # Accumulate weighted sum
            total_loss_epoch += loss_dict['total_loss'].detach() * batch_size
            ics_loss_epoch += loss_dict['ics_loss'].detach() * batch_size
            bcs_loss_epoch += loss_dict['bcs_loss'].detach() * batch_size
            res_loss_epoch += loss_dict['res_loss'].detach() * batch_size
            total_samples += batch_size
            
            accelerator.backward(loss_dict['total_loss'])
            optimizer.step()

        if epoch % config['log_interval'] == 0:
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
                wandb.log({
                    'train/total_loss': total_loss_avg,
                    'train/ics_loss': ics_loss_avg,
                    'train/bcs_loss': bcs_loss_avg,
                    'train/res_loss': res_loss_avg,
                    'epoch': epoch
                })
    return history


def main():
    """Main training function."""
    
    # Configuration
    config = {
        # Model architecture
        'layers': [2, 32, 1],
        
        # Training hyperparameters
        'num_epochs': 10,
        'learning_rate': 1e-3,
        
        # Logging and checkpointing
        'log_interval': 1,
        'save_interval': 5000,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Wandb configuration
        'use_wandb': True,
        'wandb_project': 'burgers-pinn',
        'wandb_run_name': None,  # Auto-generate if None
        'wandb_tags': ['pinn', 'burgers', 'spectral-solver'],
        
        # Accelerate configuration
        'use_accelerate': True,
        'mixed_precision': False,  # Use 'fp16' or 'bf16' for mixed precision
        'gradient_accumulation_steps': 1
    }

    model = BurgersPINN(config['layers'])
    dataloader = get_dataloader(batch_size=64)

    print(f"\nModel architecture:")
    print(model.network)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    history = train(
        model,
        config=config,
        dataloader=dataloader
    )

if __name__ == "__main__":
    main()
