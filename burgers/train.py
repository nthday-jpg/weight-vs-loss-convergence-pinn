import argparse
import json
from logging import config
from trainer import Trainer
from config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PINN for the Burgers' equation")
    parser.add_argument('--config_file', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config_file, 'r') as f:
        config_dict = json.load(f)

    print(f"Debug: Loaded config_dict = {config_dict}")
    config = Config(**config_dict)
    print(f"Debug: Config.wandb_run_name = {config.wandb_run_name}")
    trainer = Trainer(config, args.data_file)
    trainer.train()
    
if __name__ == "__main__":
    main()