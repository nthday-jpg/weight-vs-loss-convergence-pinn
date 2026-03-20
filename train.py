import argparse
import importlib.util
import json
import sys
from pathlib import Path

from share.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PINN for selected PDE problem")
    parser.add_argument('--config_file', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument(
        '--problem',
        type=str,
        default=None,
        choices=['burgers', 'navier_stokes', 'navier-stokes'],
        help='Override problem type from config',
    )
    return parser.parse_args()


def _load_trainer_class(problem: str):
    root = Path(__file__).resolve().parent
    normalized = problem.replace('-', '_').lower()

    if normalized == 'burgers':
        trainer_file = root / 'burgers' / 'trainer.py'
        module_name = 'burgers_trainer_module'
    elif normalized == 'navier_stokes':
        trainer_file = root / 'navier-stokes' / 'trainer.py'
        module_name = 'navier_stokes_trainer_module'
    else:
        raise ValueError(f"Unsupported problem '{problem}'. Use 'burgers' or 'navier_stokes'.")

    if not trainer_file.exists():
        raise FileNotFoundError(f"Trainer file not found: {trainer_file}")

    trainer_dir = str(trainer_file.parent)
    if trainer_dir not in sys.path:
        sys.path.insert(0, trainer_dir)

    spec = importlib.util.spec_from_file_location(module_name, trainer_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load trainer module from {trainer_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trainer


def main():
    args = parse_args()
    with open(args.config_file, 'r') as f:
        config_dict = json.load(f)

    config = Config(**config_dict)
    selected_problem = args.problem if args.problem is not None else config.problem

    Trainer = _load_trainer_class(selected_problem)
    trainer = Trainer(config, args.data_file)
    trainer.train()
    
if __name__ == "__main__":
    main()