"""Launch one PINN experiment from environment variables.

Required environment variables:
    PINN_CONFIG: JSON object accepted by share.config.Config
    PINN_DATA_FILE: path to the generated .pt or .npz dataset
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from share.config import Config


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


def _load_trainer_class(problem: str):
    root = Path(__file__).resolve().parent
    normalized = problem.replace("-", "_").lower()

    if normalized == "burgers":
        trainer_file = root / "burgers" / "trainer.py"
        module_name = "burgers_trainer_module"
    elif normalized == "navier_stokes":
        trainer_file = root / "navier-stokes" / "trainer.py"
        module_name = "navier_stokes_trainer_module"
    else:
        raise ValueError(
            f"Unsupported problem '{problem}'. Use 'burgers' or 'navier_stokes'."
        )

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
    try:
        config_values = json.loads(_required_env("PINN_CONFIG"))
    except json.JSONDecodeError as error:
        raise ValueError("PINN_CONFIG must contain a valid JSON object.") from error

    config = Config(**config_values)
    trainer_class = _load_trainer_class(config.problem)
    trainer_class(config, _required_env("PINN_DATA_FILE")).train()


if __name__ == "__main__":
    main()
