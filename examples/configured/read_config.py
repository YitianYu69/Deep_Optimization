"""Read validated recipe settings without importing the training stack.

Run: python examples/configured/read_config.py
     python examples/configured/read_config.py --config configs/scheduling.toml
"""

import argparse
from pathlib import Path

from deepopt.logger import log_output
from deepopt.config import load_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "baseline.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    log_output(f"Experiment: {config.name}")
    log_output(f"Epochs: {config.training.epochs}")
    log_output(f"Learning rate: {config.training.learning_rate}")


if __name__ == "__main__":
    main()
