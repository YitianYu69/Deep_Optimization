"""Read validated recipe settings without importing the training stack.

Run: python examples/configured/read_config.py
     python examples/configured/read_config.py --config configs/scheduling.toml
     python examples/configured/read_config.py --config builtin:fp8
"""

import argparse

from deepopt.logger import log_output
from deepopt.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="builtin:baseline",
                        help="TOML path or installed preset builtin:NAME")
    args = parser.parse_args()
    config = load_config(args.config)
    log_output(f"Experiment: {config.name}")
    log_output(f"Epochs: {config.training.epochs}")
    log_output(f"Learning rate: {config.training.learning_rate}")


if __name__ == "__main__":
    main()
