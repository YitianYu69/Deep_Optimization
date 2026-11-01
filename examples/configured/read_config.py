from pathlib import Path

from deepopt.logger import log_output
from deepopt.config import load_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "baseline.toml")
    log_output(f"Experiment: {config.name}")
    log_output(f"Epochs: {config.training.epochs}")
    log_output(f"Learning rate: {config.training.learning_rate}")


if __name__ == "__main__":
    main()