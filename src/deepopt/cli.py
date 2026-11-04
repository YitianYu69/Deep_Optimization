"""Keep argument parsing separate from the training implementation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys

from .config import load_config
from .logger import logging_session, log_output


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        distributed = sys.modules.get("deepopt.distributed._process")
        if distributed is not None:
            distributed.shutdown_distributed()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepopt", description="Train and evaluate reproducible baselines.")
    parser.add_argument("--config", help="Inspect a TOML path or builtin:NAME (legacy shorthand)")
    commands = parser.add_subparsers(dest="command")
    inspect = commands.add_parser("inspect-config", help="Validate TOML and print resolved settings")
    inspect.add_argument("--config", required=True, dest="inspect_config",
                         help="TOML file path or installed preset builtin:NAME")
    train_parser = commands.add_parser("train", help="Train a new run or resume last.pt")
    train_parser.add_argument("--config", dest="train_config",
                              help="TOML file path or installed preset builtin:NAME")
    train_parser.add_argument("--output", type=Path, help="New run directory")
    train_parser.add_argument("--resume", type=Path, help="Resume last.pt or last_step.pt")
    train_parser.add_argument("--epochs", type=int, help="Total target epochs, including completed epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size, or upper bound for automatic sizing")
    train_parser.add_argument("--batch-size-mode", choices=("auto", "manual"), default="auto",
                              help="Batch sizing for new or resumed training (default: auto)")
    train_parser.add_argument("--checkpoint-every-steps", type=int,
                              help="Save last_step.pt every N optimizer updates; 0 disables (default)")
    train_parser.add_argument("--no-progress", action="store_true", help="Disable progress bars on stderr")
    train_parser.add_argument("--execution-mode", choices=("max_speed", "balanced", "analyze"),
                              help="Override diagnostics without changing the training algorithm")
    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a saved checkpoint")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--device", help="Optional device override; uses fp32")
    args = parser.parse_args(argv)
    if args.command is not None and args.config is not None:
        parser.error("Place --config after the train or inspect-config subcommand")
    try:
        if args.command == "inspect-config" or (args.command is None and args.config is not None):
            path = args.inspect_config if args.command else args.config
            config = load_config(path)
            with logging_session(config):
                log_output(json.dumps(asdict(config), indent=2))
        elif args.command == "train":
            from .run import train
            config = load_config(args.train_config) if args.train_config else None
            train(config, output=args.output, resume=args.resume, epochs=args.epochs,
                  batch_size=args.batch_size, batch_size_mode=args.batch_size_mode,
                  checkpoint_every_steps=args.checkpoint_every_steps,
                  progress_bar=not args.no_progress,
                  **({"execution_mode": args.execution_mode} if args.execution_mode is not None else {}))
        elif args.command == "evaluate":
            from .run import evaluate
            log_output(json.dumps(evaluate(args.checkpoint, device=args.device), indent=2))
        else:
            parser.error("Choose train, evaluate, or inspect-config --config PATH")
    except ImportError as exc:
        parser.error(f"Missing runtime dependency: {exc}. Install this project in an environment with PyTorch.")
    except (OSError, UnicodeError, ValueError, RuntimeError, FloatingPointError,
            EOFError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    return 0
