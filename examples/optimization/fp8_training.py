"""Transform a small model and train with reference or Triton FP8 operations.

Run: python examples/optimization/fp8_training.py
     python examples/optimization/fp8_training.py --device cuda --backend triton
"""

import argparse
from dataclasses import asdict
import json

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Trainer
from deepopt.logger import log_output
from deepopt.opt import transform_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("reference", "triton"), default="reference")
    parser.add_argument("--trace-mode", choices=("full", "modules", "auto"), default="full")
    args = parser.parse_args()
    torch.manual_seed(17)

    plan = {
        "fp8_training": {
            "backend": args.backend,
            "fp8_dtypes": {
                "activation": torch.float8_e4m3fn,
                "weight": torch.float8_e4m3fn,
                "grad": torch.float8_e4m3fn if args.backend == "triton" else torch.float8_e5m2,
            },
        },
        "block_shapes": {role: None for role in ("activation", "weight", "grad")},
    }
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).to(args.device)
    model, report = transform_model(model, plan=plan, trace_mode=args.trace_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, device=args.device)
    batches = [(torch.randn(4, 32), torch.randn(4, 8)) for _ in range(3)]
    metrics = trainer.train_epoch(batches)
    log_output(json.dumps({"transformation": asdict(report), "backend": args.backend,
                           "device": args.device, "updates": trainer.state.optimizer_steps,
                           "loss": metrics["loss"]}))


if __name__ == "__main__":
    main()
