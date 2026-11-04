"""Compile only the model, then train with the ordinary DeepOpt runtime.

Run: python examples/execution/compile_training.py
     python examples/execution/compile_training.py --backend inductor
     python examples/execution/compile_training.py --device cuda --backend inductor

The default aot_eager backend demonstrates compilation/autograd without native
kernel generation. Use inductor to exercise the platform's compiler toolchain.
"""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Trainer
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.logger import log_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="aot_eager", help="PyTorch compiler backend; e.g. aot_eager or inductor")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fullgraph", action="store_true", help="Reject graph breaks")
    args = parser.parse_args()
    torch.manual_seed(17)

    model = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1)).to(args.device)
    settings = CompileConfig(enabled=True, backend=args.backend, fullgraph=args.fullgraph)
    model = compile_model(model, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, device=args.device, loss_fn=F.mse_loss,
                      execution_mode="balanced")
    inputs = torch.randn(24, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    batches = list(zip(inputs.split(8), targets.split(8)))
    # The first forward triggers tracing/compilation; runtime and optimizer stay eager.
    metrics = trainer.train_epoch(batches)
    log_output(json.dumps({"backend": args.backend, "device": args.device,
                           "updates": trainer.state.optimizer_steps, "loss": metrics["loss"]}))


if __name__ == "__main__":
    main()
