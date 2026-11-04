"""Enable automatic FP8 preparation through the ordinary Trainer API.

Run: python examples/optimization/fp8_training.py
     python examples/optimization/fp8_training.py --trace-mode modules --block-size 32
     python examples/optimization/fp8_training.py --device cuda --backend triton

Reference execution emulates quantized GEMMs; Triton requires a compatible NVIDIA
GPU. Parameters remain floating point. Activation compression/sparsity are not enabled.
"""

import argparse
from dataclasses import asdict
import json

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Trainer
from deepopt.config import FP8Config
from deepopt.logger import log_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("reference", "triton"), default="reference")
    parser.add_argument("--trace-mode", choices=("full", "modules", "auto"), default="modules")
    parser.add_argument("--block-size", type=int, help="Square scaling blocks for all operands; omitted means tensorwise")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                        help="Runtime precision; bf16 requires CUDA and is separate from the FP8 format")
    args = parser.parse_args()
    torch.manual_seed(17)
    block_shape = None if args.block_size is None else (args.block_size, args.block_size)

    settings = FP8Config(
        enabled=True, backend=args.backend, trace_mode=args.trace_mode,
        block_shapes={role: block_shape for role in ("activation", "weight", "grad")},
    )
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    # Preparation reuses master parameters, so this optimizer keeps its ownership.
    # Use trainer.model afterward: explicit full tracing can return a new root.
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, device=args.device,
                      precision=args.precision, fp8=settings)
    batches = [(torch.randn(4, 32), torch.randn(4, 8)) for _ in range(3)]
    metrics = trainer.train_epoch(batches)
    log_output(json.dumps({"transformation": asdict(trainer.runtime.transformation_report), "backend": args.backend,
                           "device": args.device, "precision": args.precision,
                           "block_shape": block_shape, "updates": trainer.state.optimizer_steps,
                           "loss": metrics["loss"]}))


if __name__ == "__main__":
    main()
