"""Choose default checks, disable diagnostics, or add a plain callable observer.

Run: python examples/observability/train_observers.py --mode all
CUDA: python examples/observability/train_observers.py --mode all --device cuda
"""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt.logger import log_output
from deepopt import Trainer, TrainingRuntime
from deepopt.observers import FiniteCheck, TensorEvent


def run(mode, device="cpu"):
    torch.manual_seed(17)
    inputs = torch.randn(12, 2)
    targets = inputs.sum(dim=1, keepdim=True)
    train = DataLoader(TensorDataset(inputs[:8], targets[:8]), batch_size=4)
    validation = DataLoader(TensorDataset(inputs[8:], targets[8:]), batch_size=4)
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    def log_activation(event: TensorEvent):
        # Reductions run on the event's device; .item() reads one scalar and
        # synchronizes CUDA. Log selectively in throughput-sensitive workloads.
        log_output(json.dumps({"kind": event.kind, "name": event.name,
                          "device": str(event.tensors[0].device),
                          "mean": event.tensors[0].mean().item()}))

    log_activation.kinds = {"activation"}
    checks = {"default": True, "off": False,
              "selective": [FiniteCheck(kinds={"loss", "gradients"}), log_activation]}[mode]
    runtime = TrainingRuntime(model, optimizer, device=device, checks=checks)

    def step(model, batch):
        features, target = batch
        hidden = model[1](model[0](features))
        runtime.observe("activation", hidden, name="hidden")
        # Keep the original differentiable tensor in the computation.
        prediction = model[2](hidden)
        return {"loss": F.mse_loss(prediction, target), "sample_count": target.shape[0]}

    # A supplied runtime owns the checks setting; the trainer inherits it.
    trainer = Trainer(runtime, step_fn=step)

    def report_epoch(_, row):
        log_output(json.dumps({"mode": mode, "train_loss": row["train"]["loss"],
                          "validation_loss": row["validation"]["loss"]}))

    trainer.fit(train, validation, epochs=1, callbacks=[report_epoch], progress_bar=False)
    log_output(f"{mode}: {trainer.state.optimizer_steps} optimizer updates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("default", "off", "selective", "all"), default="all",
                        help="Observer configuration to demonstrate; separate from execution_mode")
    parser.add_argument("--device", default="cpu", help="Training device: cpu, cuda, or cuda:N")
    args = parser.parse_args()
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if device.type not in ("cpu", "cuda"):
        parser.error("Choose cpu, cuda, or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        parser.error("The requested CUDA device index is unavailable")
    for mode in (("default", "off", "selective") if args.mode == "all" else (args.mode,)):
        run(mode, device=device)


if __name__ == "__main__":
    main()
