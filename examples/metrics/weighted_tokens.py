"""Weight accumulated gradients and loss reports by valid tokens.

Run from the repository root: python examples/metrics/weighted_tokens.py
The recipe supplies valid-token counts explicitly; sample_count is throughput.
"""

import json

import torch
from torch import nn

from deepopt.logger import log_output
from deepopt import LossResult, Trainer


def batches(seed):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for lengths in ((2, 5), (3,), (1, 4, 5), (2, 3), (4,)):
        width = max(lengths)
        features = torch.randn(len(lengths), width, 3, generator=generator)
        targets = features.sum(dim=-1, keepdim=True)
        mask = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
        result.append({"features": features, "targets": targets, "mask": mask,
                       "valid_tokens": sum(lengths)})
    return result


def step(model, batch):
    predictions = model(batch["features"])
    per_token = (predictions - batch["targets"]).square().mean(dim=-1)
    count = batch["valid_tokens"]  # Positive Python count supplied by this recipe.
    objective = (per_token * batch["mask"]).sum() / count
    return {"loss": LossResult.from_mean(objective, weight=count),
            "sample_count": batch["features"].shape[0]}


def main():
    torch.manual_seed(7)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, step_fn=step, accumulation_steps=3)
    # Five microbatches produce one complete and one partial window per epoch.
    trainer.fit(batches(11), batches(29), epochs=3, progress_bar=False,
                callbacks=[lambda _, row: log_output(json.dumps(row))])
    log_output(json.dumps({"completed_epochs": trainer.state.completed_epochs,
                      "optimizer_steps": trainer.state.optimizer_steps}))


if __name__ == "__main__":
    main()
