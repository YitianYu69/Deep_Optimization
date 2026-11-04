"""A self-supervised step aligns two modalities and reports ordinary metrics.

Run: python examples/training/multimodal.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from deepopt.logger import log_output
from deepopt import Trainer


def alignment_step(model, batch):
    image_embedding = model["image"](batch["image_features"])
    text_embedding = model["text"](batch["text_features"])
    alignment = F.mse_loss(image_embedding, text_embedding)
    # Population standard deviation also handles a final one-example batch.
    spread = (image_embedding.std(dim=0, correction=0) - 1).square().mean()
    spread = spread + (text_embedding.std(dim=0, correction=0) - 1).square().mean()
    return {
        "loss": alignment + 0.1 * spread,
        "predictions": image_embedding,
        "targets": text_embedding,
        "sample_count": image_embedding.shape[0],
    }


def main():
    # Paired observations share a latent source; no class labels are needed.
    torch.manual_seed(42)
    latent = torch.randn(48, 2)
    images = latent @ torch.randn(2, 6) + 0.05 * torch.randn(48, 6)
    texts = latent @ torch.randn(2, 4) + 0.05 * torch.randn(48, 4)
    pairs = [{"image_features": image, "text_features": text} for image, text in zip(images, texts)]
    train = DataLoader(pairs[:32], batch_size=8, shuffle=True,
                       generator=torch.Generator().manual_seed(43))
    validation = DataLoader(pairs[32:], batch_size=8)
    model = nn.ModuleDict({"image": nn.Linear(6, 3), "text": nn.Linear(4, 3)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    # Explicit predictions/targets are routed to both metrics and detached for reporting.
    trainer = Trainer(model, optimizer, step_fn=alignment_step,
                      metrics={"alignment_mse": MeanSquaredError(), "alignment_mae": MeanAbsoluteError()})
    trainer.fit(train, validation, epochs=3, progress_bar=False,
                callbacks=[lambda _, row: log_output(row)])


if __name__ == "__main__":
    main()
