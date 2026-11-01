"""Own an alternating generator/discriminator loop with two runtime helpers.

Run: python examples/optimization/alternating_updates.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from deepopt.logger import log_output
from deepopt import TrainingRuntime


def main():
    torch.manual_seed(42)
    real_samples = 1.5 + 0.4 * torch.randn(32, 2)
    loader = DataLoader(real_samples, batch_size=8, shuffle=True)
    generator = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
    discriminator = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.01)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.01)
    generation = TrainingRuntime(generator, generator_optimizer)
    discrimination = TrainingRuntime(discriminator, discriminator_optimizer)

    generator.train()
    discriminator.train()
    for epoch in range(3):
        for real in loader:
            # First update the discriminator using detached generated samples.
            discrimination.zero_grad()
            with torch.no_grad():
                fake = generator(torch.randn(real.shape[0], 3))
            real_logits, fake_logits = discriminator(real), discriminator(fake)
            discriminator_loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                                  + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
            discrimination.backward(discriminator_loss)
            discrimination.step()

            # Then update only the generator, differentiating through the discriminator.
            generation.zero_grad()
            discriminator.requires_grad_(False)
            try:
                fake = generator(torch.randn(real.shape[0], 3))
                logits = discriminator(fake)
                generator_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
                generation.backward(generator_loss)
                generation.step()
            finally:
                discriminator.requires_grad_(True)
        log_output({"epoch": epoch + 1, "last_generator_loss": generator_loss.item(),
               "last_discriminator_loss": discriminator_loss.item()})


if __name__ == "__main__":
    main()
