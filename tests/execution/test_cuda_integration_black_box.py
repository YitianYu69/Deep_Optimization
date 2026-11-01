"""Hardware-gated eager/compiled FP32/BF16 numerical and transfer checks."""

from copy import deepcopy
import unittest

import torch
from torch import nn

from deepopt import Trainer


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))

    def forward(self, inputs):
        return inputs + self.block(inputs)


class VAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean, self.log_variance, self.decoder = nn.Linear(8, 4), nn.Linear(8, 4), nn.Linear(4, 8)

    def forward(self, inputs):
        value, noise = inputs
        latent = self.mean(value) + (.5 * self.log_variance(value)).exp() * noise
        return self.decoder(latent)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is required")
class CUDAModeIntegrationTests(unittest.TestCase):
    def test_modes_compilation_precision_and_pinned_transfers(self):
        for factory in (ResidualModel, VAEModel):
            torch.manual_seed(7)
            template = factory()
            x, noise = torch.randn(4, 8).pin_memory(), torch.randn(4, 4).pin_memory()
            inputs = (x, noise) if factory is VAEModel else x
            target = torch.zeros(4, 8).pin_memory()
            for precision in ("fp32", "bf16"):
                if precision == "bf16" and not torch.cuda.is_bf16_supported():
                    continue
                for compiled in (False, True):
                    reference = None
                    for mode in ("max_speed", "balanced", "analyze"):
                        with self.subTest(model=factory.__name__, precision=precision, compiled=compiled, mode=mode):
                            model = deepcopy(template).cuda()
                            if compiled:
                                model = torch.compile(model)
                            trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=.001), device="cuda",
                                              precision=precision, accumulation_steps=2, execution_mode=mode,
                                              non_blocking=True, loss_fn=nn.functional.mse_loss)
                            trainer.train_epoch([(inputs, target)] * 3)
                            parameters = [parameter.detach().cpu().clone() for parameter in model.parameters()]
                            if reference is None:
                                reference = parameters
                            else:
                                for actual, expected in zip(parameters, reference):
                                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
