"""Hardware validation for configured Inductor compilation."""

import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.config import config_from_dict
from deepopt.experiment import build_experiment


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the Inductor GPU check")
class CUDACompileBlackBoxTests(unittest.TestCase):
    def test_configured_inductor_matches_eager_training(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        def build(enabled):
            config = config_from_dict({"name": "cuda-compile", "seed": 9,
                "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01, "optimizer": "sgd"},
                "runtime": {"device": "cuda", "execution_mode": "max_speed",
                            "compile": {"enabled": enabled, "backend": "inductor", "fullgraph": True}}})
            x = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
            dataset = TensorDataset(x, x.sum(dim=1, keepdim=True))
            return build_experiment(config, model_factory=lambda: nn.Linear(2, 1),
                                    dataset_factory=lambda: (dataset, None), loss_fn=nn.functional.mse_loss)
        compiled, loader, _, _ = build(True)
        eager, reference_loader, _, _ = build(False)
        compiled.train_epoch(loader)
        eager.train_epoch(reference_loader)
        for actual, expected in zip(compiled.model.parameters(), eager.model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
