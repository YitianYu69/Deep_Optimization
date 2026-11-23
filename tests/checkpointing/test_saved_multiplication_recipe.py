"""Exact resume must distinguish the new multiplication coefficient recipe."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

import deepopt
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint


class SavedMultiplicationRecipeTests(unittest.TestCase):
    def test_old_recipe_is_rejected_before_any_model_state_is_loaded(self):
        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.5))
            def forward(self, x):
                return self.scale * x

        def trainer():
            model = Scale()
            return deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                device="cpu", loss_fn=nn.MSELoss(), config={"runtime": {
                    "activation_compression": {"enabled": True, "bits": 2, "group_size": 1024}}})

        source, target = trainer(), trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(path, source, generators={}, history=[])
            payload = read_checkpoint(path)
        identity = payload["trainer_settings"]["activation_compression_recipe"]
        self.assertIn("multiplication_coefficient", identity)
        self.assertIn("round-then-sum", identity["multiplication_gradient"])
        restore_checkpoint(payload, target, generators={}, batch_size_mode="manual")
        with torch.no_grad():
            target.model.scale.fill_(7)
        before = deepcopy(target.model.state_dict())
        for old in ({"version": 1, "layer_norm_xhat": "centered-forward-second-moment"},
                    {**identity, "multiplication_gradient": "fp32-fused-product-reduction"}):
            with self.subTest(recipe=old):
                payload["trainer_settings"]["activation_compression_recipe"] = old
                with self.assertRaisesRegex(ValueError, "Activation compression numerical recipe"):
                    restore_checkpoint(payload, target, generators={}, batch_size_mode="manual")
                for name, value in target.model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
