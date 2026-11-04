"""Dictionary diagnostic policies preserve preset semantics and ownership."""

from collections import UserDict
from copy import deepcopy
import unittest

import torch
from torch import nn

from deepopt import DiagnosticPolicy, Trainer, TrainingRuntime, autocast
from deepopt.engine.execution.policy import ModeController


class DictionaryPolicyBlackBoxTests(unittest.TestCase):
    def test_trainer_and_runtime_accept_diagnostics_mapping(self):
        for mode in ("max_speed", "balanced", "analyze"):
            with self.subTest(mode=mode):
                raw = {"mode": mode}
                model = nn.Linear(2, 1)
                reference = deepcopy(model)
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                                  loss_fn=nn.functional.mse_loss, diagnostics=raw)
                typed = Trainer(reference, torch.optim.SGD(reference.parameters(), lr=0.01),
                                loss_fn=nn.functional.mse_loss, diagnostics=DiagnosticPolicy.for_mode(mode))
                batch = [(torch.ones(3, 2), torch.zeros(3, 1))]
                trainer.train_epoch(batch)
                typed.train_epoch(batch)
                self.assertEqual(trainer.runtime.diagnostics, typed.runtime.diagnostics)
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                runtime = TrainingRuntime(model, diagnostics=UserDict(raw))
                self.assertEqual(runtime.diagnostics, DiagnosticPolicy.for_mode(mode))
                self.assertEqual(raw, {"mode": mode})

    def test_fp8_mapping_shorthand_matches_dictionary(self):
        torch.manual_seed(74)
        model = nn.Linear(7, 3)
        inputs = torch.randn(4, 7)
        with autocast("cpu", enabled=False, fp8={"backend": "reference"}):
            expected = model(inputs)
        with autocast("cpu", enabled=False, fp8=UserDict({"backend": "reference"})):
            actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, model(inputs)))


class DictionaryPolicyWhiteBoxTests(unittest.TestCase):
    def test_preset_defaults_overrides_and_owned_fields(self):
        raw = {"mode": "analyze", "loss_finite_every": 3}
        controller = ModeController(diagnostics=raw)
        raw["loss_finite_every"] = 1
        self.assertEqual(controller.policy.loss_finite_every, 3)
        self.assertEqual(controller.policy.gradient_finite_every, 1)
        self.assertTrue(controller.policy.inspect_weights)
        self.assertTrue(controller.policy.inspect_activations)
        typed = DiagnosticPolicy.for_mode("max_speed")
        self.assertIs(ModeController(diagnostics=typed).policy, typed)

    def test_invalid_and_conflicting_policies_are_rejected(self):
        for raw in ({"mode": "fast"}, {"unknown": True}, {"loss_finite_every": 0},
                    {"inspect_weights": 1}, {"mode": "max_speed", "loss_finite_every": 1}):
            with self.subTest(raw=raw), self.assertRaises((TypeError, ValueError)):
                ModeController(diagnostics=raw)
        with self.assertRaisesRegex(ValueError, "Choose execution_mode or diagnostics"):
            ModeController(execution_mode="balanced", diagnostics={"mode": "analyze"})


if __name__ == "__main__":
    unittest.main()
