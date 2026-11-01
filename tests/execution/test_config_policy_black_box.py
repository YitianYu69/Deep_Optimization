from contextlib import redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest

from deepopt.config import ConfigError, config_from_dict
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.run import train


class ConfigPolicyTests(unittest.TestCase):
    def config(self, runtime=None):
        return config_from_dict({"name": "modes", "seed": 7,
            "training": {"epochs": 1, "batch_size": 4, "learning_rate": .01},
            "application": {"synthetic": {"train_samples": 8, "validation_samples": 4}},
            "runtime": runtime or {}, "experiment_identity": {"recipe_version": "modes-v1"}})

    def test_policy_configuration_roundtrips_and_rejects_invalid_settings(self):
        config = self.config({"execution_mode": "max_speed", "non_blocking": True,
                              "mode_schedule": {"2": "analyze", "3": "max_speed"}})
        self.assertEqual(config_from_dict(asdict(config)), config)
        for runtime in ({"execution_mode": "quick"}, {"non_blocking": 1},
                        {"mode_schedule": {"0": "analyze"}}, {"mode_schedule": {"1": "analyze", "01": "balanced"}},
                        {"diagnostics": {"loss_finite_every": 0}},
                        {"diagnostics": {"mode": "max_speed", "gradient_finite_every": 1}},
                        {"execution_mode": "analyze", "diagnostics": {}}):
            with self.subTest(runtime=runtime), self.assertRaises(ConfigError):
                self.config(runtime)

    def test_configured_run_can_resume_in_another_mode(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory, "run")
            train(self.config({"checks": False}), output=root, execution_mode="max_speed", progress_bar=False)
            train(resume=root / "last.pt", epochs=2, execution_mode="analyze", progress_bar=False)
            payload = read_checkpoint(root / "last.pt")
            self.assertEqual(payload["state"]["completed_epochs"], 2)
            self.assertEqual(payload["diagnostics"]["active_mode"], "analyze")
            self.assertEqual(payload["experiment_identity"]["required"], {"recipe_version": "modes-v1"})


if __name__ == "__main__":
    unittest.main()
