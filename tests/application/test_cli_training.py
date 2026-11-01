import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainingCLITests(unittest.TestCase):
    def test_train_resume_evaluate_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text('''name = "cli-test"
seed = 7
[training]
epochs = 1
batch_size = 7
learning_rate = 0.02
checkpoint_every_steps = 1
[application.synthetic]
train_samples = 17
validation_samples = 9
''')
            output = root / "run"

            def cli(*args):
                result = subprocess.run([sys.executable, "-m", "deepopt", *map(str, args)],
                                        capture_output=True, text=True, timeout=60,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                self.assertEqual(result.returncode, 0, result.stderr)
                return result

            first = cli("train", "--config", config, "--output", output)
            self.assertEqual(json.loads(first.stdout)["epoch"], 1)
            first_logs = list((output / "logs").glob("*.log"))
            self.assertEqual(len(first_logs), 1)
            first_text = first_logs[0].read_text()
            self.assertIn("Training started", first_text)
            self.assertIn("Epoch 1/1", first_text)
            self.assertIn("Automatic batch-size planning skipped", first_text)
            self.assertIn('"epoch": 1', first_text)
            self.assertIn("Saved step checkpoint", first_text)
            self.assertNotIn("\x1b", first_text)
            continued = cli("train", "--resume", output / "last.pt", "--epochs", 2)
            self.assertEqual(json.loads(continued.stdout)["epoch"], 2)
            self.assertEqual(len(list((output / "logs").glob("*.log"))), 2)
            self.assertEqual(first_logs[0].read_text(), first_text)
            result = json.loads(cli("evaluate", "--checkpoint", output / "last.pt", "--device", "cpu").stdout)
            self.assertEqual(result["epoch"], 2)
            self.assertEqual(result["validation"]["samples"], 9)
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 2)
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last_step.pt").is_file())
            # Refinish the final epoch from its last optimizer update, without
            # extending the total epoch target or duplicating log records.
            repeated = cli("train", "--resume", output / "last_step.pt", "--epochs", 2)
            self.assertEqual(json.loads(repeated.stdout)["epoch"], 2)
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
