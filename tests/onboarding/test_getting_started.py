"""Execute the published Python walkthrough as consecutive notebook cells."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


class GettingStartedTests(unittest.TestCase):
    def test_walkthrough_runs_in_order_and_evaluates_saved_best(self):
        root = Path(__file__).resolve().parents[2]
        guide = (root / "docs/getting-started/README.md").read_text()
        blocks = re.findall(r"```python\n(.*?)```", guide, re.S)
        script = """
import json
from pathlib import Path
import sys
import torch
namespace = {'__name__': '__main__'}
for index, block in enumerate(json.loads(sys.argv[1]), 1):
    exec(compile(block, f'getting-started cell {index}', 'exec'), namespace)
saved = torch.load('runs/progressive/checkpoints/best.pt', weights_only=True)
assert namespace['best_results']['mse'] == saved['state']['best_score']
assert Path('runs/step-example/last_step.pt').is_file()
last = torch.load('runs/progressive/checkpoints/last.pt', weights_only=True)
assert last['state']['completed_epochs'] == 3
print('walkthrough passed')
"""
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), str(root / "src"))),
               "DEEPOPT_LOGGING_LEVEL": "WARNING"}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", script, json.dumps(blocks)],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("walkthrough passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
