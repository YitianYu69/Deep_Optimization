"""Execute the documented DDP recipe through a real two-process launcher."""

import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest

import torch
from torch import distributed as dist

if __package__:
    from .launcher import run_torchrun
else:
    from launcher import run_torchrun


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class DistributedExampleBlackBoxTests(unittest.TestCase):
    def run_recipe(self, *arguments):
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                       "OMP_NUM_THREADS": "1", "DEEPOPT_LOGGING_LEVEL": "WARNING"}
        result = run_torchrun([str(ROOT / "examples/distributed/ddp_training.py"), *map(str, arguments)],
                             env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(len(rows), 1, result.stdout)
        self.assertEqual(rows[0]["distributed"]["world_size"], 2)
        self.assertEqual(rows[0]["distributed"]["global_rank"], 0)
        self.assertEqual(rows[0]["validation"]["samples"], 16)
        return rows[0]

    def test_two_rank_recipe_and_checkpoint_resume(self):
        self.assertEqual(self.run_recipe()["state"]["completed_epochs"], 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "run")
            first = self.run_recipe("--output", output, "--epochs", "1")
            resumed = self.run_recipe("--output", output, "--resume", "--epochs", "2")
            self.assertEqual(first["state"]["optimizer_steps"], 2)
            self.assertEqual(resumed["state"]["optimizer_steps"], 4)
            self.assertEqual(resumed["state"]["completed_epochs"], 2)
            payload = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(payload["distributed"]["world_size"], 2)
            self.assertEqual(len(payload["distributed"]["ranks"]), 2)

    def test_documented_fp8_ddp_core_snippet(self):
        documentation = (ROOT / "docs/fp8/README.md").read_text(encoding="utf-8")
        section = documentation.split("## Train with the DDP core", 1)[1].split("\n## ", 1)[0]
        snippet = section.split("```python\n", 1)[1].split("```", 1)[0]
        bare_section = section.split("### Own the DDP loop or call DDPCore directly", 1)[1]
        bare_block = bare_section.split("```python\n", 1)[1].split("```", 1)[0]
        start = snippet.index("    runtime = TrainingRuntime(")
        end = snippet.index("\n\ndef main():", start)
        before = (
            "initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()\n"
            "updates = []\n"
            "optimizer.register_step_post_hook(lambda *_: updates.append(1))\n"
        )
        after = '''
final = torch.cat([p.detach().flatten() for p in model.parameters()])
assert torch.isfinite(final).all()
assert not torch.equal(initial, final), "Bare DDPCore did not update parameters"
gathered = [torch.empty_like(final) for _ in range(context.world_size)]
torch.distributed.all_gather(gathered, final)
for parameters in gathered:
    torch.testing.assert_close(parameters, final, rtol=0, atol=0)
log_output(json.dumps({
    "core": type(core).__name__, "world_size": context.world_size,
    "updates": len(updates), "loss": loss.detach().item(),
    "converted_modules": report.converted_modules,
}))
'''
        bare_snippet = snippet[:start] + textwrap.indent(before + bare_block + after, "    ") + snippet[end:]
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                       "OMP_NUM_THREADS": "1", "DEEPOPT_LOGGING_LEVEL": "WARNING"}
        with tempfile.TemporaryDirectory() as directory:
            for name, code in (("trainer", snippet), ("bare", bare_snippet)):
                with self.subTest(api=name):
                    script = Path(directory, f"fp8_ddp_{name}.py")
                    script.write_text(code, encoding="utf-8")
                    result = run_torchrun([str(script)], env=environment, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                    self.assertEqual(len(rows), 1, result.stdout)
                    self.assertEqual(rows[0]["core"], "DDPCore")
                    self.assertEqual(rows[0]["world_size"], 2)
                    self.assertEqual(rows[0]["updates"], 6)
                    self.assertTrue(rows[0]["converted_modules"])
                    self.assertTrue(torch.isfinite(torch.tensor(rows[0]["loss"])).item())


if __name__ == "__main__":
    unittest.main()
