"""Compiler training on real local Gloo ranks."""

import json
import importlib.util
from pathlib import Path
import unittest

from torch import distributed as dist
from tests.distributed.launcher import run_torchrun


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(), "Gloo is required"
)
class CompilerDistributedTests(unittest.TestCase):
    def test_native_and_prepared_fp8_updates_match_independent_ddp_oracle(self):
        root = Path(__file__).resolve().parents[2]
        specification = importlib.util.spec_from_file_location(
            "distributed_installed_runtime", root / "benchmarks/compiler/installed_runtime.py"
        )
        helper = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(helper)
        environment = helper.child_environment(root)
        environment.update(OMP_NUM_THREADS="1", DEEPOPT_LOGGING_LEVEL="ERROR")
        result = run_torchrun(
            [str(Path(__file__).with_name("_compiler_worker.py"))],
            # Include the validated installation and bundled test support;
            # never forward source-checkout paths to local ranks.
            env=environment,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        self.assertEqual(
            rows,
            [
                {"compiled_ddp": True, "fp8": False, "world_size": 2},
                {"compiled_ddp": True, "fp8": True, "world_size": 2},
            ],
        )


if __name__ == "__main__":
    unittest.main()
