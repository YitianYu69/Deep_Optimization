"""Mandatory real-CUDA acceptance for v6; CPU interpreters cannot replace it."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

CUDA=torch.cuda.is_available() and torch.version.hip is None


@unittest.skipUnless(CUDA,"NVIDIA CUDA is required; skipped tests do not establish release readiness")
class RecoveryV6CudaTests(unittest.TestCase):
    def setUp(self):
        enabled=torch.are_deterministic_algorithms_enabled()
        warning=torch.is_deterministic_algorithms_warn_only_enabled()
        filling=torch.utils.deterministic.fill_uninitialized_memory
        self.addCleanup(torch.use_deterministic_algorithms,enabled,warn_only=warning)
        self.addCleanup(setattr,torch.utils.deterministic,"fill_uninitialized_memory",filling)
    def test_minimized_compiler_ownership_codec_and_linear_candidates(self):
        from analysis.gpu_preflight import main
        from analysis.engine import installed_build_fingerprint
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"preflight.json"
            code=main(["--output",str(path),"--expected-source-sha256",installed_build_fingerprint()])
            report=json.loads(path.read_text())
            failures=[r for r in report["checks"] if r["status"]!="ok"]
            self.assertEqual(code,0,json.dumps(failures,indent=2))

    @unittest.skipUnless(importlib.util.find_spec("timm") and importlib.util.find_spec("torchvision"),
                         "timm and torchvision are required; CNN weights remain None")
    def test_all_three_small_models_execute_every_mode_eager_and_graph(self):
        from tests.benchmarks.test_do_analysis_benchmark import cli,ROOT
        from analysis.engine import run_task
        config=ROOT/"benchmarks/analysis/configs/hopper_v6_preflight.json"
        args=cli.parser().parse_args(["--config",str(config),"--sweep","cartesian","--warmup","1","--iterations","1",
            "--repeats","1","--phase-steps","0","--no-memory","--deterministic","--fill-uninitialized-memory",
            "--max-numerical-elements","1024","--require-fp8-plan"])
        plan=cli.make_plan(args)
        self.assertEqual(len(plan["tasks"]),30)
        identities={}
        with tempfile.TemporaryDirectory() as temp:
            for task in plan["tasks"]:
                folder=Path(temp)/task["id"];folder.mkdir()
                row=run_task(task,plan["options"],folder)
                self.assertTrue(row["finite_after_timing"])
                self.assertTrue(row["numerical_snapshot"]["full_tensor_finite"])
                identities.setdefault(task["workload"]["name"],set()).add(
                    (row["initial_state_sha256"],row["rng_baseline_sha256"]))
            self.assertTrue(all(len(value)==1 for value in identities.values()))


if __name__=="__main__":unittest.main()
