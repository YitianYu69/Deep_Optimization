"""Target-CUDA regression for the shared compressed ViT fullgraph recipe."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AttentionCaptureCUDATests(unittest.TestCase):
    def test_shared_compressed_vit_keeps_fullgraph_native_ownership(self):
        from analysis.benchmark_do_ops import make_plan, parser
        from analysis.catalog import normalize_workload
        from analysis.engine import Session
        from analysis.matrix import DEFAULT_SETTINGS, merge
        from analysis.measurement import errors, snapshot
        from deepopt.compiler import compilation_report

        snapshots = []
        self.addCleanup(torch.compiler.reset)
        for execution in ("eager", "compile"):
            torch.compiler.reset()
            settings = merge(deepcopy(DEFAULT_SETTINGS), {"compression": {"group_size": 1024}, "runtime": {"binding": "prepared",
                "execution": execution, "optimizer": "sgd", "optimizer_warmup_steps": 0,
                "compile": {"backend": "deepopt", "fullgraph": True, "dynamic": False,
                    "options": {"require_full_aot": True}}}})
            task = {"id": "attention-capture-cuda", "settings": settings, "case": "compression",
                "workload": normalize_workload({"op": "model", "name": "vit-block", "shape": [1, 17, 96],
                    "factory": "analysis.compiler_workloads:build", "factory_kwargs": {
                        "kind": "vit-block", "batch": 1, "tokens": 17, "width": 96, "hidden": 192, "output": 96}})}
            options = make_plan(parser().parse_args(["--device", "cuda:0", "--small", "--warmup", "0",
                "--iterations", "1", "--repeats", "1", "--phase-steps", "0"]))["options"]
            session = Session(task, options)
            try:
                snapshots.append(snapshot(session))
                if execution == "compile":
                    report = compilation_report(session.model).as_dict()
                    self.assertEqual(report["captured_graphs"], 1)
                    self.assertGreater(report["native_autodiff_operations"], 0)
                    self.assertFalse(report["legacy_backward_operations"])
                    self.assertEqual(report["runtime_fallbacks"], 0)
                    self.assertTrue(all(not graph["eager_autograd_operations"] for graph in report["graphs"]))
                    operations = {rule["operation"] for graph in report["graphs"] for rule in graph["differentiation_rules"]}
                    self.assertIn("autodiff_sdpa", operations)
                    self.assertIn("autodiff_native_activation", operations)
            finally:
                session.close()
        parity = errors(snapshots[1], snapshots[0], .02, .002, max_relative_l2=.1)
        self.assertTrue(parity["passed"], parity)


if __name__ == "__main__":
    unittest.main()
