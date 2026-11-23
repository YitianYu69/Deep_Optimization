"""Fresh-process allocator configuration preserves options and resume identity."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/analysis/benchmark_allocator_compare.py"
spec = importlib.util.spec_from_file_location("allocator_comparison", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AllocatorComparisonTests(unittest.TestCase):
    def test_startup_options_preserve_user_options_and_require_native(self):
        original = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:False", "UNRELATED": "keep"}
        result = module.allocator_environment(True, original)
        self.assertEqual(result["UNRELATED"], "keep")
        self.assertIn("max_split_size_mb:128", result["PYTORCH_ALLOC_CONF"])
        self.assertIn("expandable_segments:True", result["PYTORCH_ALLOC_CONF"])
        self.assertNotIn("True", original["PYTORCH_CUDA_ALLOC_CONF"])
        with self.assertRaisesRegex(ValueError, "backend:native"):
            module.allocator_environment(True, {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            module.allocator_environment(True, {"PYTORCH_ALLOC_CONF": "expandable_segments:False",
                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

    def test_dry_run_needs_no_torch_and_separates_children_and_fill_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--dry-run", "--resume",
                "--config", str(ROOT / "benchmarks/analysis/configs/hopper_v4_vit.json"), "--output", str(out)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            jobs = json.loads((out / "allocator-experiment.json").read_text())
            self.assertEqual(len(jobs), 2)
            for job in jobs:
                self.assertIn("--resume", job["command"])
                self.assertIn("--fill-uninitialized-memory", job["command"])
                self.assertFalse(job["performance_fill"])
                self.assertTrue(job["correctness_fill"])
            self.assertNotEqual(jobs[0]["allocator_environment"], jobs[1]["allocator_environment"])
