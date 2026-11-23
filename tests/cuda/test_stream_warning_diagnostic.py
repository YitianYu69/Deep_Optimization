"""Real CUDA acceptance of the diagnostic; warning absence is not assumed."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/cuda_graph/analyze_stream_warning.py"


class StreamWarningDiagnosticCUDATests(unittest.TestCase):
    def run_cases(self, options, ranks=1):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(SCRIPT), *options, "--steps", "2", "--accumulation", "2",
                                     "--timing-steps", "0", "--output-dir", directory],
                                    text=True, capture_output=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(Path(json.loads(result.stdout)["report"]).read_text())
            for job in report["jobs"]:
                self.assertEqual(len(job["ranks"]), ranks)
                for row in job["ranks"]:
                    self.assertTrue(row["correctness"]["passed"])
                    self.assertTrue(row["training_replay"]["passed"])
                    # A version-dependent native warning must remain visible.
                    for warning in row["warnings"]:
                        self.assertIn("phase", warning)
                        self.assertIn("stream", warning)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_capture_runtime_and_trainer_reports(self):
        self.run_cases(["--cases", "deepopt_graph", "runtime", "trainer", "--execution-mode", "max_speed",
                        "--input-grad", "--validation-transition"])

    @unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
    def test_fp8_and_compressed_fp8_current_scaling(self):
        self.run_cases(["--cases", "fp8", "fp8_compressed", "--execution-mode", "max_speed"])

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and torch.distributed.is_nccl_available(), "Two NCCL GPUs are required")
    def test_partial_capture_and_ddp_reports(self):
        self.run_cases(["--cases", "deepopt_graph", "runtime", "trainer", "--execution-mode", "max_speed",
                        "--launcher", "torchrun", "--ranks", "2"], ranks=2)


if __name__ == "__main__":
    unittest.main()
