"""Release preflight stays read-only until the benchmark freeze exists."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.benchmarks.test_qualification_bundle_export import _Fixture


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/compiler/export_qualification_bundle.py"
HELPERS = ("benchmarks/analysis/numerical_ipc.py", "benchmarks/compiler/model_numerics.py",
    "benchmarks/compiler/inductor_benchmark_artifacts.py", "examples/vae/qualification.py",
    "tests/benchmarks/test_bf16_policy_matrix.py", "docs/execution/bf16_storage_policy.md",
    "benchmarks/compiler/benchmark_vision_compression.py", "benchmarks/compiler/vision_compression_support.py",
    "docs/execution/backend_qualification.md", "tests/compiler/test_native_vision_models_cuda.py")


class QualificationBundlePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = _Fixture(self.temporary.name)
        for name in HELPERS:
            self.fixture.write(name, (ROOT / name).read_bytes())
        self.fixture.freeze_benchmark()

    def command(self, destination, *options):
        return subprocess.run([sys.executable, "-B", "-S", str(self.fixture.root / SCRIPT.relative_to(ROOT)),
            "--output", str(destination), "--wheel", str(self.fixture.wheel),
            "--expected-wheel-sha256", self.fixture.wheel_sha,
            "--production-freeze", str(self.fixture.production), *options], capture_output=True, text=True, timeout=30)

    def test_stdlib_dry_run_is_read_only_and_preserves_installed_only_source_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-exported"
            result = self.command(output, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            names = plan["source_sha256"]
            self.assertFalse(any(name.startswith("src/") or "__pycache__" in name for name in names))
            for name in ("benchmarks/analysis/numerical_ipc.py", "benchmarks/compiler/model_numerics.py",
                         "benchmarks/compiler/inductor_benchmark_artifacts.py", "examples/vae/qualification.py",
                         "tests/benchmarks/test_bf16_policy_matrix.py", "docs/execution/bf16_storage_policy.md"):
                self.assertIn(name, names)
            for name in ("benchmarks/compiler/benchmark_vision_compression.py", "benchmarks/compiler/vision_compression_support.py",
                         "docs/execution/backend_qualification.md", "tests/compiler/test_native_vision_models_cuda.py"):
                self.assertIn(name, names)
            self.assertFalse(plan["benchmark_freeze_verified"])
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".zip").exists())

    def test_export_without_benchmark_freeze_is_refused_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-exported"
            result = self.command(output)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Export is held", result.stderr)
            self.assertFalse(output.exists())

    def test_incomplete_benchmark_freeze_cannot_authorize_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = root / "benchmark-freeze.json"
            freeze.write_text(json.dumps({"source_sha256": {}}))
            output = root / "not-exported"
            result = self.command(output, "--dry-run", "--benchmark-freeze", str(freeze))
            self.assertEqual(result.returncode, 2)
            self.assertIn("Benchmark freeze missing/changed", result.stderr)
            self.assertFalse(output.exists())

    def test_historical_destination_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "historical"
            output.mkdir()
            marker = output / "retained.txt"
            marker.write_text("historical bytes")
            result = self.command(output, "--dry-run")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(marker.read_text(), "historical bytes")

    def test_changed_production_source_is_rejected_without_rewriting_its_freeze(self):
        before = self.fixture.production.read_bytes()
        self.fixture.write("src/deepopt/__init__.py", b"# edited after freeze\n")
        output = Path(self.temporary.name) / "not-exported"
        result = self.command(output, "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Production sources changed", result.stderr)
        self.assertEqual(self.fixture.production.read_bytes(), before)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
