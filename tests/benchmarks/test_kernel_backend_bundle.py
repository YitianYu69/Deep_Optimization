"""Exported backend runners and workers must share the same CLI contract."""

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "benchmarks/export_comparison_benchmarks.py"
FAMILIES = {
    "benchmark_fp8_eager.py": "fp8/benchmark_fp8_eager.py",
    "benchmark_activations.py": "activation/benchmark_activations.py",
    "benchmark_normalization.py": "normalization/benchmark_normalization.py",
    "benchmark_pooling.py": "pooling/benchmark_pooling.py",
    "benchmark_convolution.py": "convolution/benchmark_convolution.py",
    "benchmark_compiler.py": "compiler/benchmark_compiler.py",
    "benchmark_activation_pack.py": "activation_compression/benchmark_activation_pack.py",
    "benchmark_activation_unpack.py": "activation_compression/benchmark_activation_unpack.py",
}
CODECS = {"benchmark_activation_pack.py", "benchmark_activation_unpack.py"}


# Dry-run and preflight must work without importing Torch or GPU compilers.
# Blocking those imports also protects this test on hosts with CUDA installed.
STDLIB_RUN = """import runpy, sys
class NoGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'tilelang'}:
            raise AssertionError('Dry-run imported GPU dependency: ' + fullname)
sys.meta_path.insert(0, NoGpuImports())
script, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
namespace = runpy.run_path(script, run_name='bundle_contract')
if STALE_CHECK:
    def forbidden_provenance(*args, **kwargs):
        raise AssertionError('Stale worker reached provenance/device discovery')
    namespace['main'].__globals__['run_provenance'] = forbidden_provenance
raise SystemExit(namespace['main']())
"""


class KernelBackendBundleTests(unittest.TestCase):
    def test_focused_export_rejects_an_old_library_wheel_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "deep_optimization-0.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as package:
                package.writestr("deepopt/_benchmarking.py", "# stale helper without timing controls\n")
            destination = root / "mismatched"
            result = subprocess.run([sys.executable, "-S", str(EXPORTER), "--kernel-backends-only",
                "--wheel", str(wheel), "--output", str(destination)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("does not match", result.stderr)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".zip").exists())

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="kernel backend bundle ")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        destination = cls.root / "exported benchmarks"
        result = subprocess.run(
            [sys.executable, "-S", str(EXPORTER), "--output", str(destination)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        description = json.loads(result.stdout)
        extracted = cls.root / "downloaded archive"
        with zipfile.ZipFile(description["archive"]) as archive:
            if archive.testzip() is not None:
                raise AssertionError("Exporter produced a corrupt archive")
            archive.extractall(extracted)
        cls.bundle = (extracted / destination.name).resolve()
        cls.manifest = json.loads((cls.bundle / "manifest.json").read_text())

    def environment(self):
        # Library imports come from this build, while every worker/helper comes
        # from the extracted ZIP. No canonical benchmark directory is supplied.
        return {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}

    def dry_run(self, output, *, stale_check=False, bundle=None):
        wrapper = STDLIB_RUN.replace("STALE_CHECK", repr(stale_check))
        return subprocess.run(
            [sys.executable, "-S", "-c", wrapper,
             str((bundle or self.bundle) / "benchmark_kernel_backends.py"),
             "--preset", "representative", "--workloads", "linear", "layer_norm", "pooling",
             "--backends", "triton", "tilelang", "mixed", "--device", "cuda:0",
             "--warmup", "10", "--iterations", "50", "--repeats", "5",
             "--time-budget", "2", "--no-autotune", "--dry-run", "--output", str(output)],
            cwd=self.root, env=self.environment(), capture_output=True, text=True, timeout=60,
        )

    def test_archive_contains_current_family_workers_and_portable_helpers(self):
        sources = {**FAMILIES, "benchmark_kernel_backends.py": "benchmark_kernel_backends.py",
                   "_gpu_lease.py": "_gpu_lease.py"}
        for name, relative in sources.items():
            with self.subTest(script=name):
                payload = (self.bundle / name).read_bytes()
                self.assertEqual(payload, (ROOT / "benchmarks" / relative).read_bytes())
                self.assertEqual(sha256(payload).hexdigest(), self.manifest["files"][name])
        for helper in ("analysis/__init__.py", "analysis/_gpu_lease.py", "analysis/scheduler.py",
                       "analysis/matrix.py", "analysis/reporting.py"):
            self.assertTrue((self.bundle / helper).is_file(), helper)
            self.assertIn(helper, self.manifest["files"])

    def test_family_help_exposes_all_forwarded_backend_and_sampling_flags(self):
        common = {"--device", "--warmup", "--output", "--deepopt-backend",
                  "--deepopt-kernel-policy", "--time-budget"}
        for name in FAMILIES:
            with self.subTest(script=name):
                # Compiler imports Torch at module scope. Its help still needs
                # neither CUDA nor either optional kernel compiler.
                command = [sys.executable]
                if name != "benchmark_compiler.py":
                    command.append("-S")
                result = subprocess.run(
                    [*command, str(self.bundle / name), "--help"], cwd=self.root,
                    env=self.environment(), capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                flags = common | ({"--steps"} if name == "benchmark_compiler.py"
                                  else {"--iterations", "--repeats"})
                if name not in CODECS:
                    flags |= {"--autotune", "--no-autotune"}
                for flag in flags:
                    self.assertIn(flag, result.stdout)

    def test_fresh_archive_dry_run_uses_matching_standalone_workers(self):
        output = self.root / "fresh matrix"
        result = self.dry_run(output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(report["matrix"]), 9)
        self.assertEqual(report["results"], [])
        self.assertEqual({row["workload"] for row in report["matrix"]},
                         {"linear", "layer_norm", "pooling"})
        self.assertEqual({row["backend"] for row in report["matrix"]},
                         {"triton", "tilelang", "mixed"})
        for row in report["matrix"]:
            command = row["command"]
            self.assertEqual(Path(command[1]).parent, self.bundle)
            self.assertEqual(sha256(Path(command[1]).read_bytes()).hexdigest(), row["script_sha256"])
            self.assertEqual(command[command.index("--time-budget") + 1], "2.0")
            self.assertIn("--no-autotune", command)
            if row["backend"] == "mixed":
                self.assertIn("--deepopt-kernel-policy", command)

    def test_focused_archive_preserves_relative_workers_and_passes_preflight(self):
        destination = self.root / "focused backend export"
        exported = subprocess.run(
            [sys.executable, "-S", str(EXPORTER), "--kernel-backends-only", "--output", str(destination)],
            cwd=self.root, env=self.environment(), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(exported.returncode, 0, exported.stdout + exported.stderr)
        description = json.loads(exported.stdout)
        extracted = self.root / "focused download"
        with zipfile.ZipFile(description["archive"]) as archive:
            self.assertIsNone(archive.testzip())
            archive.extractall(extracted)
        bundle = (extracted / destination.name).resolve()
        manifest = json.loads((bundle / "manifest.json").read_text())
        for relative in (*FAMILIES.values(), "benchmark_kernel_backends.py", "_gpu_lease.py",
                         "analysis/__init__.py", "analysis/_gpu_lease.py", "analysis/scheduler.py",
                         "analysis/matrix.py", "analysis/reporting.py"):
            with self.subTest(file=relative):
                payload = (bundle / relative).read_bytes()
                self.assertEqual(payload, (ROOT / "benchmarks" / relative).read_bytes())
                self.assertEqual(sha256(payload).hexdigest(), manifest["files"][relative])
        imported = subprocess.run(
            [sys.executable, "-S", "-c",
             "import pathlib, sys; sys.path.insert(0, sys.argv[1]); "
             "import analysis.scheduler, analysis._gpu_lease; "
             "assert pathlib.Path(analysis.scheduler.__file__).resolve().parent == "
             "pathlib.Path(sys.argv[1]) / 'analysis'", str(bundle)],
            cwd=self.root, env=self.environment(), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)
        output = self.root / "focused matrix"
        result = self.dry_run(output, bundle=bundle)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(report["matrix"]), 9)
        expected = {"linear": "fp8/benchmark_fp8_eager.py",
                    "layer_norm": "normalization/benchmark_normalization.py",
                    "pooling": "pooling/benchmark_pooling.py"}
        for row in report["matrix"]:
            self.assertEqual(Path(row["command"][1]), bundle / expected[row["workload"]])
        self.assertEqual(report["results"], [])

    def test_stale_child_is_rejected_before_output_or_device_discovery(self):
        cases = (("benchmark_fp8_eager.py", "--time-budget"),
                 ("benchmark_normalization.py", "--autotune"),
                 ("benchmark_pooling.py", "--deepopt-backend"))
        for name, flag in cases:
            with self.subTest(script=name, missing=flag):
                script = self.bundle / name
                original = script.read_text()
                tree = ast.parse(original)
                changed = 0
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "add_argument"):
                        for argument in node.args:
                            if isinstance(argument, ast.Constant) and argument.value == flag:
                                argument.value = "--stale-" + flag.removeprefix("--")
                                changed += 1
                self.assertEqual(changed, 1, "Could not construct a stale parser fixture")
                output = self.root / ("stale-" + name)
                try:
                    script.write_text(ast.unparse(tree) + "\n")
                    result = self.dry_run(output, stale_check=True)
                finally:
                    script.write_text(original)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(name, result.stderr)
                missing = "--no-autotune" if flag == "--autotune" else flag
                self.assertIn(missing, result.stderr)
                self.assertIn("bundle", result.stderr.lower())
                self.assertNotIn("Stale worker reached provenance", result.stderr)
                self.assertFalse(output.exists(), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
