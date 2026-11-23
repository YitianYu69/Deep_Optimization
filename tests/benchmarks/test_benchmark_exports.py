"""Portable benchmark exports retain full comparisons and isolated workers."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "benchmarks/export_comparison_benchmarks.py"


class BenchmarkExportTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash"), "Bash is unavailable")
    def test_attention_launcher_requires_preflight_and_forwards_resume(self):
        with tempfile.TemporaryDirectory(prefix="attention launch ") as temporary:
            directory = Path(temporary)
            destination = directory / "benchmarks"
            subprocess.run([sys.executable, "-S", str(EXPORTER), "--output", str(destination)], check=True, capture_output=True)
            recorder = directory / "python-recorder"
            recorder.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ['RECORD_FILE'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ['FAIL_PREFLIGHT'] == '1' and 'preflight' in sys.argv[1]:
    raise SystemExit(7)
if 'alignment' in sys.argv[1]:
    raise SystemExit(8)
''')
            recorder.chmod(0o755)
            for failed, count in ((True, 1), (False, 8)):
                record = directory / f"calls-{failed}.jsonl"
                result = subprocess.run([shutil.which("bash"), str(destination / "run_attention_recovery.sh"),
                    str(directory / "results with spaces"), "--resume"], capture_output=True, text=True,
                    env={**os.environ, "BENCHMARK_PYTHON": str(recorder), "RECORD_FILE": str(record),
                         "FAIL_PREFLIGHT": "1" if failed else "0"})
                self.assertEqual(result.returncode, 1, result.stderr)
                calls = [json.loads(line) for line in record.read_text().splitlines()]
                self.assertEqual(len(calls), count)
                self.assertNotIn("--resume", calls[0])
                self.assertTrue(all("--resume" in call for call in calls[1:]))

    @unittest.skipUnless(shutil.which("bash"), "Bash is unavailable")
    def test_focused_launcher_forwards_resume_and_continues_after_a_failed_stage(self):
        with tempfile.TemporaryDirectory(prefix="launcher export ") as temporary:
            directory = Path(temporary)
            destination = directory / "benchmarks"
            subprocess.run([sys.executable, "-S", str(EXPORTER), "--output", str(destination)], check=True, capture_output=True)
            recorder = directory / "python-recorder"
            recorder.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ['RECORD_FILE'], 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
if any(name in ' '.join(sys.argv) for name in ('h100_norm_recovery.json', 'hopper_layer_tuning.json', 'hopper_v4_norm_boundary.json', 'hopper_v5_norm_boundary.json')):
    raise SystemExit(7)
''')
            recorder.chmod(0o755)
            for mode, count in (("--focused", 5), ("--recovery-v3", 11), ("--recovery-v4", 10), ("--recovery-v5", 14)):
                for resume in (False, True):
                    record = directory / f"{mode}-{resume}.jsonl"
                    result = subprocess.run([shutil.which("bash"), str(destination / "run_full_comparisons.sh"),
                        str(directory / "results with spaces"), mode, "--resume" if resume else "--no-resume"],
                        env={**os.environ, "BENCHMARK_PYTHON": str(recorder), "RECORD_FILE": str(record)}, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertTrue(record.is_file(), result.stderr)
                    calls = [json.loads(line) for line in record.read_text().splitlines()]
                    self.assertEqual(len(calls), count)
                    self.assertTrue(all(("--resume" in call) == resume for call in calls))

    def test_export_contains_only_declared_files_and_runs_without_adjacent_helpers(self):
        with tempfile.TemporaryDirectory(prefix="benchmark export ") as temporary:
            destination = Path(temporary) / "comparison.v1"
            destination.mkdir()
            (destination / "unrelated.txt").write_text("Unrelated existing output must not enter the bundle")
            exported = subprocess.run([sys.executable, "-S", str(EXPORTER), "--output", str(destination)],
                                      check=True, capture_output=True, text=True)
            description = json.loads(exported.stdout)
            self.assertGreaterEqual(description["coverage"]["all_settings_tasks"], 17094)
            self.assertEqual(description["coverage"]["vit_crossed_interaction_tasks"], 240)
            self.assertEqual(description["coverage"]["runtime_speed_requested_tasks_by_matrix"]["h100_runtime_speed_comparison"], 40)
            self.assertEqual(description["archive"], str(destination.resolve()) + ".zip")
            with zipfile.ZipFile(description["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertNotIn(destination.name + "/unrelated.txt", archive.namelist())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertNotIn("unrelated.txt", manifest["files"])
            for name in ("run_runtime_speed.sh", "RUN_BENCHMARKS.txt", "test_runtime_speed_cuda.py",
                         "h100_runtime_speed_comparison.json", "h100_runtime_speed_native.json",
                         "benchmark_kernel_backends.py", "benchmark_fp8_eager.py", "benchmark_activations.py",
                         "benchmark_pooling.py", "benchmark_convolution.py", "benchmark_compiler.py", "kernel_backends.md"):
                self.assertIn(name, manifest["files"])
            guide = (destination / "RUN_BENCHMARKS.txt").read_text(encoding="utf-8")
            self.assertIn("bash run_runtime_speed.sh", guide)
            self.assertIn("bash run_optimization_recovery.sh", guide)
            environment = dict(os.environ)
            dry_output = Path(temporary) / "kernel-matrix"
            backend_matrix = subprocess.run([sys.executable, str(destination / "benchmark_kernel_backends.py"),
                "--device", "cpu", "--workloads", "activation", "layer_norm", "pooling", "compiler",
                "--dry-run", "--output", str(dry_output)], env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(backend_matrix.returncode, 0, backend_matrix.stderr)
            matrix = json.loads((dry_output / "summary.json").read_text())["matrix"]
            self.assertEqual(len(matrix), 12)
            self.assertTrue(all(Path(row["command"][1]).parent == destination.resolve() for row in matrix))

            runner = destination / "benchmark_do_ops.py"
            tree = ast.parse(runner.read_text())
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and isinstance(node.targets[0], ast.Name)
                              and node.targets[0].id == "_BENCHMARK_SOURCES")
            sources = ast.literal_eval(assignment.value)
            for name, source in sources.items():
                self.assertEqual(source, (ROOT / "benchmarks/analysis" / name).read_text())
            isolated = Path(temporary) / "isolated"
            isolated.mkdir()
            shutil.copyfile(runner, isolated / runner.name)
            subprocess.run([sys.executable, "-S", runner.name, "--help"], cwd=isolated,
                           check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, "-S", runner.name, "--config",
                            str(destination / "h100_vit_recovery.json"), "--require-fp8-plan", "--dry-run",
                            "--output", str(isolated / "plan")], cwd=isolated,
                           check=True, capture_output=True, text=True)
            plan = json.loads((isolated / "plan/plan.json").read_text())
            self.assertEqual({task["case"] for task in plan["tasks"]}, set(manifest["cases"]))
            self.assertEqual(len(plan["tasks"]), 5)
            self.assertFalse((isolated / "analysis").exists())

            # The parent must keep materialized sources alive while its worker runs.
            subprocess.run([sys.executable, runner.name, "--ops", "linear", "--small", "--device", "cpu",
                            "--cases", "do_bf16", "--warmup", "0", "--iterations", "1", "--repeats", "1",
                            "--phase-steps", "0", "--no-memory", "--output", str(isolated / "results")],
                           cwd=isolated, check=True, capture_output=True, text=True, timeout=120)
            summary = json.loads((isolated / "results/summary.json").read_text())
            self.assertEqual(summary["counts"], {"ok": 1, "skipped": 0, "error": 0})
            self.assertTrue(summary["process_isolation"])

    def test_export_rejects_a_destination_that_overwrites_canonical_benchmarks(self):
        result = subprocess.run([sys.executable, "-S", str(EXPORTER), "--output", str(EXPORTER.parent / "analysis")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("canonical sources cannot be overwritten", result.stderr)


if __name__ == "__main__":
    unittest.main()
