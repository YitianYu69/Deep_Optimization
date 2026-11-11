"""Portable benchmark exports retain full comparisons and isolated workers."""

import ast
import json
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
    def test_export_contains_only_declared_files_and_runs_without_adjacent_helpers(self):
        with tempfile.TemporaryDirectory(prefix="benchmark export ") as temporary:
            destination = Path(temporary) / "comparison.v1"
            destination.mkdir()
            (destination / "unrelated.txt").write_text("Unrelated existing output must not enter the bundle")
            exported = subprocess.run([sys.executable, "-S", str(EXPORTER), "--output", str(destination)],
                                      check=True, capture_output=True, text=True)
            description = json.loads(exported.stdout)
            self.assertEqual(description["coverage"]["all_settings_tasks"], 17094)
            self.assertEqual(description["coverage"]["vit_crossed_interaction_tasks"], 240)
            self.assertEqual(description["archive"], str(destination.resolve()) + ".zip")
            with zipfile.ZipFile(description["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertNotIn(destination.name + "/unrelated.txt", archive.namelist())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertNotIn("unrelated.txt", manifest["files"])

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
