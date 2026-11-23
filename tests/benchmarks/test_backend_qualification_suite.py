"""Portable qualification plans retain requested failures and physical evidence."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from compiler import qualify_backend as suite


class BackendQualificationSuiteTests(unittest.TestCase):
    def plan(self, *arguments):
        return suite.make_plan(suite.parser().parse_args(list(arguments)))

    def test_cpu_and_metadata_profiles_never_plan_a_real_local128_payload(self):
        cpu = self.plan("--profile", "cpu")
        self.assertFalse(any(stage["require_observed"] for stage in cpu["stages"]))
        metadata = self.plan("--profile", "metadata")
        self.assertEqual(len(metadata["stages"]), 2)
        self.assertTrue(all(stage["validator"] == "metadata" for stage in metadata["stages"]))
        self.assertTrue(all("examples.vae.audit_activation_memory" in stage["command"] for stage in metadata["stages"]))

    def test_gpu_plan_requires_real_tests_and_valid_fusion_phases(self):
        plan = self.plan("--profile", "gpu-full")
        self.assertEqual(plan["timing"]["trials"], 3)
        strict = [row for row in plan["stages"] if row["validator"] == "tests_strict"]
        self.assertTrue(strict)
        self.assertTrue(all("--require-cuda" in row["command"] and "--fail-on-skip" in row["command"] for row in strict))
        for row in plan["stages"]:
            command = row["command"]
            if row["name"] == "fusion-packed-multiply":
                self.assertEqual(command[command.index("--phases") + 1:command.index("--compression")], ("backward", "both"))
            if row["name"] == "fusion-shared-packs":
                self.assertEqual(command[command.index("--phases") + 1:command.index("--compression")], ("forward", "both"))
                self.assertEqual(row["validator"], "fusion_rejection")
            if row["name"].startswith("fusion-gemm-compressed-"):
                self.assertEqual(command[command.index("--phases") + 1:command.index("--backends")], ("backward", "both"))
        self.assertTrue(any("gelu-tanh" in row["name"] for row in plan["stages"]))

    def test_producer_and_model_pack_sharing_suites_are_required_in_cpu_and_cuda_plans(self):
        cpu = self.plan("--profile", "cpu")
        cpu_stage = next(row for row in cpu["stages"] if row["name"] == "fusion-contracts")
        self.assertIn("tests.compiler.test_frozen_producer_fusion.FrozenProducerFusionTests", cpu_stage["command"])
        self.assertIn("tests.compiler.test_fusion_model_pack_sharing.VAEFusionPackSharingTests", cpu_stage["command"])
        self.assertNotIn("tests.compiler.test_frozen_producer_fusion.FrozenProducerFusionCudaTests", cpu_stage["command"])
        gpu = self.plan("--profile", "gpu-full")
        gpu_stage = next(row for row in gpu["stages"] if row["name"] == "fusion-cuda")
        self.assertEqual(gpu_stage["validator"], "tests_strict")
        self.assertIn("--fail-on-skip", gpu_stage["command"])
        self.assertIn("tests.compiler.test_frozen_producer_fusion.FrozenProducerFusionCudaTests", gpu_stage["command"])
        self.assertIn("tests.compiler.test_fusion_model_pack_sharing.VAEFusionPackSharingCudaTests", gpu_stage["command"])

    def test_efficiency_options_preserve_every_stage_and_its_measurement_counts(self):
        default = self.plan("--profile", "gpu-full")
        strict = self.plan("--profile", "gpu-full", "--fusion-worker-isolation", "strategy")
        self.assertEqual([row["name"] for row in default["stages"]], [row["name"] for row in strict["stages"]])
        self.assertTrue(default["execution_policy"]["cpu_contracts_requested"])
        self.assertEqual(default["execution_policy"]["gpu_timing_concurrency"], 1)
        self.assertTrue(default["execution_policy"]["benchmark_cases_preserved"])
        for row in default["stages"]:
            command = row["command"]
            if command[0] != "benchmarks/compiler/isolated_fusion.py":
                continue
            self.assertEqual(command[command.index("--worker-isolation") + 1], "trial")
            self.assertEqual(command[command.index("--phase-iterations") + 1], "3")
            self.assertEqual(command[command.index("--iterations") + 1], "50")
            self.assertEqual(command[command.index("--repeats") + 1], "5")
            self.assertEqual(command[command.index("--trials") + 1], "3")
            self.assertEqual(command[command.index("--batch") + 1], "32")
            self.assertEqual(command[command.index("--width") + 1], "768")
            self.assertEqual(command[command.index("--spatial") + 1:command.index("--spatial") + 3], ("16", "16"))

    def test_representative_smoke_keeps_smoke_case_count_and_uses_larger_shapes(self):
        small = self.plan("--profile", "gpu-smoke")
        large = self.plan("--profile", "gpu-smoke", "--workload", "representative")
        self.assertEqual([row["name"] for row in small["stages"]], [row["name"] for row in large["stages"]])
        self.assertEqual(small["timing"], large["timing"])
        kernels = next(row for row in large["stages"] if row["validator"] == "kernels")["command"]
        self.assertIn("6304x768x3072", kernels)
        self.assertEqual(kernels[kernels.index("--min-measure-ms") + 1], "50")
        model = next(row for row in small["stages"] if row["validator"] == "models")["command"]
        self.assertEqual(model[model.index("--width") + 1], "24")

    def test_progress_publishes_active_worker_phase_and_clears_after_stage(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()) as output:
            directory = Path(temporary)
            phase = directory / "worker-phase.json"
            suite.write_json(phase, {"phase": "cold_compile_and_first_update"})
            suite.write_json(directory / "fusion-progress.json", {"active": {"worker_phase_path": str(phase)}})
            log = directory / "stage.log"
            log.write_text("starting worker\n")
            stage = {"name": "fusion-dense", "validator": "fusion", "report": str(directory / "fusion-summary.json")}
            summary = {"status": "running", "stages": []}
            with suite.report_stage_progress(summary, stage, log, directory, 1, 2, 3600):
                report = json.loads((directory / "qualification-summary.json").read_text())
                self.assertEqual(report["active_stage"]["worker_phase"]["phase"], "cold_compile_and_first_update")
            self.assertNotIn("active_stage", summary)
            self.assertIn("cold_compile_and_first_update", output.getvalue())

    def test_every_required_unittest_selector_exists(self):
        plan = self.plan("--profile", "gpu-full")
        for row in plan["stages"]:
            command = row["command"]
            if "--modules" not in command:
                continue
            selectors = command[command.index("--modules") + 1:command.index("--device")]
            for selector in selectors:
                parts = selector.split(".")
                source = None
                for end in range(len(parts), 0, -1):
                    candidate = suite.ROOT.joinpath(*parts[:end]).with_suffix(".py")
                    if candidate.is_file():
                        source, attributes = candidate, parts[end:]
                        break
                self.assertIsNotNone(source, selector)
                nodes = ast.parse(source.read_text()).body
                for attribute in attributes:
                    found = next((node for node in nodes if getattr(node, "name", None) == attribute), None)
                    self.assertIsNotNone(found, selector)
                    nodes = getattr(found, "body", [])

    def test_a_correct_pair_without_specific_cuda_evidence_fails(self):
        stage = {"validator": "fusion", "require_observed": True}
        result = {"correctness_passed": True, "comparisons": [{"cuda_fusion_observed": False}]}
        self.assertFalse(suite.validate_report(stage, result)[0])
        result["comparisons"][0]["cuda_fusion_observed"] = True
        self.assertTrue(suite.validate_report(stage, result)[0])
        result["comparisons"] = []
        self.assertFalse(suite.validate_report(stage, result)[0])

    def test_kernel_summary_requires_every_worker_and_actual_generated_launches(self):
        stage = {"validator": "kernels", "require_observed": True}
        case = {"status": "ok", "backend": "triton", "execution": "cuda", "generated_launch_observed": True,
                "profile": {"cuda_kernel_count": 1}}
        result = {"passed": True, "results": [{"status": "ok", "source_unchanged": True, "results": [case]}]}
        self.assertTrue(suite.validate_report(stage, result)[0])
        bad = deepcopy(result)
        bad["results"][0]["results"][0]["profile"] = None
        self.assertFalse(suite.validate_report(stage, bad)[0])
        bad = deepcopy(result)
        bad["results"].append({"status": "error", "results": []})
        self.assertFalse(suite.validate_report(stage, bad)[0])

    def test_strict_test_stage_rejects_skipped_qualification(self):
        stage = {"validator": "tests_strict"}
        self.assertFalse(suite.validate_report(stage, {"passed": True, "qualified": False, "counts": {"skipped": 1}})[0])
        self.assertTrue(suite.validate_report(stage, {"passed": True, "qualified": True, "counts": {"skipped": 0}})[0])

    def test_unsupported_public_sharing_is_a_negative_control_only(self):
        stage = {"validator": "fusion_rejection"}
        row = {"transformation": "pack_cse", "correctness_passed": True,
               "status": "unsupported", "cuda_fusion_observed": False}
        data = {"correctness_passed": True, "comparisons": [row]}
        self.assertTrue(suite.validate_report(stage, data)[0])
        data["comparisons"].append({"transformation": "kernel_fusion", "correctness_passed": True,
                                    "status": "observed", "cuda_fusion_observed": True})
        self.assertTrue(suite.validate_report(stage, data)[0])
        row["cuda_fusion_observed"] = True
        self.assertFalse(suite.validate_report(stage, data)[0])

    def test_export_excludes_deepopt_and_runs_without_torch_for_planning(self):
        with tempfile.TemporaryDirectory(prefix="qualification portable ") as temporary:
            root = Path(temporary)
            exported = suite.export_bundle(root / "backend.zip")
            with zipfile.ZipFile(exported["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any(name.startswith(("src/", "deepopt/")) for name in archive.namelist()))
                self.assertNotIn("pyproject.toml", archive.namelist())
                for name in ("benchmarks/compiler/installed_runtime.py", "tests/distributed/_compiler_worker.py",
                             "benchmarks/compiler/isolated_fusion.py", "examples/vae/activation_memory_local128.toml"):
                    self.assertIn(name, archive.namelist())
                archive.extractall(root / "snapshot")
            subprocess.run([sys.executable, "-S", "benchmarks/compiler/qualify_backend.py", "--profile", "gpu-full",
                            "--dry-run", "--output", "planned"], cwd=root / "snapshot", check=True,
                           capture_output=True, text=True, timeout=30)
            plan = json.loads((root / "snapshot/planned/qualification-plan.json").read_text())
            self.assertTrue(plan["stages"])
            manifest = json.loads((root / "snapshot/bundle-manifest.json").read_text())
            self.assertEqual(suite.source_snapshot(root / "snapshot"), manifest["sources"])


if __name__ == "__main__":
    unittest.main()
