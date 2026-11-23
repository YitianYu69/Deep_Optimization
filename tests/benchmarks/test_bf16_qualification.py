"""The compiler suite must qualify every declared, fresh numerical result."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
SPEC = importlib.util.spec_from_file_location("bf16_qualification", ROOT / "benchmarks/compiler/benchmark_bf16.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BF16QualificationTests(unittest.TestCase):
    def test_broker_startup_failure_releases_the_active_job(self):
        from compiler.model_numerics import ModelNumericalComparisons

        comparisons = ModelNumericalComparisons()
        job = {"trial": 0, "strategy": "eager", "expected_tasks": [{"id": "one"}]}
        with patch("analysis.numerical_ipc.NumericalSnapshotBroker", side_effect=OSError("no local sockets")):
            with self.assertRaisesRegex(OSError, "no local sockets"):
                with benchmark.numerical_job(job, "fresh", comparisons):
                    self.fail("failed broker cannot enter a numerical job")
        comparisons.clear_trial()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        from compiler.model_numerics import ModelNumericalComparisons
        self.numerics = ModelNumericalComparisons(receipt_timeout=0)

    def matrix(self):
        jobs = []
        for strategy in ("eager", "triton-native-control"):
            directory = self.directory / strategy
            task = {"id": "fixture", "workload": {"name": "linear-b8", "factory_kwargs": {"kind": "linear"}},
                    "case": "do_bf16", "settings": {"runtime": {"execution": "eager" if strategy == "eager" else "compile"}}}
            folder = directory / "cases" / task["id"]
            folder.mkdir(parents=True)
            row = {"status": "ok", "task": task, "run_id": strategy + "-run", "initial_state_sha256": "state",
                   "rng_baseline_sha256": "rng", "shared_optimizer_baseline_sha256": "optimizer",
                   "environment": {"deepopt_source_sha256": "build"}, "snapshot": None,
                   "snapshot_file": None, "timing": {"wall_ms": {"median": 1.}},
                   "measurement_policy": {"numerical_comparison": True, "numerical_snapshot_mode": "memory"}, "finite_after_timing": True,
                   "numerical_snapshot": {"full_tensor_finite": True, "original_shapes": {
                       "output.0": [4], "parameter_grad.weight": [4]},
                       "sampling": "deterministic", "max_sampled_elements_per_float_tensor": 65536},
                   "execution": {"compiler_report": {"static_regions": {"bf16_linear_programs": 1},
                       "native_autodiff_operations": 1, "runtime_fallbacks": 0, "graphs": [{"eager_autograd_operations": 0}]}}}
            benchmark.write_json(folder / "result.json", row)
            benchmark.write_json(directory / "plan.json", {"tasks": [task], "code_sha256": "code", "options": {"profile_steps": 0}})
            benchmark.write_json(directory / "summary.json", {"run_id": row["run_id"]})
            jobs.append({"trial": 0, "strategy": strategy, "status": "completed", "result": 0,
                         "directory": str(directory), "expected_tasks": [task], "plan_code_sha256": "code",
                         "analysis_run_id": row["run_id"]})
        self.refresh_evidence(jobs)
        return jobs

    def refresh_evidence(self, jobs, *, candidate_value=1.):
        from analysis.numerical_ipc import snapshot_descriptor
        from compiler.model_numerics import ModelNumericalComparisons

        self.numerics = ModelNumericalComparisons(receipt_timeout=0)
        for job in jobs:
            context = self.numerics.start_job(job, "test-suite")
            for task in job["expected_tasks"]:
                path = Path(job["directory"]) / "cases" / task["id"] / "result.json"
                row = json.loads(path.read_text())
                value = 1. if job["strategy"] == "eager" else candidate_value
                tensors = {"output.0": torch.ones(4) * value, "parameter_grad.weight": torch.ones(4) * .001}
                identity = {**context, "task": task, "worker_run_id": row["run_id"] + task["id"]}
                descriptor = snapshot_descriptor(identity, tensors, row["numerical_snapshot"])
                self.numerics.accept({"identity": identity, "tensors": tensors,
                                      "metadata": row["numerical_snapshot"], "descriptor": descriptor})
                row.update(comparison_input=descriptor, numerical_worker_run_id=identity["worker_run_id"])
                benchmark.write_json(path, row)
            self.numerics.end_job()

    def row_path(self, job):
        return Path(job["directory"]) / "cases/fixture/result.json"

    def update(self, job, mutate):
        path = self.row_path(job)
        row = json.loads(path.read_text())
        mutate(row)
        benchmark.write_json(path, row)

    def compare(self, jobs):
        return benchmark.compare_strategies(jobs, cuda=True, require_native=True, numerical_comparisons=self.numerics)

    def positive_matrix(self):
        jobs = self.matrix()
        jobs[1]["strategy"] = "triton-native-fused"
        for job in jobs:
            task = deepcopy(job["expected_tasks"][0])
            task["workload"].update(name="mlp-b8", factory_kwargs={"kind": "mlp"})
            job["expected_tasks"] = [task]
            plan_path = Path(job["directory"]) / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["tasks"] = [task]
            benchmark.write_json(plan_path, plan)
            self.update(job, lambda row: row.update(task=task))
        candidate = {"candidate_id": "graph0:fc1:forward:gelu", "selected": True,
                     "phase": "train", "execution_phase": "forward",
                     "profile_range": "deepopt.compiler.fc1.forward.forward",
                     "implementation_fingerprint": "source",
                     "kernel_evidence": {"required_kernel_roles": {"gemm_activation": ["_bf16_gemm"]}}}
        self.update(jobs[1], lambda row: row["execution"]["compiler_report"].update(
            fused_deepopt_regions=1, fusion_candidates=[candidate]))
        self.refresh_evidence(jobs)
        return jobs, candidate

    def test_positive_fixture_without_profile_remains_unqualified(self):
        jobs, _candidate = self.positive_matrix()
        result = self.compare(jobs)
        self.assertFalse(result["passed"])
        comparison, = result["comparisons"]
        self.assertTrue(comparison["expected_fusion_selected"])
        self.assertIsNone(comparison["fusion_execution_observed"])
        self.assertEqual(comparison["fusion_qualification"]["status"], "unknown")

    def test_only_selected_candidate_cuda_work_satisfies_positive_fixture(self):
        jobs, candidate = self.positive_matrix()
        unrelated = {"linked_cuda_launches_per_step": 1, "kernel_names": ["_bias_gradient_reduce"]}
        profiles = [
            {"compiler_regions": {"deepopt.compiler.fc2.backward.db": unrelated}},
            {"compiler_regions": {candidate["profile_range"]: unrelated}},
            {"compiler_regions": {candidate["profile_range"]: {
                "linked_cuda_launches_per_step": 1, "kernel_names": ["_bf16_gemm"]}}},
        ]
        for profile, accepted in zip(profiles, (False, False, True)):
            with self.subTest(profile=profile):
                self.update(jobs[1], lambda row: row.update(profile=profile))
                result = self.compare(jobs)
                self.assertEqual(result["passed"], accepted)
                self.assertEqual(result["comparisons"][0]["fusion_qualification"]["all_selected_observed"], accepted)

    def test_complete_fresh_matrix_passes_behavior_but_does_not_qualify_gpu_speed(self):
        result = self.compare(self.matrix())
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["comparisons"]), 1)
        self.assertFalse(result["comparisons"][0]["gpu_performance_qualified"])
        self.assertFalse(result["comparisons"][0]["fusion_execution_observed"])
        self.assertEqual(result["comparisons"][0]["fusion_qualification"]["status"], "unfused")

    def test_memory_evidence_is_required_and_never_loads_legacy_files(self):
        jobs = self.matrix()
        with patch.object(torch, "load", side_effect=AssertionError("disk load forbidden")):
            self.assertFalse(benchmark.compare_strategies(jobs)["passed"])
            self.assertTrue(self.compare(jobs)["passed"])

    def test_scalar_candidate_comparison_survives_trial_tensor_release(self):
        jobs = self.matrix()
        self.assertGreater(self.numerics.retained_bytes, 0)
        self.numerics.clear_trial()
        self.assertEqual(self.numerics.retained_bytes, 0)
        self.assertTrue(self.compare(jobs)["passed"])
        self.refresh_evidence(jobs, candidate_value=2.)
        self.numerics.clear_trial()
        self.assertFalse(self.compare(jobs)["passed"])

    def test_missing_or_unequal_sampling_metadata_fails_closed(self):
        jobs = self.matrix()
        original = json.loads(self.row_path(jobs[1]).read_text())
        for name, value in (("sampling", "changed"), ("max_sampled_elements_per_float_tensor", 2)):
            with self.subTest(name=name):
                altered = deepcopy(original)
                altered["numerical_snapshot"][name] = value
                benchmark.write_json(self.row_path(jobs[1]), altered)
                self.assertFalse(self.compare(jobs)["passed"])

    def test_duplicate_unexpected_and_old_payloads_poison_the_live_job(self):
        from analysis.numerical_ipc import snapshot_descriptor
        from compiler.model_numerics import ModelNumericalComparisons

        jobs = self.matrix()
        row = json.loads(self.row_path(jobs[0]).read_text())
        for corruption in ("duplicate", "unexpected", "stale"):
            with self.subTest(corruption=corruption):
                store = ModelNumericalComparisons(receipt_timeout=0)
                context = store.start_job(jobs[0], "new-suite")
                identity = {**context, "task": row["task"], "worker_run_id": "fresh-worker"}
                tensors = {"output.0": torch.ones(4), "parameter_grad.weight": torch.ones(4) * .001}
                payload = {"identity": identity, "tensors": tensors, "metadata": row["numerical_snapshot"]}
                if corruption == "duplicate":
                    payload["descriptor"] = snapshot_descriptor(identity, tensors, payload["metadata"])
                    store.accept(payload)
                elif corruption == "unexpected":
                    identity["task"] = {**identity["task"], "id": "other-task"}
                else:
                    identity["suite_run_id"] = "old-suite"
                payload["descriptor"] = snapshot_descriptor(identity, tensors, payload["metadata"])
                with self.assertRaises(ValueError):
                    store.accept(payload)
                self.assertFalse(store.validate_row(row)["passed"])
                store.end_job()
                store.clear_trial()

    def test_eager_storage_bound_is_enforced_without_disk_fallback(self):
        from compiler.model_numerics import ModelNumericalComparisons

        jobs = self.matrix()
        row = json.loads(self.row_path(jobs[0]).read_text())
        store = ModelNumericalComparisons(max_retained_bytes=16)
        context = store.start_job(jobs[0], "bounded-suite")
        identity = {**context, "task": row["task"], "worker_run_id": "bounded-worker"}
        with self.assertRaises(MemoryError):
            store.accept({"identity": identity, "tensors": {"output.0": torch.ones(4),
                "parameter_grad.weight": torch.ones(4)}, "metadata": row["numerical_snapshot"],
                "descriptor": {"test": True}})
        self.assertEqual(store.retained_bytes, 0)
        self.assertFalse(list(self.directory.rglob("*.pt")))

    def real_eager_analysis_without_tensor_files(self, in_process):
        import analysis.benchmark_do_ops as analysis
        from compiler.model_numerics import ModelNumericalComparisons

        config = self.directory / "eager.json"
        benchmark.write_json(config, {"settings": {"compression": {"group_size": 1024}, "runtime": {"execution": "eager", "optimizer": "sgd"}},
            "workloads": [{"name": "linear-b1", "op": "model", "shape": [1, 257, 4],
                "factory": "analysis.compiler_workloads:build", "factory_kwargs": {
                    "kind": "linear", "batch": 1, "tokens": 257, "width": 4, "hidden": 8, "output": 4}}]})
        output = self.directory / "real"
        command = ["--config", str(config), "--device", "cpu", "--cases", "torch_bf16", "do_bf16", "compression",
            "--iterations", "1", "--warmup", "0", "--repeats", "1", "--phase-steps", "0", "--no-memory",
            "--numerical-snapshot-mode", "memory", "--output", str(output)]
        if in_process:
            command.append("--in-process")
        plan = analysis.make_plan(analysis.parser().parse_args(command))
        job = {"trial": 0, "strategy": "eager", "expected_tasks": plan["tasks"]}
        store = ModelNumericalComparisons(receipt_timeout=3)
        with patch.object(torch, "save", side_effect=AssertionError("tensor save forbidden")), \
                patch.object(torch, "load", side_effect=AssertionError("tensor load forbidden")):
            with benchmark.numerical_job(job, "real-suite", store):
                self.assertEqual(analysis.main(command, numerical_comparisons=store), 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["counts"]["ok"], 3)
        self.assertEqual(len(summary["comparisons"]), 2)
        self.assertTrue(all(row["accuracy_passed"] is not None for row in summary["comparisons"]))
        for row in summary["results"]:
            self.assertTrue(store.validate_row(row)["passed"])
            self.assertIsNone(row["snapshot"])
            self.assertIsNone(row["snapshot_file"])
        compressed = next(row for row in summary["results"] if row["task"]["case"] == "compression")
        self.assertIn("correctness_vs_uncompressed", compressed)
        self.assertFalse(list(output.rglob("*.pt")))
        self.assertFalse(list(output.rglob("*.pt.tmp")))
        store.clear_trial()
        self.assertEqual(store.retained_bytes, 0)

    def test_real_eager_analysis_compares_all_cases_without_tensor_files(self):
        self.real_eager_analysis_without_tensor_files(in_process=True)

    def test_fresh_worker_eager_analysis_compares_all_cases_without_tensor_files(self):
        self.real_eager_analysis_without_tensor_files(in_process=False)

    def test_memory_mode_cannot_resume_legacy_disk_evidence(self):
        import analysis.benchmark_do_ops as analysis

        arguments = analysis.parser().parse_args(["--device", "cpu", "--ops", "linear", "--small",
            "--numerical-snapshot-mode", "memory", "--resume"])
        with self.assertRaisesRegex(ValueError, "fresh workers"):
            analysis.make_plan(arguments)

    def test_missing_memory_channel_fails_without_disk_fallback(self):
        import analysis.benchmark_do_ops as analysis

        args = analysis.parser().parse_args(["--device", "cpu", "--ops", "linear", "--small",
            "--cases", "torch_bf16", "--iterations", "1", "--warmup", "0", "--repeats", "1",
            "--phase-steps", "0", "--no-memory", "--numerical-snapshot-mode", "memory"])
        plan = analysis.make_plan(args)
        options = {**plan["options"], "_numerical_worker_run_id": "missing-channel-worker"}
        with patch.object(torch, "save", side_effect=AssertionError("disk fallback forbidden")), \
                patch.dict(os.environ, {"DEEPOPT_NUMERICAL_CONTEXT": '{"suite_run_id":"missing-channel"}',
                                       "DEEPOPT_NUMERICAL_CHANNEL": "null"}):
            row = analysis.execute_task(plan["tasks"][0], options, self.directory / "no-channel")
        self.assertEqual(row["status"], "error")
        self.assertIn("private numerical channel", row["reason"])
        self.assertFalse(list(self.directory.rglob("*.pt")))
        self.assertFalse(list(self.directory.rglob("*.pt.tmp")))

    def test_worker_error_after_successful_numerical_payload_stays_failed(self):
        jobs = self.matrix()
        self.update(jobs[1], lambda row: row.update(status="error", reason="Worker exited 1 after result"))
        jobs[1].update(result=1, status="failed")
        result = self.compare(jobs)
        self.assertFalse(result["passed"])
        self.assertIn("execution_error", {row.get("reason") for row in result["comparisons"]})

    def test_nonzero_child_exit_overrides_fresh_successful_result(self):
        import analysis.benchmark_do_ops as analysis
        import analysis.engine as engine

        args = analysis.parser().parse_args(["--device", "cpu", "--ops", "linear", "--small",
            "--cases", "torch_bf16", "--output", str(self.directory / "worker-exit")])
        plan = analysis.make_plan(args)
        captured = []

        def execute(command, **kwargs):
            request_path = Path(command[-1])
            request = json.loads(request_path.read_text())
            benchmark.write_json(request_path.parent / "result.json", {
                "status": "ok", "run_id": request["run_id"], "task": request["task"]})
            return SimpleNamespace(returncode=9)

        def finalize(args, plan, results, run_id):
            captured.extend(results)
            return 1

        with patch.object(analysis.subprocess, "run", side_effect=execute), \
                patch.object(analysis, "finalize", side_effect=finalize), \
                patch.object(engine, "installed_build_fingerprint", return_value="build"), \
                patch.object(engine, "resume_environment_fingerprint", return_value="environment"), \
                patch.dict(os.environ, {"DEEPOPT_BENCHMARK_LEASE": ""}):
            self.assertEqual(analysis.run(args, plan), 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["status"], "error")
        self.assertIn("exited 9 after publishing", captured[0]["reason"])

    def test_declared_missing_job_or_task_cannot_disappear_from_qualification(self):
        jobs = self.matrix()
        self.row_path(jobs[1]).unlink()
        result = self.compare(jobs)
        self.assertFalse(result["passed"])
        self.assertIn("missing_or_invalid_result", {row.get("reason") for row in result["comparisons"]})
        jobs[1].update(status="pending", result=None)
        self.assertIn("job_incomplete_or_failed", {row.get("reason") for row in self.compare(jobs)["comparisons"]})

    def test_old_run_id_and_wrong_normalized_task_are_rejected(self):
        jobs = self.matrix()
        original = json.loads(self.row_path(jobs[1]).read_text())
        for field in ("run_id", "task"):
            with self.subTest(field=field):
                row = deepcopy(original)
                if field == "run_id":
                    row[field] = "old-run"
                else:
                    row[field]["settings"]["runtime"]["execution"] = "eager"
                benchmark.write_json(self.row_path(jobs[1]), row)
                result = self.compare(jobs)
                self.assertFalse(result["passed"])
                self.assertIn("stale_or_mismatched_result", {r.get("reason") for r in result["comparisons"]})

    def test_extra_stale_result_and_altered_plan_are_rejected(self):
        jobs = self.matrix()
        benchmark.write_json(Path(jobs[1]["directory"]) / "cases/old/result.json", {"status": "ok"})
        result = self.compare(jobs)
        self.assertFalse(result["passed"])
        self.assertIn("unexpected_result", {row.get("reason") for row in result["comparisons"]})
        benchmark.write_json(Path(jobs[1]["directory"]) / "plan.json", {"tasks": [], "code_sha256": "old"})
        self.assertIn("invalid_job_identity", {row.get("reason") for row in self.compare(jobs)["comparisons"]})

    def test_both_reference_and_candidate_must_be_finite_with_matching_original_shapes(self):
        jobs = self.matrix()
        originals = [json.loads(self.row_path(job).read_text()) for job in jobs]
        for side in (0, 1):
            for field in ("full_tensor_finite", "finite_after_timing", "original_shapes"):
                with self.subTest(side=side, field=field):
                    for job, original in zip(jobs, originals):
                        benchmark.write_json(self.row_path(job), original)
                    row = deepcopy(originals[side])
                    if field == "finite_after_timing":
                        row[field] = False
                    elif field == "original_shapes":
                        row["numerical_snapshot"][field]["output.0"] = [2, 2]
                    else:
                        row["numerical_snapshot"][field] = False
                    benchmark.write_json(self.row_path(jobs[side]), row)
                    self.assertFalse(self.compare(jobs)["passed"])

    def test_missing_corrupt_and_replaced_receipts_fail_explicitly(self):
        jobs = self.matrix()
        for side in (0, 1):
            with self.subTest(side=side):
                row = json.loads(self.row_path(jobs[side]).read_text())
                for corruption in ("missing", "descriptor", "worker_identity", "legacy_disk"):
                    with self.subTest(corruption=corruption):
                        altered = deepcopy(row)
                        if corruption == "missing":
                            altered.pop("comparison_input")
                        elif corruption == "descriptor":
                            altered["comparison_input"]["descriptor_sha256"] = "old"
                        elif corruption == "worker_identity":
                            altered["numerical_worker_run_id"] = "old"
                        else:
                            altered["snapshot"] = str(self.directory / "old.pt")
                        benchmark.write_json(self.row_path(jobs[side]), altered)
                        result = self.compare(jobs)
                        self.assertFalse(result["passed"])
                benchmark.write_json(self.row_path(jobs[side]), row)
        self.assertFalse(list(self.directory.rglob("*.pt")))

    def test_nonpositive_nonfinite_or_missing_timing_is_rejected(self):
        jobs = self.matrix()
        for side in (0, 1):
            original = json.loads(self.row_path(jobs[side]).read_text())
            for value in (0., -1., float("nan"), float("inf"), None):
                with self.subTest(side=side, value=value):
                    row = deepcopy(original)
                    row["timing"]["wall_ms"]["median"] = value
                    # Deliberately exercise malformed historical JSON containing NaN/Infinity.
                    self.row_path(jobs[side]).write_text(json.dumps(row))
                    result = self.compare(jobs)
                    self.assertFalse(result["passed"])
                    self.assertIn("invalid_timing", {r.get("reason") for r in result["comparisons"]})
            benchmark.write_json(self.row_path(jobs[side]), original)

    def test_missing_baseline_fingerprints_do_not_match_each_other(self):
        jobs = self.matrix()
        for job in jobs:
            self.update(job, lambda row: row.pop("rng_baseline_sha256"))
        result = self.compare(jobs)
        self.assertFalse(result["passed"])
        self.assertFalse(result["comparisons"][0]["same_initial_state"])

    def test_requested_profiles_must_exist_for_control_linear_and_eager(self):
        jobs = self.matrix()
        for side in (0, 1):
            with self.subTest(side=side):
                jobs[side]["profile_steps"] = 1
                result = self.compare(jobs)
                self.assertFalse(result["passed"])
                self.assertIn("invalid_profile_artifacts", {row.get("reason") for row in result["comparisons"]})
                trace = Path(jobs[side]["directory"]) / "trace.json"
                trace.write_text("{}\n")
                self.update(jobs[side], lambda row: row.update(profile={"steps": 1, "detail": "trace", "trace": str(trace)}))
                self.assertTrue(self.compare(jobs)["passed"])
                self.update(jobs[side], lambda row: row.update(artifact_errors=[{"phase": "profile", "reason": "export failure"}]))
                self.assertFalse(self.compare(jobs)["passed"])
                self.update(jobs[side], lambda row: row.pop("artifact_errors"))

    def main_arguments(self):
        return ["--output", str(self.directory / "suite"), "--device", "cpu", "--models", "linear",
                "--batches", "1", "--width", "4", "--hidden", "8", "--tokens", "2", "--trials", "1",
                "--iterations", "1", "--warmup", "0", "--profile-steps", "0"]

    def test_failed_job_is_checkpointed_and_later_jobs_still_run(self):
        import analysis.benchmark_do_ops as analysis

        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            manifest = json.loads((self.directory / "suite/manifest.json").read_text())
            self.assertEqual(len(manifest["jobs"]), 7)
            self.assertTrue(all(Path(job["configuration"]).is_file() and job["expected_tasks"] for job in manifest["jobs"]))
            if len(calls) == 1:
                raise RuntimeError("fixture failure")
            directory = Path(command[command.index("--output") + 1])
            benchmark.write_json(directory / "summary.json", {"run_id": f"run-{len(calls)}"})
            return 0

        with patch.object(analysis, "main", side_effect=execute), patch.object(benchmark, "compare_strategies", return_value={"passed": True, "comparisons": []}):
            self.assertEqual(benchmark.main(self.main_arguments()), 1)
        manifest = json.loads((self.directory / "suite/manifest.json").read_text())
        self.assertEqual(len(calls), 7)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual([job["status"] for job in manifest["jobs"]], ["failed"] + ["completed"] * 6)
        self.assertIn("fixture failure", manifest["jobs"][0]["error"])

    def test_interrupt_checkpoints_the_declared_unstarted_matrix_before_reraising(self):
        import analysis.benchmark_do_ops as analysis

        with patch.object(analysis, "main", side_effect=KeyboardInterrupt("fixture interrupt")), self.assertRaises(KeyboardInterrupt):
            benchmark.main(self.main_arguments())
        manifest = json.loads((self.directory / "suite/manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual([job["status"] for job in manifest["jobs"]], ["interrupted"] + ["pending"] * 6)
        self.assertEqual(json.loads((self.directory / "suite/comparison.json").read_text())["status"], "interrupted")

    def test_scheduler_cancellation_does_not_continue_independent_jobs(self):
        import analysis.benchmark_do_ops as analysis

        with patch.object(analysis, "main", side_effect=InterruptedError("scheduler cancellation")) as execute, self.assertRaises(InterruptedError):
            benchmark.main(self.main_arguments())
        self.assertEqual(execute.call_count, 1)
        manifest = json.loads((self.directory / "suite/manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual([job["status"] for job in manifest["jobs"]], ["interrupted"] + ["pending"] * 6)

    def test_planning_failure_invalidates_previous_qualification_before_checkpointing(self):
        import analysis.benchmark_do_ops as analysis

        comparison_path = self.directory / "suite/comparison.json"
        benchmark.write_json(comparison_path, {"passed": True, "suite_run_id": "old"})
        with patch.object(analysis, "make_plan", side_effect=OSError("planning failure")):
            self.assertEqual(benchmark.main(self.main_arguments()), 1)
        comparison = json.loads(comparison_path.read_text())
        self.assertFalse(comparison["passed"])
        self.assertNotEqual(comparison["suite_run_id"], "old")
        self.assertEqual(comparison["reason"], "planning_error")
        manifest = json.loads((self.directory / "suite/manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["matrix"]), 7)

    def test_planning_interrupt_checkpoints_a_new_nonpassing_suite(self):
        import analysis.benchmark_do_ops as analysis

        benchmark.write_json(self.directory / "suite/comparison.json", {"passed": True, "suite_run_id": "old"})
        with patch.object(analysis, "make_plan", side_effect=KeyboardInterrupt("planning interrupt")), self.assertRaises(KeyboardInterrupt):
            benchmark.main(self.main_arguments())
        comparison = json.loads((self.directory / "suite/comparison.json").read_text())
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["status"], "interrupted")

    def test_existing_suite_lock_prevents_checkpoint_overwrites(self):
        from analysis.scheduler import output_lock

        directory = self.directory / "suite"
        previous = {"suite_run_id": "owned", "passed": True}
        benchmark.write_json(directory / "comparison.json", previous)
        with output_lock(directory), self.assertRaisesRegex(RuntimeError, "Another coordinator"):
            benchmark.main(self.main_arguments())
        self.assertEqual(json.loads((directory / "comparison.json").read_text()), previous)

    def test_comparison_failure_is_checkpointed(self):
        import analysis.benchmark_do_ops as analysis

        def execute(command, **kwargs):
            directory = Path(command[command.index("--output") + 1])
            benchmark.write_json(directory / "summary.json", {"run_id": "fresh"})
            return 0

        with patch.object(analysis, "main", side_effect=execute), patch.object(benchmark, "compare_strategies", side_effect=RuntimeError("comparison failure")):
            self.assertEqual(benchmark.main(self.main_arguments()), 1)
        comparison = json.loads((self.directory / "suite/comparison.json").read_text())
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["reason"], "comparison_error")
        self.assertEqual(json.loads((self.directory / "suite/manifest.json").read_text())["status"], "failed")

    def test_dry_run_uses_stdlib_and_new_results_directories_each_time(self):
        command = [sys.executable, "-S", str(ROOT / "benchmarks/compiler/benchmark_bf16.py"), *self.main_arguments(), "--dry-run"]
        paths = []
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((self.directory / "suite/manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run")
            self.assertTrue(all(job["status"] == "planned" for job in manifest["jobs"]))
            paths.append({job["directory"] for job in manifest["jobs"]})
        self.assertFalse(paths[0] & paths[1])

    def test_inductor_reference_adds_only_native_torch_case_and_keeps_deepopt_matrix(self):
        self.assertEqual(benchmark.main([*self.main_arguments(), "--dry-run"]), 0)
        manifest = json.loads((self.directory / "suite/manifest.json").read_text())
        self.assertEqual(len(manifest["jobs"]), 7)
        job, = [job for job in manifest["jobs"] if job["strategy"] == "torch-inductor"]
        self.assertEqual({task["case"] for task in job["expected_tasks"]}, {"torch_bf16"})
        self.assertTrue(all(task["settings"]["runtime"]["compile"]["backend"] == "inductor"
                            and task["settings"]["runtime"]["execution"] == "compile" for task in job["expected_tasks"]))
        controls = [job for job in manifest["jobs"] if job["strategy"] not in ("eager", "torch-inductor")]
        self.assertEqual(len(controls), 5)
        self.assertTrue(all({task["case"] for task in job["expected_tasks"]} == {"do_bf16", "compression"}
                            for job in controls))
        self.assertTrue(manifest["inductor_reference"]["native_torch_ops_only"])

    def test_inductor_reference_can_be_explicitly_omitted(self):
        self.assertEqual(benchmark.main([*self.main_arguments(), "--dry-run", "--no-inductor-reference"]), 0)
        manifest = json.loads((self.directory / "suite/manifest.json").read_text())
        self.assertEqual(len(manifest["jobs"]), 6)
        self.assertFalse(manifest["inductor_reference"]["enabled"])

    def test_presets_preserve_270_original_tasks_and_18_native_controls(self):
        for preset in ("default", "cast-preserving"):
            with self.subTest(preset=preset):
                self.assertEqual(benchmark.main(["--output", str(self.directory / preset), "--dry-run",
                    "--models", "linear", "mlp", "vit-block", "--batches", "8", "32", "--trials", "3",
                    "--include-legacy", "--inductor-arithmetic-preset", preset]), 0)
                manifest = json.loads((self.directory / preset / "manifest.json").read_text())
                original = [task for job in manifest["jobs"] if job["strategy"] != "torch-inductor"
                            for task in job["expected_tasks"]]
                native = [task for job in manifest["jobs"] if job["strategy"] == "torch-inductor"
                          for task in job["expected_tasks"]]
                self.assertEqual(len(original), 270)
                self.assertEqual(len(native), 18)
                self.assertTrue(all(task["case"] == "torch_bf16" for task in native))
                self.assertTrue(all(task["settings"]["runtime"]["compile"]["options"]["emulate_precision_casts"]
                                    == (preset == "cast-preserving") for task in native))

    def inductor_matrix(self):
        jobs = self.matrix()
        jobs[1]["strategy"] = "torch-inductor"
        for job in jobs:
            task = deepcopy(job["expected_tasks"][0])
            task["case"] = "torch_bf16"
            task["settings"]["runtime"]["compile"] = {"backend": "inductor" if job is jobs[1] else "deepopt"}
            job["expected_tasks"] = [task]
            plan_path = Path(job["directory"]) / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["tasks"] = [task]
            benchmark.write_json(plan_path, plan)
            self.update(job, lambda row: row.update(task=task))
        self.update(jobs[1], lambda row: row.update(execution={"compiler_report": None}, inductor_artifacts={"fixture": True}))
        self.refresh_evidence(jobs)
        return jobs

    def test_inductor_uses_saved_native_compiler_evidence_without_deepopt_autodiff_claim(self):
        jobs = self.inductor_matrix()
        with patch("compiler.inductor_benchmark_artifacts.validate_inductor_artifacts",
                   return_value={"passed": True, "native_ops_only": True, "cuda_execution_observed": True}):
            result = self.compare(jobs)
        self.assertTrue(result["passed"])
        row, = result["comparisons"]
        self.assertFalse(row["native_autodiff_required"])
        self.assertFalse(row["native_autodiff_coverage"])
        self.assertEqual(row["runtime_reference"], "torch_inductor_native_ops")

    def test_inductor_missing_code_custom_case_or_missing_cuda_cannot_qualify(self):
        jobs = self.inductor_matrix()
        for evidence in ({"passed": False}, {"passed": True, "cuda_execution_observed": False}):
            with self.subTest(evidence=evidence), patch("compiler.inductor_benchmark_artifacts.validate_inductor_artifacts", return_value=evidence):
                self.assertFalse(self.compare(jobs)["passed"])

    def test_inductor_reference_rejects_a_custom_case_even_with_claimed_code(self):
        jobs = self.matrix()
        jobs[1]["strategy"] = "torch-inductor"
        self.refresh_evidence(jobs)
        with patch("compiler.inductor_benchmark_artifacts.validate_inductor_artifacts",
                   return_value={"passed": True, "cuda_execution_observed": True}):
            self.assertFalse(self.compare(jobs)["passed"])


if __name__ == "__main__":
    unittest.main()
