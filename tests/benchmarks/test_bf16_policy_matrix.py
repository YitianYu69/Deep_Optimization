"""Policy-specific model cells and CPU source contracts, not CUDA qualification."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

from analysis import benchmark_do_ops as analysis
from analysis.catalog import normalize_workload
from analysis.engine import Session, _run_task
from analysis.matrix import (DEFAULT_SETTINGS, bf16_policy_identity, numerical_recipe_identity, fingerprint,
                             incompatibility, merge, validate_settings)
from compiler import benchmark_bf16 as benchmark


def descriptor(name):
    return {"name": name, "version": 1}


def build_float_group_norm(*, device):
    """Real FP32 model inputs distinguish CPU AMP from BF16 storage outputs."""
    return {"model": torch.nn.GroupNorm(2, 4).to(device),
            "inputs": (torch.randn(2, 4, 17, 17, device=device, requires_grad=True),)}


class BF16PolicyMatrixTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def plan(self, policy=None):
        output = self.directory / (policy or "default")
        command = ["--device", "cpu", "--models", "linear", "mlp", "vit-block",
                   "--batches", "8", "32", "--backends", "triton", "--trials", "3",
                   "--include-legacy", "--output", str(output), "--dry-run"]
        if policy is not None:
            command.extend(("--bf16-policy", policy))
        self.assertEqual(benchmark.main(command), 0)
        return json.loads((output / "manifest.json").read_text())

    def test_cli_default_and_each_policy_preserve_the_complete_model_matrix(self):
        manifests = {policy: self.plan(policy) for policy in (None, "torch_amp", "bf16_storage")}
        self.assertEqual(manifests[None]["bf16_policy"], descriptor("torch_amp"))
        controls = {}
        for selected, manifest in manifests.items():
            selected = selected or "torch_amp"
            self.assertEqual(manifest["bf16_policy"], descriptor(selected))
            self.assertEqual(manifest["numerical_policy"]["cross_bf16_policy_comparisons"], "descriptive_only")
            self.assertEqual(manifest["inductor_reference"]["bf16_policy"], descriptor("torch_amp"))
            original, inductor = [], []
            for job in manifest["jobs"]:
                tasks = job["expected_tasks"]
                self.assertTrue(tasks)
                self.assertTrue(all(task["skip_reason"] is None for task in tasks))
                self.assertTrue(all(task["settings"]["compression"]["group_size"] == 1024 for task in tasks))
                expected = "torch_amp" if job["strategy"] == "torch-inductor" else selected
                self.assertEqual(job["bf16_policy"], descriptor(expected))
                self.assertTrue(all(task["settings"]["runtime"]["bf16_policy"] == expected for task in tasks))
                for task in tasks:
                    self.assertEqual(bf16_policy_identity(task["settings"], task["case"]),
                                     descriptor("torch_amp" if task["case"] == "torch_bf16" else selected))
                    (inductor if job["strategy"] == "torch-inductor" else original).append(task)
            self.assertEqual(len(original), 270)
            self.assertEqual(len(inductor), 18)
            self.assertEqual({task["case"] for task in inductor}, {"torch_bf16"})
            self.assertEqual(len({task["workload"]["name"] for task in original}), 6)
            self.assertTrue(all(task["settings"]["runtime"]["compile"]["backend"] == "inductor"
                                for task in inductor))
            controls[selected] = [task["settings"] for task in inductor]
        self.assertEqual(controls["torch_amp"], controls["bf16_storage"])

    def test_cli_and_runtime_reject_unknown_policy(self):
        with self.assertRaises(SystemExit):
            benchmark.main(["--bf16-policy", "fp8", "--output", str(self.directory)])
        for invalid in (None, True, "fp8", {"name": "bf16_storage", "version": 1}):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "bf16_policy"):
                validate_settings(merge(DEFAULT_SETTINGS, {"runtime": {"bf16_policy": invalid}}))

    def test_storage_policy_identity_changes_and_fp8_cells_are_rejected(self):
        amp = deepcopy(DEFAULT_SETTINGS)
        storage = merge(amp, {"runtime": {"bf16_policy": "bf16_storage"}})
        self.assertNotEqual(fingerprint(amp), fingerprint(storage))
        workload = normalize_workload({"op": "linear", "shape": [2, 65, 17]})
        for case in ("fp8", "fp8_compression"):
            self.assertIn("requires FP8 disabled", incompatibility(storage, case, workload, "cuda:0"))
        for field in ("input_dtype", "output_dtype"):
            invalid = merge(storage, {"runtime": {field: "fp8"}})
            self.assertIn("requires FP8 disabled", incompatibility(invalid, "do_bf16", workload, "cuda:0"))
        self.assertIsNone(incompatibility(storage, "torch_bf16", workload, "cpu"))

    def receipt_row(self, selected="bf16_storage", case="compression"):
        effective = "torch_amp" if case == "torch_bf16" else selected
        recipe = numerical_recipe_identity({"runtime": {"bf16_policy": selected}}, case)
        return {"task": {"case": case, "settings": {"runtime": {"bf16_policy": selected}}},
                "numerical_recipe": recipe,
                "bf16_policy": descriptor(effective), "requested_bf16_policy": descriptor(selected),
                "execution": {"bf16_policy": descriptor(effective),
                              "numerical_recipe": recipe,
                              "requested_bf16_policy": descriptor(selected),
                              "native_torch_ops_only": case == "torch_bf16"},
                "numerical_snapshot": {"bf16_policy": descriptor(effective), "numerical_recipe": recipe}}

    def test_receipts_reject_missing_or_mismatched_effective_and_requested_identity(self):
        for selected in ("torch_amp", "bf16_storage"):
            for case in ("torch_bf16", "do_bf16", "compression"):
                with self.subTest(policy=selected, case=case):
                    row = self.receipt_row(selected, case)
                    self.assertTrue(benchmark._bf16_policy_receipt(row)["passed"])
                    for path in (("bf16_policy",), ("execution", "bf16_policy"),
                                 ("numerical_snapshot", "bf16_policy"), ("requested_bf16_policy",),
                                 ("execution", "requested_bf16_policy"), ("execution", "native_torch_ops_only")):
                        broken = deepcopy(row)
                        owner = broken if len(path) == 1 else broken[path[0]]
                        owner.pop(path[-1])
                        self.assertFalse(benchmark._bf16_policy_receipt(broken)["passed"], path)
                    for invalid in (2, True, "1"):
                        row["execution"]["bf16_policy"]["version"] = invalid
                        self.assertFalse(benchmark._bf16_policy_receipt(row)["passed"])

    def test_old_storage_primitive_receipt_does_not_qualify_current_recipe(self):
        row = self.receipt_row()
        row["execution"]["numerical_recipe"] = {**row["execution"]["numerical_recipe"], "primitive_recipe_version": 1}
        self.assertFalse(benchmark._bf16_policy_receipt(row)["passed"])

    def comparison_jobs(self, eager_policy, candidate_policy):
        from analysis.numerical_ipc import snapshot_descriptor
        from compiler.model_numerics import ModelNumericalComparisons

        comparisons = ModelNumericalComparisons(receipt_timeout=0)
        jobs = []
        for strategy, selected in (("eager", eager_policy), ("triton-native-control", candidate_policy)):
            directory = self.directory / strategy
            task = {"id": strategy, "case": "do_bf16", "settings": {"runtime": {
                "bf16_policy": selected, "execution": "eager" if strategy == "eager" else "compile"}},
                    "workload": {"name": "linear-b8", "factory_kwargs": {"kind": "linear"}}}
            job = {"trial": 0, "strategy": strategy, "status": "completed", "result": 0,
                   "directory": str(directory), "expected_tasks": [task], "plan_code_sha256": "code",
                   "analysis_run_id": strategy}
            identity = {**comparisons.start_job(job, "policy-suite"), "task": task, "worker_run_id": strategy}
            metadata = {"full_tensor_finite": True, "original_shapes": {"output.0": [4]},
                        "sampling": "all", "max_sampled_elements_per_float_tensor": 65536,
                        "bf16_policy": descriptor(selected), "numerical_recipe": numerical_recipe_identity(task["settings"], task["case"])}
            tensors = {"output.0": torch.ones(4)}
            receipt = snapshot_descriptor(identity, tensors, metadata)
            comparisons.accept({"identity": identity, "tensors": tensors, "metadata": metadata,
                                "descriptor": receipt})
            comparisons.end_job()
            row = {"status": "ok", "task": task, "run_id": strategy, "initial_state_sha256": "state",
                   "rng_baseline_sha256": "rng", "shared_optimizer_baseline_sha256": "optimizer",
                   "environment": {"deepopt_source_sha256": "build"}, "timing": {"wall_ms": {"median": 1.}},
                   "measurement_policy": {"numerical_comparison": True, "numerical_snapshot_mode": "memory"},
                   "finite_after_timing": True, "numerical_snapshot": metadata,
                   "bf16_policy": descriptor(selected), "requested_bf16_policy": descriptor(selected),
                   "numerical_recipe": metadata["numerical_recipe"],
                   "execution": {"bf16_policy": descriptor(selected), "requested_bf16_policy": descriptor(selected),
                       "numerical_recipe": metadata["numerical_recipe"],
                       "native_torch_ops_only": False, "compiler_report": {
                       "static_regions": {"bf16_linear_programs": 1}, "native_autodiff_operations": 1,
                       "runtime_fallbacks": 0, "graphs": [{"eager_autograd_operations": 0}]}},
                   "comparison_input": receipt, "numerical_worker_run_id": strategy}
            benchmark.write_json(directory / "cases" / task["id"] / "result.json", row)
            benchmark.write_json(directory / "plan.json", {"tasks": [task], "code_sha256": "code", "options": {}})
            benchmark.write_json(directory / "summary.json", {"run_id": strategy})
            jobs.append(job)
        return jobs, comparisons

    def test_compiler_eager_policy_match_is_mandatory_even_when_tensors_match(self):
        for expected, actual in (("torch_amp", "torch_amp"), ("bf16_storage", "bf16_storage"),
                                 ("torch_amp", "bf16_storage")):
            with self.subTest(eager=expected, compiler=actual):
                jobs, numerical = self.comparison_jobs(expected, actual)
                result = benchmark.compare_strategies(jobs, numerical_comparisons=numerical)
                self.assertEqual(result["passed"], expected == actual)
                candidate = next(row for row in result["comparisons"] if row["strategy"] != "eager")
                self.assertTrue(candidate["numerical"]["passed"])
                self.assertEqual(candidate["same_bf16_policy"], expected == actual)

    def session_task(self, selected, case, *, compiled=False):
        settings = merge(DEFAULT_SETTINGS, {"fp8": {"backend": "reference"},
            "compression": {"group_size": 1024}, "runtime": {"bf16_policy": selected, "binding": "prepared",
                "execution": "compile" if compiled else "eager", "optimizer": "sgd",
                "compile": {"backend": "deepopt", "fullgraph": True, "dynamic": False,
                            "options": {"use_native_autodiff": True, "require_full_aot": True}}}})
        options = analysis.make_plan(analysis.parser().parse_args(["--device", "cpu", "--small",
            "--warmup", "0", "--iterations", "1", "--repeats", "1", "--phase-steps", "0",
            "--no-memory", "--numerical-snapshot-mode", "memory"]))["options"]
        task = {"id": fingerprint(settings), "settings": settings, "case": case, "variant_id": "cpu-source",
                "skip_reason": None, "workload": normalize_workload({"op": "model",
                    "shape": [2, 4, 17, 17], "factory":
                        "tests.benchmarks.test_bf16_policy_matrix:build_float_group_norm"})}
        return task, options

    def test_source_cpu_real_eager_and_native_scopes_keep_bf16_and_fp8_independent(self):
        from analysis.measurement import errors, snapshot

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.compiler.reset)
        for selected in ("torch_amp", "bf16_storage"):
            for case in ("do_bf16", "compression"):
                values = []
                for compiled in (False, True):
                    with self.subTest(policy=selected, case=case, compiled=compiled):
                        task, options = self.session_task(selected, case, compiled=compiled)
                        session = Session(task, options)
                        try:
                            self.assertFalse(session.fp8_config.enabled)
                            self.assertEqual(session.plan["bf16_policy"], descriptor(selected))
                            with session.context():
                                from deepopt._bf16_policy import current_bf16_policy

                                self.assertEqual(current_bf16_policy("cpu").as_dict(), descriptor(selected))
                            output, _ = session.forward()
                            self.assertEqual(output.dtype, torch.bfloat16 if selected == "bf16_storage" else torch.float32)
                            values.append(snapshot(session))
                            self.assertEqual(session.execution_info()["bf16_policy"], descriptor(selected))
                        finally:
                            session.close()
                self.assertTrue(errors(values[1], values[0], rtol=.02, atol=.002,
                                       max_relative_l2=.1)["passed"])

    def test_source_cpu_raw_torch_is_native_amp_inside_storage_scope(self):
        from deepopt import autocast
        from deepopt._bf16_policy import current_bf16_policy

        task, options = self.session_task("bf16_storage", "torch_bf16")
        session = Session(task, options)
        try:
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage"):
                with session.context():
                    self.assertEqual(current_bf16_policy("cpu").name, "torch_amp")
                output, _ = session.forward()
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(current_bf16_policy("cpu").name, "bf16_storage")
            self.assertEqual(session.execution_info()["bf16_policy"], descriptor("torch_amp"))
            self.assertTrue(session.execution_info()["native_torch_ops_only"])
        finally:
            session.close()

    def test_source_cpu_memory_row_binds_policy_to_live_snapshot_metadata(self):
        from analysis.numerical_ipc import snapshot_descriptor

        task, options = self.session_task("bf16_storage", "compression")
        options["_numerical_worker_run_id"] = "policy-worker"
        receipts = []

        def receive(identity, tensors, metadata):
            receipt = snapshot_descriptor(identity, tensors, metadata)
            receipts.append(receipt)
            return receipt

        with patch.dict(os.environ, {"DEEPOPT_NUMERICAL_CONTEXT": json.dumps({"suite_run_id": "policy-suite"})}), \
                patch("analysis.numerical_ipc.send_snapshot", side_effect=receive), \
                patch("torch.save", side_effect=AssertionError("tensor files forbidden")), \
                patch("torch.load", side_effect=AssertionError("tensor files forbidden")):
            row = _run_task(task, options, self.directory)
        self.assertEqual(len(receipts), 1)
        row["task"] = task
        self.assertTrue(benchmark._bf16_policy_receipt(row)["passed"])
        self.assertEqual(receipts[0]["metadata"]["bf16_policy"], descriptor("bf16_storage"))
        self.assertIsNone(row["snapshot"])
        self.assertIsNone(row["snapshot_file"])
        self.assertFalse(list(self.directory.rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
