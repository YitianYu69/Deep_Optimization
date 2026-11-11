"""Behavioral contracts for reproducible factorial comparisons and operator coverage."""

from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/analysis/benchmark_do_ops.py"
SPEC = importlib.util.spec_from_file_location("do_analysis_cli", SCRIPT)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
from analysis.catalog import default_workload, normalize_workload
from analysis.engine import Session
from analysis.matrix import DEFAULT_SETTINGS, fingerprint, incompatibility, merge, variants
from analysis.measurement import errors, snapshot
from analysis.reporting import summarize


class AnalysisMatrixTests(unittest.TestCase):
    def test_help_inventory_and_dry_run_do_not_import_torch(self):
        original = __import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "deepopt", "triton"):
                raise AssertionError("Planning must not import " + name)
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch("builtins.__import__", new=guarded):
            for args in (["--list-ops"], ["--dry-run", "--ops", "linear", "--output", directory]):
                with patch("sys.stdout", new=io.StringIO()):
                    self.assertEqual(cli.main(args), 0)
            with patch("sys.stdout", new=io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(["--help"])
            self.assertEqual(error.exception.code, 0)

    def test_public_do_functions_are_all_accounted_for(self):
        import ast
        module = ast.parse((ROOT / "src/deepopt/opt/nn/functional.py").read_text())
        public = {node.name.removeprefix("do_") for node in module.body
                  if isinstance(node, ast.FunctionDef) and node.name.startswith("do_")}
        self.assertFalse(public - set(cli.OPS))

    def test_declared_cartesian_grid_is_exhaustive_and_sensitivity_is_deduplicated(self):
        grid = {"compression.bits": [2, 4], "fp8.backward_precision": ["native", "auto"]}
        one = list(variants(DEFAULT_SETTINGS, grid, "one-at-a-time"))
        crossed = list(variants(DEFAULT_SETTINGS, grid, "cartesian"))
        self.assertEqual((len(one), len(crossed)), (3, 4))
        self.assertEqual(len({key for key, _ in crossed}), 4)
        self.assertEqual(DEFAULT_SETTINGS["compression"]["bits"], 2)

    def test_invalid_or_unknown_settings_fail_before_launch(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            merge(DEFAULT_SETTINGS, {"fp8": {"backward_precison": "auto"}})
        with self.assertRaisesRegex(ValueError, "bits"):
            list(variants(DEFAULT_SETTINGS, {"compression.bits": [3]}, "cartesian"))
        args = cli.parser().parse_args(["--preset", "extended", "--sweep", "cartesian", "--max-experiments", "10"])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            cli.make_plan(args)

    def test_incompatibilities_do_not_silently_fallback(self):
        settings = deepcopy(DEFAULT_SETTINGS)
        settings["fp8"].update(scaling_recipe="delayed", block_shapes={"activation": [1, 32], "weight": None, "grad": None})
        op = default_workload("linear", small=True)
        self.assertIn("tensorwise", incompatibility(settings, "fp8_compression", op, "cuda:0"))
        self.assertIsNone(incompatibility(settings, "torch_bf16", op, "cuda:0"))

    def test_numerical_nonfinite_and_exact_integer_failures_are_visible(self):
        self.assertFalse(errors({"x": torch.tensor([float("nan")])}, {"x": torch.ones(1)}, .1, .1)["finite"])
        exact = errors({"index": torch.tensor([100000001])}, {"index": torch.tensor([100000000])}, .1, .1)
        self.assertFalse(exact["passed"])
        self.assertTrue(exact["tensors"]["index"]["exact"])

    def test_report_recovers_known_19_second_interaction_and_gates_bad_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for case, seconds in (("torch_bf16", 21), ("do_bf16", 21), ("compression", 27), ("fp8", 22), ("fp8_compression", 47)):
                path = Path(directory) / f"{case}.pt"
                torch.save({"output.0": torch.tensor([2.0 if case == "fp8_compression" else 1.0])}, path)
                rows.append({"status": "ok", "task": {"case": case, "variant_id": "same", "workload": default_workload("linear", True)},
                             "snapshot": str(path), "initial_state_sha256": "same",
                             "timing": {"wall_ms": {"median": seconds * 1000, "stdev": 0, "samples": [seconds * 1000]}, "cuda_stream_ms": None}})
            report = summarize(rows, {"default": {"rtol": 0, "atol": 0}}, {"bf16": 21, "compression": 27, "fp8": 22, "combined": 47})
            self.assertEqual(report["interactions"][0]["interaction_wall_ms"], 19000)
            self.assertEqual(report["observed_epoch_context"]["interaction_seconds"], 19)
            last = next(row for row in report["comparisons"] if row["case"] == "fp8_compression")
            self.assertIsNone(last["qualified_wall_speedup_vs_torch"])
            self.assertIsNotNone(last["raw_wall_speedup_vs_torch"])


class AnalysisExecutionTests(unittest.TestCase):
    def task(self, op="linear", case="fp8_compression", update=None):
        settings = deepcopy(DEFAULT_SETTINGS)
        settings["fp8"]["backend"] = "reference"
        if update:
            settings = merge(settings, update)
        args = cli.parser().parse_args(["--device", "cpu", "--small", "--warmup", "0", "--iterations", "1", "--repeats", "1", "--phase-steps", "0"])
        options = cli.make_plan(args)["options"]
        task = {"id": fingerprint(settings), "settings": settings, "case": case,
                "workload": normalize_workload(default_workload(op, True)), "variant_id": "test", "skip_reason": None}
        return task, options

    def test_snapshot_restores_optimizer_and_model_and_emits_real_gradients(self):
        task, options = self.task(update={"runtime": {"optimizer": "sgd"}})
        session = Session(task, options)
        try:
            before = {name: p.detach().clone() for name, p in session.model.named_parameters()}
            first, second = snapshot(session), snapshot(session)
            for name, parameter in session.model.named_parameters():
                self.assertTrue(torch.equal(parameter, before[name]))
                self.assertGreater(first[f"parameter_update.{name}"].abs().sum().item(), 0)
                self.assertTrue(torch.equal(first[f"parameter_grad.{name}"], second[f"parameter_grad.{name}"]))
            self.assertTrue(session.inputs[0].grad is not None)
        finally:
            session.close()

    def test_functional_family_ablation_really_disables_packing(self):
        for families, expected in (([], False), (["linear"], True)):
            task, options = self.task(update={"runtime": {"compression_families": families}})
            session = Session(task, options)
            try:
                self.assertEqual(session.scope_compression.enabled, expected)
                output, _ = session.forward()
                saved = output.grad_fn.saved_tensors
                self.assertEqual(any(t is not None and t.dtype == torch.int32 for t in saved), expected)
            finally:
                session.close()

    def test_delayed_scaling_owns_state_for_parameterless_functional_activation(self):
        task, options = self.task(op="gelu", update={"fp8": {"scaling_recipe": "delayed"}, "runtime": {"output_dtype": "fp8"}})
        session = Session(task, options)
        try:
            first, second = snapshot(session), snapshot(session)
            self.assertIsNotNone(session.owner_state)
            self.assertTrue(torch.equal(first["output.0"], second["output.0"]))
        finally:
            session.close()

    def test_prepared_native_plan_ablation_changes_the_owned_policy(self):
        task, options = self.task(op="layer_norm", case="compression", update={
            "runtime": {"binding": "prepared", "compression_families": []}})
        session = Session(task, options)
        try:
            self.assertFalse(session.model.plan["activation_compression"]["enabled"])
            session.step()
            self.assertFalse(session.model.execution_info["compression"])
        finally:
            session.close()

    def test_disabled_fp8_controls_survive_incompatible_fp8_grid_values(self):
        for case in ("torch_bf16", "compression"):
            task, options = self.task(case=case, update={"fp8": {"scaling_recipe": "delayed",
                "backward_backend": "native_fast", "block_shapes": {"activation": [1, 32]}}})
            session = Session(task, options)
            try:
                session.step()
                self.assertFalse(session.fp8_config.enabled)
                self.assertEqual(session.fp8_config.scaling_recipe, "current")
            finally:
                session.close()

    def test_factory_fetch_rewinds_targets_and_releases_the_initial_input_reference(self):
        task, options = self.task(case="torch_bf16")
        task["workload"] = normalize_workload({"op": "model", "name": "loader", "shape": [2, 8, 16],
            "factory": "analysis.example_model:build", "factory_kwargs": {"batch_size": 2, "sequence_length": 8,
            "width": 16, "heads": 2, "layers": 1, "input_mode": "pinned"}})
        session = Session(task, options)
        try:
            session.step()
            self.assertIs(session.inputs[0], session.raw_inputs[0])
            first, second = snapshot(session), snapshot(session)
            self.assertTrue(torch.equal(first["loss"], second["loss"]))
        finally:
            session.close()

    def test_full_cli_reports_all_five_cases_indices_and_worker_profiling(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(SCRIPT), "--device", "cpu", "--backend", "reference", "--small",
                       "--ops", "linear", "max_pool2d_with_indices", "--warmup", "0", "--iterations", "1",
                       "--repeats", "1", "--phase-steps", "1", "--profile-steps", "1", "--output", directory]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(report["counts"], {"ok": 10, "skipped": 0, "error": 0})
            self.assertEqual(len(report["interactions"]), 2)
            self.assertTrue(all(row["same_initial_state"] for row in report["comparisons"]))
            self.assertTrue(all(Path(row["profile"]["trace"]).exists() for row in report["results"]))
            self.assertTrue((Path(directory) / "results.csv").is_file())
            indexed = next(row for row in report["results"] if row["task"]["case"] == "compression" and row["task"]["workload"]["op"] == "max_pool2d_with_indices")
            self.assertTrue(indexed["correctness_vs_torch_bf16"]["tensors"]["output.1"]["exact"])


if __name__ == "__main__":
    unittest.main()
