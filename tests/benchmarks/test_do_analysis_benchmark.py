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
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/analysis/benchmark_do_ops.py"
SPEC = importlib.util.spec_from_file_location("do_analysis_cli", SCRIPT)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
from analysis.catalog import default_workload, normalize_workload, family, inventory
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

    def test_pad_mul_plans_have_real_references_and_positive_group1024_shapes(self):
        args = cli.parser().parse_args(["--device", "cpu", "--backend", "reference", "--small", "--ops", "pad", "mul"])
        plan = cli.make_plan(args)
        self.assertEqual(len(plan["tasks"]), 10)
        references = {row["op"]: row["torch"] for row in inventory()}
        self.assertEqual(references["mul"], "torch.mul")
        self.assertEqual(references["pad"], "torch.nn.functional.pad")
        self.assertEqual((family("mul"), family("pad")), ("binary", "padding"))
        for task in plan["tasks"]:
            self.assertIsNone(task["skip_reason"])
            self.assertEqual(task["settings"]["compression"]["group_size"], 1024)
            self.assertGreater(__import__("math").prod(task["workload"]["shape"]), 1024)

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

    def test_gradient_norm_gate_rejects_errors_hidden_by_large_absolute_tolerance(self):
        reference = {"parameter_grad.weight": torch.tensor([1e-5, 2e-5])}
        actual = {"parameter_grad.weight": reference["parameter_grad.weight"] * 1.234}
        result = errors(actual, reference, .05, .01,
                        gradient_policy={"rtol": .05, "atol": .01, "max_relative_l2": .1, "min_cosine": .99})
        self.assertTrue(result["tensors"]["parameter_grad.weight"]["allclose_passed"])
        self.assertFalse(result["passed"])
        self.assertGreater(result["tensors"]["parameter_grad.weight"]["relative_l2"], .23)

    def test_report_recovers_synthetic_interaction_and_gates_bad_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for case, seconds in (("torch_bf16", 10), ("do_bf16", 10), ("compression", 12),
                                  ("fp8", 13), ("fp8_compression", 20)):
                path = Path(directory) / f"{case}.pt"
                torch.save({"output.0": torch.tensor([2.0 if case == "fp8_compression" else 1.0])}, path)
                rows.append({"status": "ok", "task": {"case": case, "variant_id": "same", "workload": default_workload("linear", True)},
                             "snapshot": str(path), "initial_state_sha256": "same",
                             "timing": {"wall_ms": {"median": seconds * 1000, "stdev": 0, "samples": [seconds * 1000]}, "cuda_stream_ms": None}})
            report = summarize(rows, {"default": {"rtol": 0, "atol": 0}},
                               {"bf16": 10, "compression": 12, "fp8": 13, "combined": 20})
            self.assertEqual(report["interactions"][0]["interaction_wall_ms"], 5000)
            self.assertEqual(report["observed_epoch_context"]["interaction_seconds"], 5)
            last = next(row for row in report["comparisons"] if row["case"] == "fp8_compression")
            self.assertIsNone(last["qualified_wall_speedup_vs_torch"])
            self.assertIsNotNone(last["raw_wall_speedup_vs_torch"])


class AnalysisExecutionTests(unittest.TestCase):
    def test_reused_scope_avoids_constructor_and_random_probe_is_repeatable(self):
        task, options = self.task(case="fp8_compression", update={"runtime": {"scope_lifetime": "reused", "numerical_probe": "random_upstream"}})
        session = Session(task, options)
        try:
            first = snapshot(session)
            with patch.object(session.deepopt, "autocast", side_effect=AssertionError("constructed again")):
                second = snapshot(session)
                session.step()
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                if first[key] is not None:
                    self.assertTrue(torch.equal(first[key], second[key]), key)
            self.assertEqual(session.snapshot_metadata["numerical_probe"], "random_upstream")
        finally:
            session.close()

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

    def test_mul_workloads_execute_matching_codec_backward_for_all_bindings(self):
        from deepopt.opt.nn import functional as doF
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        for binding in ("functional", "prepared", "intercept"):
            for case in ("torch_bf16", "do_bf16", "compression", "fp8", "fp8_compression"):
                with self.subTest(binding=binding, case=case):
                    task, options = self.task("mul", case, {"runtime": {"binding": binding}})
                    task["workload"].update(layout="strided", kwargs={"other_shape": [1, 1031]})
                    session = Session(task, options)
                    try:
                        self.assertEqual(session.compression.group_size, 1024)
                        left, right = session.inputs
                        self.assertGreater(min(left.numel(), right.numel()), 1024)
                        self.assertFalse(left.is_contiguous())
                        with patch.object(doF, "do_mul", wraps=doF.do_mul) as called:
                            output, _ = session.forward()
                        self.assertEqual(called.call_count, int(case != "torch_bf16"))
                        torch.testing.assert_close(output, left * right, rtol=0, atol=0)
                        upstream = torch.randn_like(output)

                        def coefficient(value):
                            if case not in ("compression", "fp8_compression"):
                                return value.detach()
                            packed = quantize_pack_activation(value.detach(), 1024, bits=2)
                            return dequantize_unpack_activation(packed, dtype=value.dtype, row_alignment=1)

                        expected_left = (upstream * coefficient(right)).sum_to_size(left.shape)
                        expected_right = (upstream * coefficient(left)).sum_to_size(right.shape)
                        if case in ("compression", "fp8_compression"):
                            self.assertEqual(output.grad_fn.descriptors[0][0], "packed")
                            self.assertEqual(output.grad_fn.descriptors[0][5], 1024)
                            self.assertGreater(session.info["saved_state_bytes"], 0)
                        output.backward(upstream)
                        torch.testing.assert_close(left.grad, expected_left, rtol=0, atol=0)
                        torch.testing.assert_close(right.grad, expected_right, rtol=0, atol=0)
                        # Execute the actual benchmark step as well as its oracle.
                        session.clear()
                        session.step()
                        self.assertTrue(all(bool(torch.isfinite(value.grad).all()) for value in session.inputs))
                    finally:
                        session.close()

    def test_pad_workloads_execute_exact_geometry_backward_with_compression_enabled(self):
        from deepopt.opt.nn import functional as doF
        from torch.nn import functional as F

        for mode in ("constant", "reflect", "replicate", "circular"):
            for case in ("torch_bf16", "do_bf16", "compression", "fp8", "fp8_compression"):
                with self.subTest(mode=mode, case=case):
                    task, options = self.task("pad", case, {"runtime": {"binding": "prepared"}})
                    settings = {"pad": [-1, 2], "mode": mode}
                    if mode == "constant":
                        settings["value"] = .25
                    task["workload"].update(layout="strided", kwargs=settings)
                    session = Session(task, options)
                    try:
                        self.assertEqual(session.compression.group_size, 1024)
                        value = session.inputs[0]
                        self.assertGreater(value.numel(), 1024)
                        reference = value.detach().clone().requires_grad_()
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            expected = F.pad(reference, **settings)
                        with patch.object(doF, "do_pad", wraps=doF.do_pad) as called:
                            output, _ = session.forward()
                        self.assertEqual(called.call_count, int(case != "torch_bf16"))
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        upstream = torch.randn_like(output)
                        expected.backward(upstream)
                        # Padding's exact adjoint stores geometry, no value codec.
                        with session.context():
                            if case == "torch_bf16":
                                actual = session.call(value)
                            else:
                                with torch.autograd.graph.saved_tensors_hooks(
                                        lambda tensor: self.fail("Padding retained a tensor"), lambda tensor: tensor):
                                    actual = session.call(value)
                        actual.backward(upstream)
                        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
                        session.clear()
                        session.step()
                        self.assertTrue(bool(torch.isfinite(value.grad).all()))
                    finally:
                        session.close()

    def test_binary_family_ablation_disables_actual_multiplication_packing(self):
        for families, enabled in (([], False), (["binary"], True)):
            task, options = self.task("mul", "compression", {"runtime": {"compression_families": families}})
            session = Session(task, options)
            try:
                output, _ = session.forward()
                self.assertEqual(any(value.dtype == torch.int32 for value in output.grad_fn.saved_tensors), enabled)
                session.backward(output.float().square().mean())
            finally:
                session.close()

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

    def test_model_factory_post_step_ema_rewinds_with_diagnostics(self):
        task, options = self.task(case="fp8_compression", update={"runtime": {"optimizer": "sgd"}})
        task["workload"] = normalize_workload({"op": "model", "shape": [2, 3, 4, 4],
            "factory": "analysis.example_model:build_timm_vit", "factory_kwargs": {"batch_size": 2,
                "image_size": 4, "classes": 3, "ema_decay": 0.95}})
        fake_timm = SimpleNamespace(create_model=lambda *a, **k: torch.nn.Sequential(torch.nn.Flatten(),
            torch.nn.Linear(48, 16), torch.nn.GELU(), torch.nn.Linear(16, 3)))
        with patch.dict(sys.modules, {"timm": fake_timm}):
            session = Session(task, options)
        try:
            self.assertIsNotNone(session.after_step)
            first, second = snapshot(session), snapshot(session)
            for name in first:
                if first[name] is not None:
                    self.assertTrue(torch.equal(first[name], second[name]), name)
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
