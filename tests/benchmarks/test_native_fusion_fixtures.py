"""Isolated fixtures keep numerical, paired-control and GPU claims separate."""

from dataclasses import asdict
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
from math import prod
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location("isolated_fusion_fixtures", BENCHMARKS / "compiler/isolated_fusion.py")
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)


class NativeFusionFixturePlanTests(unittest.TestCase):
    def test_policy_and_experimental_native_group_are_explicit_paired_controls(self):
        args = fixtures.parser().parse_args(['--fixtures', 'groupnorm-activation', '--bf16-policy', 'bf16_storage',
            '--qualify-native-group-fusion', '--output', '/tmp/fusion-storage-plan'])
        plan = fixtures.make_plan(args)
        for job in (*plan['jobs'], *plan['diagnostic_jobs']):
            self.assertEqual(job['fixture']['bf16_policy'], 'bf16_storage')
            self.assertEqual(job['fixture']['group_size'], 1024)
            self.assertTrue(job['strategy']['options']['qualify_native_group_fusion'])
        default = fixtures.Fixture('groupnorm-activation')
        self.assertEqual(default.bf16_policy, 'torch_amp')
        self.assertFalse(any('qualify_native_group_fusion' in s['options'] for s in fixtures.strategies('triton', default)))

    def test_trial_workers_preserve_randomized_order_and_fresh_trial_boundaries(self):
        args = fixtures.parser().parse_args([
            "--fixtures", "gemm-activation", "--phases", "both", "--trials", "3",
            "--worker-isolation", "trial", "--output", "/tmp/fusion-trial-plan"])
        plan = fixtures.make_plan(args)
        groups = fixtures.worker_groups(plan)
        self.assertFalse(plan["fresh_process_per_strategy"])
        self.assertTrue(plan["fresh_process_per_trial"])
        self.assertEqual(len(groups), 3)
        self.assertEqual([job for group in groups for job in group], plan["jobs"])
        for trial, group in enumerate(groups):
            self.assertEqual({job["trial"] for job in group}, {trial})
            self.assertEqual(len(group), 3)
            self.assertTrue(all(job["worker_isolation"] == "trial" for job in group))
        self.assertEqual(plan["jobs"][0]["measurement"]["phase_iterations"], 3)
        self.assertEqual(plan["jobs"][0]["measurement"]["phase_repeats"], 1)
        args.worker_isolation = "strategy"
        strategy_plan = fixtures.make_plan(args)
        self.assertTrue(strategy_plan["fresh_process_per_strategy"])
        self.assertEqual(len(fixtures.worker_groups(strategy_plan)), 9)
        self.assertEqual(len(fixtures.comparison_groups(strategy_plan)), 3)
        self.assertEqual([job for group in fixtures.comparison_groups(strategy_plan) for job in group],
                         strategy_plan["jobs"])

    def test_group_failure_is_persisted_and_other_sequential_controls_still_run(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            entries = [{"job": {"id": name, "device": "cpu", "worker_isolation": "trial"},
                        "result_path": str(directory / name / "result.json")}
                       for name in ("first", "second")]
            for entry in entries:
                Path(entry["result_path"]).parent.mkdir()
            with patch.object(fixtures, "run_worker", side_effect=[RuntimeError("compile failed"),
                                                                   {"status": "ok"}]) as execute, \
                    patch.object(fixtures, "_cleanup_strategy") as cleanup:
                rows = fixtures.execute_worker_jobs(entries, progress_path=directory / "worker-phase.json")
            self.assertEqual([row["status"] for row in rows], ["error", "ok"])
            self.assertEqual([call.args[0]["id"] for call in execute.call_args_list], ["first", "second"])
            self.assertEqual(cleanup.call_count, 2)
            self.assertEqual({row["worker_process_id"] for row in rows}, {os.getpid()})
            self.assertEqual({row["worker_isolation"] for row in rows}, {"trial"})
            first = json.loads(Path(entries[0]["result_path"]).read_text())
            self.assertIn("compile failed", first["reason"])
            self.assertTrue(Path(entries[1]["result_path"]).exists())

    def test_cleanup_failure_blocks_remaining_controls_in_the_group(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            entries = [{"job": {"id": name, "device": "cuda:0", "worker_isolation": "trial"},
                        "result_path": str(directory / name / "result.json")}
                       for name in ("first", "second")]
            for entry in entries:
                Path(entry["result_path"]).parent.mkdir()
            with patch.object(fixtures, "run_worker", return_value={"status": "ok"}) as execute, \
                    patch.object(fixtures, "_cleanup_strategy", side_effect=RuntimeError("CUDA cleanup failed")) as cleanup:
                rows = fixtures.execute_worker_jobs(entries)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(cleanup.call_count, 1)
            self.assertEqual([row["status"] for row in rows], ["error", "error"])
            self.assertEqual(rows[1]["reason"], "not_run_after_strategy_cleanup_failure")
            self.assertEqual(rows[1]["blocking_job_id"], "first")

    def test_phase_diagnostics_have_a_separate_small_budget_or_can_be_disabled(self):
        model = torch.nn.Linear(3, 2)
        x = torch.randn(2, 3, requires_grad=True)
        gradient = torch.randn(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        settings = {"iterations": 50, "repeats": 5}
        calls = []
        handle = model.register_forward_hook(lambda *args: calls.append(True))
        try:
            result = fixtures._phase_timings(model, x, gradient, optimizer, nullcontext, settings,
                                            torch.device("cpu"))
            self.assertEqual(len(calls), 3)
            self.assertEqual(result["forward"]["iterations"], 3)
            self.assertEqual(result["forward"]["repeats"], 1)
            calls.clear()
            result = fixtures._phase_timings(model, x, gradient, optimizer, nullcontext,
                                            {**settings, "phase_iterations": 0}, torch.device("cpu"))
            self.assertEqual(calls, [])
            self.assertEqual(result["forward"]["mode"], "disabled")
            self.assertIsNone(result["backward"]["wall_ms"])
        finally:
            handle.remove()

    def test_worker_phase_progress_reports_compile_work_and_lifecycle_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            progress = fixtures.WorkerProgress({"id": "fixture", "strategy": {"name": "fused"}},
                                              directory)
            progress.start("cold_compile_and_first_update")
            active = json.loads((directory / "worker-phase.json").read_text())
            self.assertEqual(active["phase"], "cold_compile_and_first_update")
            self.assertEqual(active["status"], "running")
            self.assertIn("timestamp_utc", active)
            timing = progress.finish()
            self.assertGreaterEqual(timing["total"], timing["cold_compile_and_first_update"])
            self.assertEqual(json.loads((directory / "worker-phase.json").read_text())["status"], "completed")

    def test_dry_run_needs_only_helpers_and_never_imports_the_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("compiler", "analysis"):
                shutil.copytree(BENCHMARKS / directory, root / "benchmarks" / directory,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            blocker = root / "import-blocker"
            blocker.mkdir()
            (blocker / "sitecustomize.py").write_text(
                "import sys\n"
                "class RejectWorkload:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in ('torch', 'deepopt'):\n"
                "            raise RuntimeError('dry run imported ' + fullname)\n"
                "sys.meta_path.insert(0, RejectWorkload())\n")
            env = {**os.environ, "PYTHONPATH": str(blocker)}
            output = root / "output"
            command = [sys.executable, str(root / "benchmarks/compiler/isolated_fusion.py"),
                       "--fixtures", "gemm-activation", "--phases", "both", "--dry-run",
                       "--output", str(output)]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((output / "fusion-plan.json").read_text())
            self.assertEqual(len(plan["jobs"]), 9)
            self.assertNotIn("installed_deepopt", plan)

    def test_plan_contains_independent_phase_and_same_backend_pairs(self):
        args = fixtures.parser().parse_args([
            "--fixtures", "gemm-activation", "gemm-residual", "layernorm-activation", "groupnorm-activation",
            "packed-multiply", "shared-packs", "--compression", "on", "--bits", "1", "2", "4", "8",
            "--output", "/tmp/native-fusion-test-plan", "--trials", "3"])
        plan = fixtures.make_plan(args)
        json.dumps(plan, allow_nan=False)
        self.assertTrue(plan["fresh_process_per_strategy"])
        self.assertEqual(len(plan["jobs"]), 17 * 3 * 4 * 3)
        self.assertEqual({job["fixture"]["phase"] for job in plan["jobs"]}, set(fixtures.PHASES))
        self.assertEqual({job["fixture"]["bits"] for job in plan["jobs"]}, {1, 2, 4, 8})
        for fixture in (fixtures.Fixture("gemm-activation", phase=phase) for phase in fixtures.PHASES):
            unfused, fused, library = fixtures.strategies("triton", fixture)
            ao, bo = dict(unfused["options"]), dict(fused["options"])
            self.assertFalse(ao.pop("fuse_deepopt_activations"))
            self.assertTrue(bo.pop("fuse_deepopt_activations"))
            self.assertEqual(ao, bo)
            self.assertEqual(library["options"]["bf16_gemm_backend"], "native")
            self.assertEqual(bo["fusion_phases"], fixture.phase)

    def test_residual_plan_has_same_generated_pair_library_control_and_explicit_phase(self):
        args = fixtures.parser().parse_args([
            "--fixtures", "gemm-residual", "--phases", "forward", "both",
            "--output", "/tmp/native-residual-test-plan", "--trials", "3"])
        plan = fixtures.make_plan(args)
        self.assertEqual(len(plan["jobs"]), 18)
        self.assertEqual({job["fixture"]["phase"] for job in plan["jobs"]}, {"forward", "both"})
        for phase in fixtures.PHASES:
            routes = fixtures.strategies("triton", fixtures.Fixture("gemm-residual", phase=phase))
            unfused, fused, library = (dict(row["options"]) for row in routes)
            self.assertFalse(unfused.pop("fuse_deepopt_activations"))
            self.assertTrue(fused.pop("fuse_deepopt_activations"))
            self.assertEqual(unfused, fused)
            self.assertEqual(fused["fusion_phases"], phase)
            self.assertEqual(library["bf16_gemm_backend"], "native")

    def test_residual_fixture_accumulates_the_aliased_input_gradient(self):
        fixture = fixtures.Fixture("gemm-residual", batch=2, tokens=3, width=7, dtype="fp32")
        model, x = fixtures.build_fixture(fixture, torch.device("cpu"), 39)
        gradient = torch.randn_like(x)
        model(x).backward(gradient)
        expected = gradient @ model.projection.weight.detach() + gradient
        torch.testing.assert_close(x.grad, expected)
        torch.testing.assert_close(model.projection.weight.grad,
                                   gradient.reshape(-1, 7).T @ x.detach().reshape(-1, 7))

    def test_shape_and_codec_boundaries_affect_stable_fixture_identity(self):
        base = fixtures.Fixture("packed-multiply", compression=True)
        self.assertEqual(base.identity, fixtures.Fixture(**asdict(base)).identity)
        for changes in ({"bits": 8}, {"layout": "strided"},
                        {"gradient_mask": "input"}, {"backward_order": "dw_first"}, {"activation_state": "exact"}):
            self.assertNotEqual(base.identity, fixtures.Fixture(**{**asdict(base), **changes}).identity)
        with self.assertRaisesRegex(ValueError, "divisible"):
            fixtures.Fixture("groupnorm-activation", width=63, groups=8)
        with self.assertRaisesRegex(ValueError, "Channels-last"):
            fixtures.Fixture("gemm-activation", layout="channels-last")

    def test_builds_broadcast_selective_gradient_and_strided_controls(self):
        for mask in ("all", "input", "weight", "none"):
            fixture = fixtures.Fixture("packed-multiply", width=7, gradient_mask=mask, layout="strided")
            model, x = fixtures.build_fixture(fixture, torch.device("cpu"), 13)
            self.assertEqual(model.scale.shape, (1, 1, 7))
            self.assertEqual(x.stride()[-1], 2)
            self.assertEqual(x.requires_grad, mask in ("all", "input"))
            self.assertEqual(model.scale.requires_grad, mask in ("all", "weight"))
            output = model(x)
            if output.requires_grad:
                output.sum().backward()
                self.assertEqual(model.scale.grad is not None, mask in ("all", "weight"))


class NativeFusionFixtureExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        self.snapshots = {}

    def job(self, fixture, strategy="generated-unfused"):
        return {"id": "test", "fixture": asdict(fixture), "fixture_id": fixture.identity,
                "trial": 0, "seed": 173, "backend": "triton", "device": "cpu",
                "strategy": next(item for item in fixtures.strategies("triton", fixture) if item["name"] == strategy),
                "source_sha256": "test-source", "measurement": {"warmup": 0, "iterations": 1,
                                                               "repeats": 1, "profile_steps": 0}}

    def execute(self, fixture, directory, strategy="generated-unfused"):
        with patch.object(fixtures, "source_identity", return_value="test-source"):
            result = fixtures.run_worker(self.job(fixture, strategy), directory,
                snapshot_sink=lambda job, snapshot: self.snapshots.__setitem__(str(directory), snapshot))
        result["directory"] = str(directory)
        self.assertTrue(result["numerical"]["passed"], result["numerical"])
        self.assertTrue(result["native_autodiff_ownership"])
        self.assertFalse(result["gpu_performance_qualified"])
        return result

    def compare(self, unfused, fused):
        return fixtures.compare_pair(unfused, fused, snapshot_inputs=(
            self.snapshots[unfused["directory"]], self.snapshots[fused["directory"]]))

    def test_cpu_norm_and_gemm_fixtures_execute_complete_matching_gradients(self):
        for kind in ("gemm-activation", "layernorm-activation", "groupnorm-activation"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                result = self.execute(fixtures.Fixture(kind, width=8, hidden=11, groups=2,
                                      batch=1, tokens=3, spatial=(2, 3), dtype="fp32"), Path(directory))
                self.assertTrue(result["timing"]["forward"]["wall_ms"]["median"] > 0)
                self.assertTrue(result["timing"]["backward"]["wall_ms"]["median"] > 0)
                self.assertEqual(result["candidate_evidence"]["observed_candidates"], 0)
                self.assertEqual(list(Path(directory).rglob("*.pt")), [])

    def test_full_numerical_comparison_never_calls_tensor_checkpoint_io(self):
        fixture = fixtures.Fixture("layernorm-activation", width=8, batch=1, tokens=3, dtype="fp32")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch, "save", side_effect=AssertionError("checkpoint write")), \
                patch.object(torch, "load", side_effect=AssertionError("checkpoint read")):
            first, second = Path(directory) / "unfused", Path(directory) / "fused"
            first.mkdir()
            second.mkdir()
            unfused, fused = self.execute(fixture, first), self.execute(fixture, second, "generated-fused")
            comparison = self.compare(unfused, fused)
            self.assertTrue(comparison["correctness_passed"], comparison)
            self.assertEqual(list(Path(directory).rglob("*.pt")), [])
            json.dumps(unfused, allow_nan=False)
            json.dumps(fused, allow_nan=False)

    def test_packed_multiply_oracle_preserves_unexpanded_coefficients_and_rounding(self):
        fixture = fixtures.Fixture("packed-multiply", width=63, batch=2, tokens=17,
                                   compression=True, bits=2, group_size=1024)
        model, x = fixtures.build_fixture(fixture, torch.device("cpu"), 11)
        gradient = torch.randn_like(x)
        oracle = fixtures._packed_multiply_oracle(model, x, gradient, fixture)
        self.assertEqual(oracle["status"], "oracle_only")
        self.assertEqual(oracle["unexpanded_coefficient_shape"], [1, 1, 63])
        self.assertEqual(oracle["scale_gradient_shape"], [1, 1, 63])
        self.assertEqual(oracle["product_dtype"], "torch.bfloat16")
        self.assertEqual(len(oracle["codec_components"]), 3)
        self.assertEqual(oracle["model_owned_scale"], "exact")

    def test_compressed_multiply_uses_the_matching_explicit_gradient_oracle(self):
        fixture = fixtures.Fixture("packed-multiply", width=64, batch=2, tokens=17,
                                   compression=True, bits=4, group_size=1024)
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(fixture, Path(directory), "generated-fused")
        self.assertTrue(result["multiplication_recipe"]["encoded"])
        self.assertEqual(result["multiplication_recipe"]["model_owned_scale"], "exact")
        self.assertTrue(result["numerical"]["matching_dtypes"])

    def test_shared_pack_controls_toggle_cse_and_producer_fusion_independently(self):
        fixture = fixtures.Fixture("shared-packs", compression=True)
        routes = {row["name"]: row["options"] for row in fixtures.strategies("triton", fixture)}
        for prefix in ("generated-unfused", "generated-fused"):
            first, second = dict(routes[prefix]), dict(routes[prefix + "-cse"])
            self.assertFalse(first.pop("share_compatible_packs"))
            self.assertTrue(second.pop("share_compatible_packs"))
            self.assertEqual(first, second)

    def test_all_frozen_norm_does_not_require_training_residuals(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(fixtures.Fixture("layernorm-activation", width=8, batch=1,
                                  tokens=3, gradient_mask="none"), Path(directory))
        self.assertFalse(result["native_autodiff_required"])
        self.assertEqual(result["memory"]["saved_tensors"], [])
        self.assertIsNone(result["timing"]["backward"]["wall_ms"])

    def test_exact_norm_activation_keeps_xhat_recipe_and_coefficient_rejection_explicit(self):
        for kind in ("layernorm-activation", "groupnorm-activation"):
            reports = {}
            for state in ("recipe", "exact"):
                with self.subTest(kind=kind, state=state), tempfile.TemporaryDirectory() as directory:
                    fixture = fixtures.Fixture(kind, width=8, groups=2, batch=1, tokens=3, spatial=(2, 3),
                                               compression=True, group_size=1024, activation_state=state)
                    result = self.execute(fixture, Path(directory), "generated-fused")
                    reports[state] = result["compiler_report"]
                    self.assertEqual(result["fixture_contract"]["activation_state"], state)
                    self.assertTrue(result["job"]["fixture"]["compression"])
            recipe_reasons = {row["reason"] for row in reports["recipe"]["fusion_candidates"]}
            exact_reasons = {row["reason"] for row in reports["exact"]["fusion_candidates"]}
            self.assertIn("activation_saved_coefficient_codec", recipe_reasons)
            self.assertNotIn("activation_saved_coefficient_codec", exact_reasons)

    def test_public_shared_projection_does_not_claim_explicit_pack_cse(self):
        fixture = fixtures.Fixture("shared-packs", width=8, hidden=11, batch=1, tokens=3,
                                   compression=True, group_size=1024)
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(fixture, Path(directory), "generated-fused-cse")
        self.assertEqual(result["compiler_report"]["pack_sharing_decisions"], [])
        self.assertEqual(result["fixture_contract"]["shared_projection_cse"], "unsupported_no_explicit_pack_operations")
        self.assertFalse(result["candidate_evidence"]["selected_candidates"])

    def test_cpu_pair_never_claims_cuda_even_when_numerically_correct(self):
        fixture = fixtures.Fixture("gemm-activation", width=8, hidden=11, batch=1, tokens=3)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "unfused", Path(directory) / "fused"
            first.mkdir()
            second.mkdir()
            unfused, fused = self.execute(fixture, first), self.execute(fixture, second, "generated-fused")
            comparison = self.compare(unfused, fused)
            self.assertTrue(comparison["correctness_passed"])
            self.assertTrue(comparison["same_backend_control"])
            self.assertFalse(comparison["cuda_fusion_observed"])
            self.assertIn(comparison["status"], ("unsupported", "unqualified"))
            self.assertFalse(comparison["gpu_performance_qualified"])
            original = fused["installed_deepopt"]
            fused["installed_deepopt"] = {**original, "source_sha256": "another-installed-wheel"}
            self.assertFalse(self.compare(unfused, fused)["correctness_passed"])
            fused["installed_deepopt"] = original
            fused["job"]["strategy"]["options"]["bf16_tile"] = 128
            self.assertFalse(self.compare(unfused, fused)["correctness_passed"])

    def test_residual_pair_matches_eager_and_rejects_ownership_or_recipe_mismatch(self):
        fixture = fixtures.Fixture("gemm-residual", width=7, batch=1, tokens=3)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "unfused", Path(directory) / "fused"
            first.mkdir()
            second.mkdir()
            unfused, fused = self.execute(fixture, first), self.execute(fixture, second, "generated-fused")
            comparison = self.compare(unfused, fused)
            self.assertTrue(comparison["correctness_passed"])
            self.assertTrue(comparison["same_backend_control"])
            self.assertFalse(comparison["cuda_fusion_observed"])
            fused["native_autodiff_ownership"] = False
            self.assertEqual(self.compare(unfused, fused)["status"], "failed")
            fused["native_autodiff_ownership"] = True
            fused["job"]["strategy"]["options"]["bf16_decode"] = "tiled"
            self.assertFalse(self.compare(unfused, fused)["same_backend_control"])

    def test_source_change_is_rejected_before_execution(self):
        fixture = fixtures.Fixture("gemm-activation")
        with patch.object(fixtures, "source_identity", return_value="changed-source"):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                fixtures.run_worker(self.job(fixture), Path("/tmp/unused-native-fusion-test"))

    def test_installed_distribution_change_is_rejected_before_execution(self):
        fixture = fixtures.Fixture("gemm-activation")
        job = {**self.job(fixture), "installed_deepopt": {"source_sha256": "planned"}}
        runtime = Mock()
        runtime.installed_identity.return_value = {"source_sha256": "replacement"}
        with patch.object(fixtures, "source_identity", return_value="test-source"), \
                patch.object(fixtures, "_installed_runtime", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "Installed DeepOpt changed"):
                fixtures.run_worker(job, Path("/tmp/unused-native-fusion-test"))


class FusionMemoryComparisonTests(unittest.TestCase):
    def entries(self, kind="layernorm-activation"):
        args = fixtures.parser().parse_args(["--fixtures", kind, "--phases", "both", "--trials", "1",
            "--batch", "1", "--tokens", "2", "--width", "4", "--hidden", "6", "--groups", "2",
            "--spatial", "2", "3", "--worker-isolation", "trial", "--output", "/tmp/fusion-ram-plan"])
        jobs = fixtures.make_plan(args)["jobs"]
        return [{"job": job, "result_path": f"/unused/{job['id']}/result.json"} for job in jobs]

    def numerical(self, job):
        return {name: torch.arange(prod(shape), dtype=torch.float32).reshape(shape) / 10
                for name, shape in fixtures._snapshot_shapes(job).items()}

    def row(self, job, tensors):
        identity = {"source_sha256": "installed-wheel"}
        return {"status": "ok", "job": job, "numerical": {"passed": True},
                "comparison_input": fixtures.snapshot_descriptor(job, tensors),
                "source_unchanged": True, "native_autodiff_ownership": True, "finite_after_timing": True,
                "initial_state_sha256": "state", "rng_baseline_sha256": "rng", "gradient_sha256": "gradient",
                "source_before_sha256": "source", "source_after_sha256": "source",
                "installed_deepopt": identity, "installed_deepopt_after": identity,
                "compiler_report": {}, "candidate_evidence": {"selected_candidates": 0, "all_selected_observed": False},
                "timing": {"full_update": {"wall_ms": {"median": 1.0}}}}

    def pair(self):
        routes = {entry["job"]["strategy"]["name"]: entry["job"] for entry in self.entries()}
        a, b = routes["generated-unfused"], routes["generated-fused"]
        first, second = self.numerical(a), self.numerical(b)
        return self.row(a, first), self.row(b, second), first, second

    def test_missing_malformed_changed_and_nonfinite_inputs_fail_closed(self):
        for variant in ("missing", "malformed", "changed", "nonfinite", "dtype"):
            with self.subTest(variant=variant):
                a, b, first, second = self.pair()
                if variant == "missing":
                    values = None
                elif variant == "malformed":
                    values = (first, {})
                else:
                    if variant == "changed":
                        second["output"].add_(1)
                    elif variant == "nonfinite":
                        second["output"].flatten()[0] = float("nan")
                        b["comparison_input"] = fixtures.snapshot_descriptor(b["job"], second)
                    else:
                        second["output"] = second["output"].bfloat16()
                        b["comparison_input"] = fixtures.snapshot_descriptor(b["job"], second)
                    values = (first, second)
                comparison = fixtures.compare_pair(a, b, snapshot_inputs=values)
                self.assertFalse(comparison["correctness_passed"])
                self.assertEqual(comparison["status"], "failed")

    def test_nullable_gradient_contract_matches_all_masks_and_unused_gemm_modules(self):
        for kind in fixtures.FIXTURES:
            for phase in fixtures.PHASES if kind == "gemm-activation" else ("both",):
                for mask in ("all", "input", "weight", "bias", "none"):
                    with self.subTest(kind=kind, phase=phase, mask=mask):
                        fixture = fixtures.Fixture(kind, phase=phase, gradient_mask=mask, dtype="fp32",
                            batch=1, tokens=2, width=4, hidden=6, groups=2, spatial=(2, 3))
                        model, x = fixtures.build_fixture(fixture, torch.device("cpu"), 173)
                        baseline = {name: value.detach().clone() for name, value in model.named_parameters()}
                        output = model(x)
                        if output.requires_grad:
                            output.sum().backward()
                        snapshot = fixtures._snapshot(model, x, output, baseline)
                        job = {**self.entries()[0]["job"], "fixture": asdict(fixture)}
                        descriptor = fixtures.snapshot_descriptor(job, snapshot)
                        shapes = fixtures._snapshot_shapes(job, include_absent=True)
                        self.assertEqual({entry["name"]: entry["shape"] for entry in descriptor["tensors"]},
                            {name: list(shape) if shape is not None else None for name, shape in shapes.items()})
                        required = next((name for name, shape in shapes.items()
                                         if name.startswith("parameter_grad.") and shape is not None), None)
                        if required is not None:
                            snapshot[required] = None
                            with self.assertRaisesRegex(ValueError, "Required numerical tensor"):
                                fixtures.snapshot_descriptor(job, snapshot)

    def test_eager_failure_remains_failed_when_direct_pair_is_identical(self):
        a, b, first, second = self.pair()
        a["numerical"] = {"passed": False, "tensors": {"parameter_grad.norm.weight": {"passed": False}}}
        b["numerical"] = a["numerical"]
        comparison = fixtures.compare_pair(a, b, snapshot_inputs=(first, second))
        self.assertTrue(comparison["numerical_fused_vs_unfused"]["passed"])
        self.assertFalse(comparison["correctness_passed"])
        self.assertEqual(comparison["status"], "failed")

    def test_malformed_scalar_results_fail_without_aborting_remaining_strategies(self):
        invalid = ({"numerical": {"passed": "false"}}, {"numerical": {}},
                   {"source_unchanged": "false"}, {"finite_after_timing": 1},
                   {"compiler_report": {"fusion_candidates": None}},
                   {"compiler_report": {"graphs": [None]}},
                   {"timing": {"full_update": {"wall_ms": {"median": float("nan")}}}})
        for changes in invalid:
            with self.subTest(changes=changes):
                a, b, first, second = self.pair()
                a.update(changes)
                self.assertFalse(fixtures.compare_pair(a, b, snapshot_inputs=(first, second))["correctness_passed"])
                entries = self.entries()
                collector = fixtures.NumericalComparisons(entries)
                for index, entry in enumerate(entries):
                    job = entry["job"]
                    tensors = self.numerical(job)
                    collector.add_snapshot(job, tensors)
                    row = self.row(job, tensors)
                    if not index:
                        row.update(changes)
                    collector.add_result(row)
                report = collector.finish()
                self.assertFalse(report["comparisons"][0]["correctness_passed"])
                self.assertEqual(len(collector.rows), len(entries))
                json.dumps({"results": list(collector.rows.values()), **report}, allow_nan=False)

    def test_shared_pack_group_retains_all_four_direct_pairs_and_releases_ram(self):
        entries = self.entries("shared-packs")
        collector = fixtures.NumericalComparisons(entries)
        rows = []
        with patch.object(torch, "save", side_effect=AssertionError("checkpoint write")), \
                patch.object(torch, "load", side_effect=AssertionError("checkpoint read")):
            for entry in entries:
                job = entry["job"]
                tensors = self.numerical(job)
                collector.add_snapshot(job, tensors)
                row = self.row(job, tensors)
                collector.add_result(row)
                rows.append(row)
            report = collector.finish()
        self.assertEqual(len(report["comparisons"]), 4)
        self.assertTrue(all(row["correctness_passed"] for row in report["comparisons"]))
        self.assertEqual(collector.snapshots, {})
        self.assertEqual(collector.snapshot_bytes, 0)
        self.assertEqual(report["numerical_storage"]["persisted_tensor_bytes"], 0)
        self.assertLessEqual(report["numerical_storage"]["peak_snapshot_bytes"],
            5 * fixtures.snapshot_descriptor(entries[0]["job"], self.numerical(entries[0]["job"]))["total_bytes"])
        self.assertEqual(len(fixtures._validate_group_report([entry["job"] for entry in entries], rows, report)), 4)
        json.dumps(report, allow_nan=False)
        report["comparisons"].pop()
        with self.assertRaises(ValueError):
            fixtures._validate_group_report([entry["job"] for entry in entries], rows, report)

    def test_missing_snapshot_or_failed_worker_keeps_required_comparisons_failed(self):
        entries = self.entries()
        for failed in (False, True):
            collector = fixtures.NumericalComparisons(entries)
            for index, entry in enumerate(entries):
                job = entry["job"]
                tensors = self.numerical(job)
                if index:
                    collector.add_snapshot(job, tensors)
                row = self.row(job, tensors)
                if not index and failed:
                    row.update(status="error", reason="compile failed")
                collector.add_result(row)
            report = collector.finish()
            self.assertEqual(len(report["comparisons"]), 1)
            self.assertFalse(report["comparisons"][0]["correctness_passed"])
            self.assertEqual(report["numerical_storage"]["remaining_snapshot_bytes"], 0)

    def test_receiver_failure_and_nonzero_strategy_exit_fail_closed(self):
        for variant in ("transfer", "exit", "malformed-result"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                entries = self.entries()
                for entry in entries:
                    entry["job"]["installed_deepopt"] = {"source_sha256": "installed-wheel"}
                entry = entries[0]
                entry["result_path"] = str(Path(directory) / "result.json")
                job = entry["job"]
                tensors = self.numerical(job)
                row = self.row(job, tensors)
                Path(entry["result_path"]).write_text(json.dumps(row) if variant != "malformed-result" else "{")
                collector = fixtures.NumericalComparisons(entries)
                broker = Mock()
                broker.environment = {}
                broker.receive.return_value = {"descriptor": row["comparison_input"], "tensors": tensors}
                if variant == "transfer":
                    broker.receive.side_effect = ValueError("truncated numerical payload")
                manager = Mock()
                manager.__enter__ = Mock(return_value=broker)
                manager.__exit__ = Mock(return_value=False)
                runtime = Mock()
                runtime.child_environment.return_value = {}
                with patch("analysis.numerical_ipc.NumericalSnapshotBroker", return_value=manager), \
                        patch.object(fixtures, "_installed_runtime", return_value=runtime), \
                        patch.object(fixtures.subprocess, "run", return_value=Mock(returncode=1 if variant == "exit" else 0)):
                    result = fixtures._isolated_strategy(entry, collector)
                self.assertEqual(result["status"], "error")
                report = collector.finish()
                self.assertFalse(report["comparisons"][0]["correctness_passed"])
                self.assertEqual(list(Path(directory).rglob("*.pt")), [])

    def test_idle_receiver_polling_keeps_waiting_for_a_long_running_strategy(self):
        import threading
        from analysis.numerical_ipc import IdleTimeout

        with tempfile.TemporaryDirectory() as directory:
            entries = self.entries()
            for entry in entries:
                entry["job"]["installed_deepopt"] = {"source_sha256": "installed-wheel"}
            entry = entries[0]
            entry["result_path"] = str(Path(directory) / "result.json")
            job = entry["job"]
            tensors = self.numerical(job)
            row = self.row(job, tensors)
            Path(entry["result_path"]).write_text(json.dumps(row))
            ready = threading.Event()
            calls = []
            def receive(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise IdleTimeout("worker still compiling")
                ready.set()
                return {"descriptor": row["comparison_input"], "tensors": tensors}
            broker = Mock()
            broker.environment = {}
            broker.receive.side_effect = receive
            manager = Mock()
            manager.__enter__ = Mock(return_value=broker)
            manager.__exit__ = Mock(return_value=False)
            runtime = Mock()
            runtime.child_environment.return_value = {}
            def complete(*args, **kwargs):
                self.assertTrue(ready.wait(2))
                return Mock(returncode=0)
            with patch("analysis.numerical_ipc.NumericalSnapshotBroker", return_value=manager), \
                    patch.object(fixtures, "_installed_runtime", return_value=runtime), \
                    patch.object(fixtures.subprocess, "run", side_effect=complete):
                result = fixtures._isolated_strategy(entry, fixtures.NumericalComparisons(entries))
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(len(calls), 2)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"),
                     "Isolated physical-fusion qualification requires CUDA and Triton")
class NativeFusionFixtureCudaTests(unittest.TestCase):
    def qualify(self, fixture, *, expect_candidate=True):
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("Generated BF16 requires NVIDIA compute capability >= 8")
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        job = {"id": "cuda-test", "fixture": asdict(fixture), "fixture_id": fixture.identity,
               "trial": 0, "seed": 173, "backend": "triton", "device": "cuda:0",
               "strategy": next(item for item in fixtures.strategies("triton", fixture)
                                if item["name"] == "generated-fused"),
               "source_sha256": "test-source", "measurement": {"warmup": 1, "iterations": 1,
                                                               "repeats": 1, "profile_steps": 1}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(fixtures, "source_identity", return_value="test-source"):
                result = fixtures.run_worker(job, Path(directory))
        self.assertTrue(result["numerical"]["passed"], result["numerical"])
        self.assertTrue(result["native_autodiff_ownership"])
        if expect_candidate:
            self.assertTrue(result["candidate_evidence"]["all_selected_observed"], result["candidate_evidence"])
            self.assertGreater(result["candidate_evidence"]["selected_candidates"], 0)
        else:
            self.assertEqual(result["candidate_evidence"]["selected_candidates"], 0)
            self.assertEqual(result["candidate_evidence"]["observed_candidates"], 0)
        self.assertFalse(result["gpu_performance_qualified"])
        return result

    def test_gemm_forward_backward_and_combined_have_candidate_specific_launches(self):
        for phase in fixtures.PHASES:
            with self.subTest(phase=phase):
                self.qualify(fixtures.Fixture("gemm-activation", phase=phase, width=63, hidden=96,
                                             batch=1, tokens=17))

    def test_gemm_residual_candidate_executes_and_preserves_aliased_gradient(self):
        for phase in ("forward", "both"):
            with self.subTest(phase=phase):
                result = self.qualify(fixtures.Fixture("gemm-residual", phase=phase, width=63,
                                                      batch=1, tokens=17))
                selected = [row for row in result["compiler_report"]["fusion_candidates"] if row["selected"]]
                self.assertEqual({row["pattern"] for row in selected}, {"bf16_residual_gemm_epilogue"})
                self.assertEqual({row["execution_phase"] for row in selected}, {"forward"})
                self.assertTrue(all("_bf16_gemm" in row["kernel_evidence"]["required_kernel_roles"]["gemm_epilogue"]
                                    for row in selected))

    def test_backward_only_residual_fixture_does_not_claim_forward_epilogue(self):
        result = self.qualify(fixtures.Fixture("gemm-residual", phase="backward", width=63,
                                               batch=1, tokens=17), expect_candidate=False)
        self.assertFalse(any(row["selected"] for row in result["compiler_report"]["fusion_candidates"]))

    def test_normalization_forward_backward_and_combined_use_matching_codec(self):
        for kind in ("layernorm-activation", "groupnorm-activation"):
            for phase in fixtures.PHASES:
                with self.subTest(kind=kind, phase=phase):
                    result = self.qualify(fixtures.Fixture(kind, phase=phase, width=64, groups=4,
                        batch=1, tokens=17, spatial=(4, 4), compression=True, bits=4, group_size=1024,
                        activation_state="exact"))
                    self.assertEqual(result["output_dtype"], "torch.bfloat16")

    def test_compressed_activation_coefficients_are_visible_norm_rejections(self):
        for kind in ("layernorm-activation", "groupnorm-activation"):
            with self.subTest(kind=kind):
                result = self.qualify(fixtures.Fixture(kind, width=64, groups=4, batch=1, tokens=17,
                    spatial=(4, 4), compression=True, group_size=1024, activation_state="recipe"), expect_candidate=False)
                self.assertEqual({row["reason"] for row in result["compiler_report"]["fusion_candidates"]},
                                 {"activation_saved_coefficient_codec"})

    def test_compressed_gemm_forward_without_packing_consumer_is_unsupported(self):
        result = self.qualify(fixtures.Fixture("gemm-activation", phase="forward", width=64, hidden=96,
            batch=1, tokens=17, compression=True, group_size=1024, activation_state="exact"), expect_candidate=False)
        self.assertFalse(any(row["selected"] for row in result["compiler_report"]["fusion_candidates"]))

    def test_packed_broadcast_multiply_removes_standalone_decode_product(self):
        self.qualify(fixtures.Fixture("packed-multiply", width=64, batch=2, tokens=17,
                                    compression=True, bits=4, group_size=1024))



class CounterbalancedFusionQualificationTests(unittest.TestCase):
    def test_full_matrix_keeps_915_original_strategies_plus_separate_diagnostics(self):
        from compiler import qualify_backend

        suite = qualify_backend.make_plan(qualify_backend.parser().parse_args(["--profile", "gpu-full"]))
        policies = {name: {"jobs": [], "diagnostic_jobs": []}
                    for name in ("torch_amp", "bf16_storage")}
        for stage in suite["stages"]:
            command = stage["command"]
            if command[0] != "benchmarks/compiler/isolated_fusion.py":
                continue
            plan = fixtures.make_plan(fixtures.parser().parse_args(command[1:]))
            for collection in ("jobs", "diagnostic_jobs"):
                for job in plan[collection]:
                    policy = job["fixture"]["bf16_policy"]
                    self.assertIn(policy, policies)
                    policies[policy][collection].append(job)
        for policy, collections in policies.items():
            with self.subTest(bf16_policy=policy):
                original, diagnostics = collections["jobs"], collections["diagnostic_jobs"]
                self.assertEqual(len(original), 915)
                self.assertEqual(len(diagnostics), 1116)
                self.assertTrue(all(job["fixture"]["group_size"] == 1024 for job in original + diagnostics))
                self.assertTrue(all(job["measurement"]["iterations"] == 50 and job["measurement"]["repeats"] == 5
                                    for job in original + diagnostics))

    def test_parent_rejects_malformed_order_and_scalar_evidence(self):
        helper = FusionMemoryComparisonTests()
        entries = helper.entries()
        collector = fixtures.NumericalComparisons(entries)
        rows = []
        for entry in entries:
            job = entry["job"]
            tensors = helper.numerical(job)
            collector.add_snapshot(job, tensors)
            row = helper.row(job, tensors)
            rows.append(row)
            collector.add_result(row)
        report = collector.finish()
        for field, value in (("diagnostic_order", "BA"), ("cuda_fusion_observed", "false"),
                             ("wall_speedup_generated_pair", float("nan"))):
            altered = json.loads(json.dumps(report))
            altered["comparisons"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                fixtures._validate_group_report([entry["job"] for entry in entries], rows, altered)

    def plan(self, kind="layernorm-activation"):
        args = fixtures.parser().parse_args(["--fixtures", kind, "--phases", "both", "--trials", "3",
            "--compression", "on", "--bits", "1", "2", "4", "8", "--worker-isolation", "trial",
            "--output", "/unused/counterbalanced"])
        return args, fixtures.make_plan(args)

    def test_additive_balanced_orders_preserve_main_jobs_and_measurement_policy(self):
        args, plan = self.plan()
        args.counterbalanced_norms = False
        control = fixtures.make_plan(args)
        self.assertEqual(plan["jobs"], control["jobs"])
        self.assertEqual(len(plan["jobs"]), 24)
        self.assertEqual(len(plan["diagnostic_jobs"]), 48)
        self.assertEqual(len({j["id"] for j in plan["jobs"] + plan["diagnostic_jobs"]}), 72)
        main = {(j["fixture_id"], j["trial"], j["strategy"]["name"]): j for j in plan["jobs"]}
        for group in fixtures.comparison_groups({"jobs": plan["diagnostic_jobs"]}):
            order = group[0]["diagnostic_order"]
            self.assertEqual([j["strategy"]["name"] for j in group],
                ["generated-unfused", "generated-fused"] if order == "AB" else ["generated-fused", "generated-unfused"])
            for job in group:
                original = main[job["fixture_id"], job["trial"], job["strategy"]["name"]]
                for field in ("seed", "fixture", "strategy", "measurement", "source_sha256"):
                    self.assertEqual(job[field], original[field])
                self.assertEqual(job["fixture"]["group_size"], 1024)
        self.assertEqual(self.plan("gemm-activation")[1]["diagnostic_jobs"], [])

    def evidence(self):
        _, plan = self.plan()
        # One bits cell with three independent trials and both diagnostic orders.
        fixture = plan["jobs"][0]["fixture_id"]
        jobs = [j for j in plan["diagnostic_jobs"] if j["fixture_id"] == fixture]
        helper = FusionMemoryComparisonTests()
        rows = [helper.row(job, helper.numerical(job)) for job in jobs]
        pairs = [{"fixture_id": fixture, "backend": "triton", "trial": trial, "diagnostic_order": order,
                  "correctness_passed": True, "cuda_fusion_observed": True, "wall_speedup_generated_pair": 1.05}
                 for trial in range(3) for order in ("AB", "BA")]
        return jobs, rows, pairs

    def test_balanced_gate_rejects_missing_order_cpu_evidence_reversal_and_state_drift(self):
        for variant in ("valid", "missing", "cpu", "reversal", "state"):
            jobs, rows, pairs = self.evidence()
            if variant == "missing":
                pairs.pop()
            elif variant == "cpu":
                pairs[0]["cuda_fusion_observed"] = False
            elif variant == "reversal":
                pairs[0]["wall_speedup_generated_pair"] = 0.98
            elif variant == "state":
                rows[0]["rng_baseline_sha256"] = "different"
            report, = fixtures.counterbalanced_summary(jobs, rows, pairs)
            self.assertEqual(report["order_stable_latency_gain"], variant == "valid", variant)
            self.assertFalse(report["gpu_performance_qualified"])
            self.assertEqual(len(report["trials"]), 3)

    def test_identity_session_wraps_one_comparison_group(self):
        from contextlib import contextmanager
        events = []
        @contextmanager
        def session(root):
            events.append("enter")
            yield
            events.append("exit")
        with patch.object(fixtures, "_installed_runtime", return_value=Mock(identity_session=session)), \
                patch.object(fixtures, "_execute_comparison_group", side_effect=lambda *a, **k: events.append("execute") or {}):
            fixtures.execute_comparison_group([], isolation="trial")
        self.assertEqual(events, ["enter", "execute", "exit"])

    def cost_input(self):
        args, plan = self.plan("gemm-activation")
        fixture = plan["jobs"][0]["fixture_id"]
        helper = FusionMemoryComparisonTests()
        rows = []
        for job in plan["jobs"]:
            if job["fixture_id"] != fixture:
                continue
            row = helper.row(job, helper.numerical(job))
            row["timing"]["full_update"]["wall_ms"]["median"] = {
                "generated-fused": 1.0, "generated-unfused": 1.1, "library-control": 0.8}[job["strategy"]["name"]]
            row["memory"] = {"peak_extra_allocated_bytes": 10 if job["strategy"]["name"] == "generated-fused" else 20}
            rows.append(row)
        pairs = [{"fixture_id": fixture, "backend": "triton", "trial": trial, "transformation": "kernel_fusion",
                  "correctness_passed": True, "cuda_fusion_observed": True, "control_options": {}}
                 for trial in range(3)]
        return rows, pairs

    def test_cost_decision_prefers_faster_library_or_explicit_capacity_tradeoff(self):
        rows, pairs = self.cost_input()
        decision, = fixtures.measured_cost_decisions(rows, pairs, [])
        self.assertEqual(decision["recommendation"], "library-control")
        capacity, = fixtures.measured_cost_decisions(rows, pairs, [], objective="capacity", capacity_bytes=10)
        self.assertEqual(capacity["recommendation"], "generated-fused")
        self.assertFalse(capacity["automatic_default_changed"])
        self.assertFalse(capacity["gpu_performance_qualified"])
        for variant in ("eager_fail", "cuda_missing", "few_trials", "budget_missing", "unknown_peak"):
            rows, pairs = self.cost_input()
            kwargs = {}
            if variant == "eager_fail":
                pairs[0]["correctness_passed"] = False
            elif variant == "cuda_missing":
                pairs[0]["cuda_fusion_observed"] = False
            elif variant == "few_trials":
                pairs.pop()
            else:
                kwargs = {"objective": "capacity", "capacity_bytes": None if variant == "budget_missing" else 10}
                for row in rows:
                    row["memory"]["peak_extra_allocated_bytes"] = None
            decision, = fixtures.measured_cost_decisions(rows, pairs, [], **kwargs)
            self.assertIsNone(decision["recommendation"], variant)
            self.assertTrue(decision["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
