"""Reporting contracts and actual CPU autograd for the standalone eager comparison."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/fp8/benchmark_fp8_linear_compare.py"
spec = importlib.util.spec_from_file_location("fp8_linear_compare", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def args(*extra):
    options = benchmark.parser().parse_args(list(extra))
    options.num_stages = 3
    return options


def case():
    return dict(projection="qkv", M=7, K=5, N=3, input_dtype="fp32", multiplicity=12)


def result(name, time=2.):
    return dict(implementation=name, math_policy="amp", batch=512, status="ok", controls={"eager": True},
                environment={"torch": "same", "gpu": "same"}, cases=[dict(
                    projection="qkv", status="ok", initial={"weight": "same", "input": "same"},
                    numerical={"all_finite": True}, timing={"wall_ms": {"median": time}}, multiplicity=12)])


class ComparisonContractTests(unittest.TestCase):
    def test_native_backward_is_an_explicit_opt_in_benchmark_control(self):
        self.assertEqual(args().backward_backend, "inherit")
        self.assertEqual(args("--backward-backend", "native_fast").backward_backend, "native_fast")
        session = benchmark.Session(args("--device", "cpu"), "deepopt-current", "amp", case(), torch,
                                    deepopt_backend="reference")
        capture = benchmark.Capture(session)
        with capture.enter():
            session.step(update=False)
        self.assertEqual({record["actual_backend"] for record, _ in capture.gemms}, {"reference"})
        self.assertEqual(benchmark.library_state(session)["numerical_recipe"]["version"], 5)

    def test_defaults_are_eager_real_updates_all_implementations_and_both_math_policies(self):
        options = args()
        self.assertEqual(options.implementations, list(benchmark.IMPLEMENTATIONS))
        self.assertEqual(options.math_policies, ["amp", "matched-fp32"])
        self.assertTrue(options.optimizer_updates)
        self.assertFalse(options.fill_uninitialized_memory)
        self.assertTrue(options.deterministic)
        self.assertTrue(options.trace)

    def test_real_vit_shapes_tail_and_head_contract(self):
        rows = benchmark.cases(args(), 336)
        self.assertEqual((rows[0]["M"], rows[0]["K"], rows[0]["N"]), (66192, 768, 2304))
        self.assertEqual(rows[1]["input_dtype"], "bf16")
        self.assertEqual(rows[-1]["M"], 336)
        self.assertEqual(rows[-1]["N"], 10)
        self.assertEqual(sum(row["multiplicity"] for row in rows), 49)

    def test_seed_does_not_depend_on_projection_filter_or_implementation_order(self):
        first = args("--projections", "qkv", "fc1")
        second = args("--projections", "fc1")
        self.assertEqual(benchmark.case_seed(first, benchmark.cases(first, 512)[1]),
                         benchmark.case_seed(second, benchmark.cases(second, 512)[0]))

    def test_torchao_formats_and_all_three_accumulation_policies_are_explicit(self):
        fake = SimpleNamespace(CastConfig=lambda **kw: kw, Float8GemmConfig=lambda **kw: kw,
                               Float8LinearConfig=lambda **kw: kw)
        with patch.dict(sys.modules, {"torchao.float8": fake}):
            configuration = benchmark.torchao_config(args("--format", "hybrid"), torch)
            self.assertEqual(configuration["cast_config_grad_output"]["target_dtype"], torch.float8_e5m2)
            self.assertEqual(configuration["cast_config_input"]["target_dtype"], torch.float8_e4m3fn)
            for role in ("output", "grad_input", "grad_weight"):
                self.assertTrue(configuration["gemm_config_" + role]["use_fast_accum"])
            defaults = benchmark.torchao_config(args("--accumulation", "torchao-default"), torch)
            self.assertFalse(any(key.startswith("gemm_config") for key in defaults))

    def test_stats_preserve_chronological_order(self):
        self.assertEqual(benchmark.stats([3, 1, 2])["samples"], [3, 1, 2])

    def test_comparison_rejects_different_inputs_nonfinite_values_and_different_controls(self):
        for mutation in ("initial", "finite", "controls", "environment"):
            baseline, fp8 = result("bf16"), result("deepopt-current", 1.)
            if mutation == "initial":
                fp8["cases"][0]["initial"]["input"] = "different"
            elif mutation == "finite":
                fp8["cases"][0]["numerical"]["all_finite"] = False
            elif mutation == "controls":
                fp8["controls"]["eager"] = False
            else:
                fp8["environment"]["gpu"] = "different"
            with self.subTest(mutation=mutation):
                self.assertIsNone(benchmark.comparisons([baseline, fp8])[0]["speedup_vs_bf16"])

    def test_comparison_includes_torchao_peer_without_calling_it_numerical_parity(self):
        rows = benchmark.comparisons([result("bf16", 3.), result("deepopt-delayed", 2.), result("torchao", 1.)])
        deepopt = next(row for row in rows if row["implementation"] == "deepopt-delayed")
        self.assertEqual(deepopt["speedup_vs_bf16"], 1.5)
        self.assertEqual(deepopt["speedup_vs_torchao"], .5)
        self.assertIn("may differ", deepopt["internal_dtype_contract"])

    def test_matched_peer_ratio_requires_observed_fp32_products_in_both_libraries(self):
        deepopt, torchao = result("deepopt-delayed"), result("torchao")
        for job in (deepopt, torchao):
            job["math_policy"] = "matched-fp32"
            job["cases"][0].update(three_gemms_observed=True, gemm_calls=[
                {"role": role, "output": {"dtype": "torch.float32"}}
                for role in ("forward", "input_gradient", "weight_gradient")])
        self.assertEqual(benchmark.comparisons([deepopt, torchao])[0]["speedup_vs_torchao"], 1.)
        torchao["cases"][0]["gemm_calls"][1]["output"]["dtype"] = "torch.bfloat16"
        self.assertIsNone(benchmark.comparisons([deepopt, torchao])[0]["speedup_vs_torchao"])

    def test_changed_source_fingerprints_suppress_comparison(self):
        baseline, fp8 = result("bf16"), result("deepopt-current")
        baseline["environment"]["packages"] = {"deepopt": {"source": {"sha256": "old"}}}
        fp8["environment"]["packages"] = {"deepopt": {"source": {"sha256": "new"}}}
        self.assertIsNone(benchmark.comparisons([baseline, fp8])[0]["speedup_vs_bf16"])

    def test_cuda_user_ranges_are_excluded_from_activity_sum(self):
        cpu, gpu = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
        profiler = SimpleNamespace(key_averages=lambda: [SimpleNamespace(
            device_type=cpu, key="aten::mm", count=2, self_cpu_time_total=500, cpu_time_total=4000)],
            events=lambda: [SimpleNamespace(device_type=gpu, name="gemm", device_time_total=2000),
                            SimpleNamespace(device_type=gpu, name="linear.forward", device_time_total=99000,
                                            is_user_annotation=True)])
        report = benchmark.profile_summary(profiler, torch, 2, 10)
        self.assertEqual(report["cuda_activity_sum_ms_per_update"], 1.)
        self.assertEqual(report["top_cpu_ops"][0]["self_ms_per_update"], .25)

    def test_zip_contains_only_this_run_and_parent_uses_normal_installed_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            options = args("--output", directory, "--implementations", "bf16", "--math-policies", "amp")
            Path(directory, "old.trace.json").write_text("stale")
            commands = []

            def launch(command, **kwargs):
                commands.append((command, kwargs))
                payload = json.loads(command[-1])
                data = {"run_id": payload["run_id"], "status": "ok", "cases": []}
                if payload.get("check"):
                    data["environment"] = {"cuda_available": True}
                benchmark.write_json(Path(payload["directory"]) / "result.json", data)
                return SimpleNamespace(returncode=0)

            with patch.object(benchmark.subprocess, "run", side_effect=launch), patch("sys.stdout", new=io.StringIO()):
                report = benchmark.run(options)
            archive = Path(directory + ".zip")
            self.addCleanup(lambda: archive.unlink(missing_ok=True))
            with zipfile.ZipFile(archive) as bundle:
                self.assertNotIn("old.trace.json", bundle.namelist())
                self.assertIn("summary.json", bundle.namelist())
                self.assertTrue(any(report["run_id"] in name for name in bundle.namelist()))
            for command, kwargs in commands:
                self.assertEqual(command[0], sys.executable)
                self.assertNotIn("cwd", kwargs)
                self.assertNotIn("env", kwargs)

    def test_failed_worker_keeps_partial_measurements_in_summary_and_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            options = args("--output", directory, "--implementations", "torchao", "--math-policies", "amp")

            def launch(command, **kwargs):
                payload = json.loads(command[-1])
                if payload.get("check"):
                    data = {"run_id": payload["run_id"], "status": "ok", "environment": {"cuda_available": True}}
                    code = 0
                else:
                    data = {**result("torchao"), "run_id": payload["run_id"]}
                    code = 1
                benchmark.write_json(Path(payload["directory"]) / "result.json", data)
                return SimpleNamespace(returncode=code)

            with patch.object(benchmark.subprocess, "run", side_effect=launch), patch("sys.stdout", new=io.StringIO()):
                report = benchmark.run(options)
            self.addCleanup(lambda: Path(directory + ".zip").unlink(missing_ok=True))
            self.assertEqual(report["results"][0]["status"], "error")
            self.assertEqual(report["results"][0]["cases"][0]["timing"]["wall_ms"]["median"], 2.)

    def test_script_help_runs_after_copying_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "compare.py")
            target.write_text(SCRIPT.read_text())
            completed = subprocess.run([sys.executable, str(target), "--help"], cwd=directory, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--prepared-gemms", completed.stdout)

    def test_case_failure_does_not_erase_checkpointed_timing(self):
        options = args("--projections", "qkv")
        settings = {key: value for key, value in vars(options).items() if key not in ("output", "_worker")}
        env = {"cuda_available": True, "capability": [9, 0],
               "packages": {name: {"status": "ok"} for name in ("deepopt", "triton", "torchao")}}

        def fail_after_timing(session, options, directory, row, checkpoint):
            row["timing"] = {"wall_ms": {"median": 1.25}}
            checkpoint()
            raise RuntimeError("later diagnostic failed")

        with tempfile.TemporaryDirectory() as directory:
            payload = dict(settings=settings, directory=directory, run_id="test", implementation="bf16",
                           math_policy="amp", batch=512)
            with patch.object(benchmark, "environment", return_value=env), patch.object(benchmark, "configure"), \
                 patch.object(benchmark, "Session"), patch.object(benchmark, "run_case", side_effect=fail_after_timing), \
                 patch("sys.stdout", new=io.StringIO()):
                benchmark.worker(payload)
            report = json.loads(Path(directory, "result.json").read_text())
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["cases"][0]["timing"]["wall_ms"]["median"], 1.25)
        self.assertIn("later diagnostic failed", report["cases"][0]["error"])


class CPUAutogradIntegrationTests(unittest.TestCase):
    def test_native_amp_capture_records_three_real_matmuls_without_changing_gradients(self):
        session = benchmark.Session(args("--device", "cpu"), "bf16", "amp", case(), torch)
        session.step(update=False)
        expected = [session.x.grad.clone(), *(p.grad.clone() for p in session.parameters)]
        capture = benchmark.Capture(session)
        with capture.enter():
            session.step(update=False)
        self.assertEqual({record["role"] for record, _ in capture.gemms},
                         {"forward", "input_gradient", "weight_gradient"})
        for actual, reference in zip([session.x.grad, *(p.grad for p in session.parameters)], expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_deepopt_delayed_shared_boundary_and_immutable_operands_execute_on_cpu(self):
        session = benchmark.Session(args("--device", "cpu", "--history-length", "3"), "deepopt-delayed", "amp",
                                    case(), torch, deepopt_backend="reference")
        capture = benchmark.Capture(session)
        with capture.enter():
            output = session.step(update=False)
        self.assertEqual([record["role"] for record in capture.quantization], ["activation", "weight", "grad"])
        self.assertEqual({record["role"] for record, _ in capture.gemms},
                         {"forward", "input_gradient", "weight_gradient"})
        self.assertEqual(session.projection.fp8_scaling.updates.tolist(), [1, 1, 1])
        self.assertTrue(benchmark.numerical(session, output)["all_finite"])

    def test_matched_mode_records_fp32_outputs_for_all_three_fp8_products(self):
        session = benchmark.Session(args("--device", "cpu"), "deepopt-current", "matched-fp32", case(), torch,
                                    deepopt_backend="reference")
        capture = benchmark.Capture(session)
        with capture.enter():
            output = session.step(update=False)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(all(record["output"]["dtype"] == "torch.float32" for record, _ in capture.gemms))
        self.assertIs(session.projection.weight, session.source.weight)

    def test_amp_mode_records_bf16_dw_before_fp32_parameter_accumulation(self):
        for implementation in ("deepopt-current", "deepopt-delayed"):
            with self.subTest(implementation=implementation):
                session = benchmark.Session(args("--device", "cpu"), implementation, "amp", case(), torch,
                                            deepopt_backend="reference")
                capture = benchmark.Capture(session)
                with capture.enter():
                    session.step(update=False)
                dtypes = {record["role"]: record["output"]["dtype"] for record, _ in capture.gemms}
                self.assertEqual(dtypes, {"forward": "torch.bfloat16", "input_gradient": "torch.float32",
                                          "weight_gradient": "torch.bfloat16"})
                self.assertEqual(session.source.weight.grad.dtype, torch.float32)

    def test_diagnostic_patches_restore_after_failure_and_current_gradients_are_unchanged(self):
        from deepopt.opt.nn.autograd import linear as linear_autograd_func

        session = benchmark.Session(args("--device", "cpu"), "deepopt-current", "amp", case(), torch,
                                    deepopt_backend="reference")
        expected = session.step(update=False)
        gradient = session.source.weight.grad.clone()
        original = linear_autograd_func.gemm_nt
        with benchmark.Capture(session).enter():
            actual = session.step(update=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(session.source.weight.grad, gradient, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with benchmark.Capture(session).enter():
                raise RuntimeError("probe failure")
        self.assertIs(linear_autograd_func.gemm_nt, original)

    def test_optimizer_steps_change_original_fp32_parameters_and_clear_previous_gradients(self):
        session = benchmark.Session(args("--device", "cpu"), "bf16", "amp", case(), torch)
        before = session.source.weight.detach().clone()
        session.step()
        self.assertEqual(session.completed_updates, 1)
        self.assertFalse(torch.equal(before, session.source.weight))
        self.assertTrue(all(p.dtype == torch.float32 for p in session.parameters))
        session.clear()
        self.assertIsNone(session.x.grad)
        self.assertTrue(all(p.grad is None for p in session.parameters))

    def test_head_falls_back_explicitly_without_importing_torchao(self):
        head = {**case(), "projection": "head"}
        session = benchmark.Session(args("--device", "cpu"), "torchao", "amp", head, torch)
        self.assertTrue(session.dense)
        self.assertIsNone(session.configuration)
        session.step(update=False)

    def test_saved_tensor_report_counts_shared_storage_once(self):
        session = benchmark.Session(args("--device", "cpu"), "bf16", "amp", case(), torch)
        report = benchmark.saved_tensors(session)
        self.assertGreater(report["unique_storage_bytes"], 0)
        self.assertEqual(report["unique_storage_bytes"], sum(report["bytes_by_physical_dtype"].values()))


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip is None, "Requires a real NVIDIA CUDA device")
class GPUProbeIntegrationTests(unittest.TestCase):
    def test_real_torchao_records_internal_scaled_gemms_in_both_math_modes(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("Requires SM90+")
        try:
            import torchao.float8  # noqa: F401
        except ImportError as error:
            self.skipTest(str(error))
        small = dict(projection="attention", M=64, K=64, N=128, input_dtype="bf16", multiplicity=12)
        for policy in ("amp", "matched-fp32"):
            session = benchmark.Session(args(), "torchao", policy, small, torch)
            capture = benchmark.Capture(session)
            with capture.enter():
                session.step()
            self.assertEqual(len(capture.gemms), 3)
            self.assertEqual({record["role"] for record, _ in capture.gemms},
                             {"forward", "input_gradient", "weight_gradient"})
            if policy == "matched-fp32":
                self.assertTrue(all(record["output"]["dtype"] == "torch.float32" for record, _ in capture.gemms))


if __name__ == "__main__":
    unittest.main()
