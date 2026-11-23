"""Role geometry, codec oracles and fail-closed launch benchmark behavior."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/compiler/benchmark_bf16_configurations.py"
SPEC = importlib.util.spec_from_file_location("bf16_configurations_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BF16ConfigurationsBenchmarkTests(unittest.TestCase):
    def arguments(self, *options):
        return benchmark.parser().parse_args(["--output", "unused.json", "--shapes", "37x35x17", *options])

    def test_dry_run_needs_only_standard_library_and_derives_all_gradient_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan.json"
            process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--dry-run", "--shapes", "7x5x3",
                                      "--roles", "forward", "dx", "dw", "--trials", "1", "--configs", "candidates",
                                      "--output", str(output)], capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text())
        jobs = {job["role"]: job for job in result["jobs"]}
        self.assertEqual(jobs["forward"]["a_shape"], [7, 5])
        self.assertEqual(jobs["forward"]["b_shape"], [3, 5])
        self.assertEqual(jobs["dx"]["a_shape"], [7, 3])
        self.assertEqual(jobs["dx"]["b_shape"], [5, 3])
        self.assertEqual(jobs["dw"]["a_shape"], [3, 7])
        self.assertEqual(jobs["dw"]["b_shape"], [5, 7])
        self.assertEqual(jobs["dw"]["routes"], ["dense", "materialized", "tiled"])
        self.assertTrue(result["fresh_process_per_role_shape_trial"])
        self.assertFalse(result["gpu_performance_qualified"])

    def test_invalid_dimensions_duplicates_and_measurements_reject_before_execution(self):
        for options in (("--shapes", "7x0x3"), ("--shapes", "2x3"),
                        ("--shapes", "7x5x3", "7x5x3"), ("--trials", "0"),
                        ("--warmup", "-1"), ("--timeout", "nan"),
                        ("--roles", "dw", "dw"), ("--max-configurations", "0"),
                        ("--min-measure-ms", "-1"), ("--min-measure-ms", "nan"),
                        ("--min-measure-ms", "inf"), ("--max-iterations", "49")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                benchmark.make_plan(self.arguments(*options))

    def test_measurement_window_options_reach_each_worker_plan(self):
        plan = benchmark.make_plan(self.arguments("--min-measure-ms", "50", "--max-iterations", "2000"))
        for job in plan["jobs"]:
            self.assertEqual(job["measurement"]["min_measure_ms"], 50)
            self.assertEqual(job["measurement"]["max_iterations"], 2000)

    def test_panel_is_explicit_and_preserves_every_original_job_and_control(self):
        baseline = benchmark.make_plan(self.arguments())
        panel = benchmark.make_plan(self.arguments("--panel-decoding"))
        self.assertEqual([row["id"] for row in baseline["jobs"]], [row["id"] for row in panel["jobs"]])
        for before, after in zip(baseline["jobs"], panel["jobs"]):
            self.assertEqual(after["group_size"], 1024)
            self.assertEqual(before["seed"], after["seed"])
            self.assertEqual(before["measurement"], after["measurement"])
            self.assertEqual(after["routes"], before["routes"] + (["panel"] if before["role"] == "dw" else []))

    def test_panel_forwards_bounded_route_and_keeps_independent_matching_codec_oracle(self):
        from deepopt.opt.backends.bf16 import product
        from deepopt.opt.quantization.activation import dequantize_unpack_activation

        calls = []

        def panel_reference(a, b, **kwargs):
            if kwargs.get("packed_route") == "panel":
                self.assertIsNone(b)
                self.assertEqual(kwargs["panel_size"], 1024)
                self.assertEqual(kwargs["packed_b"].group_size, 1024)
                calls.append(1)
                operand = dequantize_unpack_activation(kwargs["packed_b"], dtype=torch.bfloat16, transpose=True)
                return product(a, operand, backend="native", role="dw")
            return product(a, b, **kwargs)

        with patch("deepopt.opt.backends.bf16.product", side_effect=panel_reference):
            result = self.run_worker("dw", routes=["dense", "materialized", "tiled", "panel"])
        self.assertEqual(result["status"], "ok")
        self.assertTrue(calls)
        self.assertEqual({row["route"] for row in result["results"]}, {"dense", "materialized", "tiled", "panel"})
        self.assertFalse(result["gpu_performance_qualified"])

    def test_resource_rejection_keeps_structured_negative_evidence(self):
        class Rejected(RuntimeError):
            evidence = {"status": "rejected", "required": 344064, "limit": 232448, "route": "direct"}

        from deepopt.opt.backends.bf16 import product

        def reject(*args, **kwargs):
            if kwargs.get("backend") == "triton":
                raise Rejected("shared memory")
            return product(*args, **kwargs)

        with patch("deepopt.opt.backends.bf16.product", side_effect=reject):
            result = self.run_worker("forward")
        self.assertEqual(result["status"], "failed")
        rejected = next(row for row in result["results"] if row["status"] == "error")
        self.assertEqual(rejected["resource_rejection"], Rejected.evidence)

    def test_cuda_calibration_targets_total_window_and_excludes_warmup_and_pilot(self):
        calls, samples = [], []

        def timer(call, torch, device, warmup, iterations, repeats):
            samples.append((warmup, iterations, repeats))
            for _ in range(warmup + iterations * repeats):
                call()
            return {"cuda_stream_ms": {"median": 2., "samples": [2.] * repeats}}

        settings = {"warmup": 3, "iterations": 2, "repeats": 4,
                    "min_measure_ms": 80, "max_iterations": 100}
        timing = benchmark.measure_role(lambda: calls.append(1), None, SimpleNamespace(type="cuda"), settings, timer)
        self.assertEqual(samples, [(3, 2, 1), (0, 10, 4)])
        self.assertEqual(len(calls), 45)
        sampling = timing["sampling"]
        self.assertEqual(sampling["actual_cuda_event_total_ms"], 80)
        self.assertEqual(sampling["calibration_calls"], 2)
        self.assertEqual(sampling["calibration_cuda_event_total_ms"], 4)
        self.assertTrue(sampling["target_reached"])
        self.assertFalse(sampling["iteration_cap_reached"])
        self.assertFalse(sampling["calibration_in_reported_samples"])
        self.assertEqual(sampling["synchronization"], "repeat_boundaries_only")

    def test_cuda_calibration_is_bounded_and_reports_unmet_target(self):
        batches = []

        def timer(call, torch, device, warmup, iterations, repeats):
            batches.append((warmup, iterations, repeats))
            return {"cuda_stream_ms": {"median": .1, "samples": [.1] * repeats}}

        settings = {"warmup": 1, "iterations": 2, "repeats": 4,
                    "min_measure_ms": 80, "max_iterations": 10}
        timing = benchmark.measure_role(lambda: None, None, SimpleNamespace(type="cuda"), settings, timer)
        self.assertEqual(batches, [(1, 2, 1), (0, 10, 4)])
        self.assertTrue(timing["sampling"]["iteration_cap_reached"])
        self.assertFalse(timing["sampling"]["target_reached"])

    def test_achieved_window_rechecks_faster_than_pilot_and_preserves_sample_count(self):
        batches = []

        def timer(call, torch, device, warmup, iterations, repeats):
            batches.append((warmup, iterations, repeats))
            cost = 2. if len(batches) == 1 else .5
            return {"cuda_stream_ms": {"median": cost, "samples": [cost] * repeats}}

        timing = benchmark.measure_role(lambda: None, None, SimpleNamespace(type="cuda"),
            {"warmup": 3, "iterations": 2, "repeats": 4, "min_measure_ms": 80, "max_iterations": 100}, timer)
        self.assertEqual(batches[:2], [(3, 2, 1), (0, 10, 4)])
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(timing["cuda_stream_ms"]["samples"]), 4)
        self.assertTrue(timing["sampling"]["target_reached"])
        self.assertEqual(timing["sampling"]["discarded_window_attempts"][0]["cuda_event_total_ms"], 20)
        self.assertEqual(timing["sampling"]["additional_calibration_calls"], 40)

    def test_invalid_achieved_stream_window_is_not_reported_as_success(self):
        for samples in ([0.] * 4, [float("nan")] * 4, [1.], [float("inf")] * 4):
            calls = []

            def timer(*args):
                calls.append(1)
                return {"cuda_stream_ms": {"median": 2., "samples": [2.] if len(calls) == 1 else samples}}

            with self.subTest(samples=samples), self.assertRaisesRegex(RuntimeError, "every repeat"):
                benchmark.measure_role(lambda: None, None, SimpleNamespace(type="cuda"),
                    {"warmup": 0, "iterations": 2, "repeats": 4, "min_measure_ms": 80}, timer)

    def test_fixed_counts_preserve_cpu_work_and_allow_cuda_calibration_opt_out(self):
        for device, target, stream in (("cpu", 200, None), ("cuda", 0, {"samples": [2., 2.]})):
            with self.subTest(device=device):
                batches = []

                def timer(call, torch, device, warmup, iterations, repeats):
                    batches.append((warmup, iterations, repeats))
                    return {"cuda_stream_ms": stream}

                settings = {"warmup": 3, "iterations": 2, "repeats": 2,
                            "min_measure_ms": target, "max_iterations": 100}
                timing = benchmark.measure_role(lambda: None, None, SimpleNamespace(type=device), settings, timer)
                self.assertEqual(batches, [(3, 2, 2)])
                self.assertEqual(timing["sampling"]["mode"], "fixed_iterations")
                self.assertEqual(timing["sampling"]["calibration_calls"], 0)
                self.assertIsNone(timing["sampling"]["target_reached"])

    def test_cuda_calibration_rejects_invalid_event_measurements(self):
        for cost in (0., -1., float("nan"), float("inf")):
            with self.subTest(cost=cost), self.assertRaisesRegex(RuntimeError, "finite positive"):
                benchmark.measure_role(lambda: None, None, SimpleNamespace(type="cuda"),
                    {"warmup": 0, "iterations": 2, "repeats": 2, "min_measure_ms": 50},
                    lambda *args: {"cuda_stream_ms": {"median": cost}})

    def test_dw_codec_groups_follow_original_forward_input_and_match_independent_oracle(self):
        from deepopt.opt.backends.bf16 import product
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        torch.manual_seed(10)
        a, b, source = benchmark.build_operands(torch, (37, 35, 17), "dw", torch.device("cpu"))
        self.assertEqual(tuple(source.shape), (37, 35))
        self.assertEqual(tuple(b.shape), (35, 37))
        self.assertEqual(source.untyped_storage()._cdata, b.untyped_storage()._cdata)
        for bits in (1, 2, 4, 8):
            packed = quantize_pack_activation(source, 1024, bits=bits)
            reconstructed = dequantize_unpack_activation(packed, dtype=torch.bfloat16)
            # Deliberately use an independent matrix product, rather than the
            # same packed execution function as both candidate and oracle.
            expected = a @ reconstructed
            actual = product(a, None, packed_b=packed, backend="triton", role="dw")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def run_worker(self, role, **changes):
        args = self.arguments("--device", "cpu", "--roles", role, "--configs", "legacy", "--trials", "1",
                              "--warmup", "0", "--iterations", "1", "--repeats", "1")
        job = benchmark.make_plan(args)["jobs"][0]
        job.update(changes)
        # Other sessions may edit this shared checkout. This test verifies
        # execution, while source-change rejection is checked separately.
        with tempfile.TemporaryDirectory() as temporary, patch.object(benchmark, "source_identity", return_value={"stable": True}):
            sys.path.insert(0, str(ROOT / "benchmarks"))
            try:
                return benchmark.execute_worker(job, Path(temporary))
            finally:
                sys.path.pop(0)

    def test_cpu_worker_exercises_all_roles_without_claiming_generated_launches(self):
        for role, count in (("forward", 2), ("dx", 2), ("dw", 6)):
            with self.subTest(role=role):
                result = self.run_worker(role)
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(len(result["results"]), count)
                self.assertTrue(all(row["correctness"] == "pass" for row in result["results"]))
                self.assertTrue(all(row["execution"] == "cpu_semantic_reference" for row in result["results"]))
                self.assertFalse(any(row["generated_launch_observed"] for row in result["results"]))
                self.assertFalse(result["gpu_performance_qualified"])
                self.assertFalse(result["packing_in_timed_region"])
                json.dumps(result)

    def test_failed_generated_configuration_is_retained_alongside_passing_native_control(self):
        from deepopt.opt.backends.bf16 import product

        def fail_generated(*args, **kwargs):
            if kwargs.get("backend") == "triton":
                raise RuntimeError("test generated failure")
            return product(*args, **kwargs)

        with patch("deepopt.opt.backends.bf16.product", side_effect=fail_generated):
            result = self.run_worker("forward")
        self.assertEqual(result["status"], "failed")
        records = {row["backend"]: row for row in result["results"]}
        self.assertEqual(records["native"]["status"], "ok")
        self.assertEqual(records["triton"]["status"], "error")
        self.assertIn("test generated failure", records["triton"]["error"])

    def test_bounded_role_candidates_execute_as_cpu_references_only(self):
        result = self.run_worker("dw", configs="candidates", max_configurations=2)
        self.assertEqual(result["status"], "ok", result)
        generated = [row for row in result["results"] if row["backend"] == "triton"]
        # The bounded table can reject redundant large tiles for this tiny
        # shape; the cap is an upper bound, never an obligation to launch them.
        self.assertGreaterEqual(len(generated), 3)
        self.assertLessEqual(len(generated), 6)
        self.assertTrue(all(row["execution"] == "cpu_semantic_reference" for row in generated))
        self.assertEqual(result["packed_source_shape"], (37, 35))
        self.assertTrue(all(row["diagnostics"][0]["role"] == "dw" for row in generated))

    def test_changed_source_rejects_otherwise_passing_worker(self):
        args = self.arguments("--device", "cpu", "--roles", "forward", "--configs", "legacy", "--trials", "1",
                              "--warmup", "0", "--iterations", "1", "--repeats", "1")
        job = benchmark.make_plan(args)["jobs"][0]
        with tempfile.TemporaryDirectory() as temporary, patch.object(benchmark, "source_identity", side_effect=[{"a": 1}, {"a": 2}]):
            sys.path.insert(0, str(ROOT / "benchmarks"))
            try:
                result = benchmark.execute_worker(job, Path(temporary))
            finally:
                sys.path.pop(0)
        self.assertTrue(all(row["status"] == "ok" for row in result["results"]))
        self.assertFalse(result["source_unchanged"])
        self.assertEqual(result["status"], "failed")

    def test_worker_rejects_different_installed_distribution_before_operands(self):
        with (patch.object(benchmark, "source_identity", return_value={"installed_deepopt": {"version": "first"}}),
              patch.object(benchmark, "build_operands") as build):
            with self.assertRaisesRegex(RuntimeError, "differs from its supervisor"):
                benchmark.execute_worker({"installed_deepopt": {"version": "second"}}, Path("unused"))
        build.assert_not_called()

    def test_supervisor_pins_installation_and_rejects_different_worker_identity(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        try:
            from compiler import installed_runtime

            identity = {"version": "installed", "package_path": "/environment/site-packages/deepopt",
                        "source_sha256": "a" * 64}
            child_environment = {"PYTHONPATH": "/environment/site-packages", "PINNED": "yes"}

            def child(command, **kwargs):
                self.assertEqual(kwargs["env"], child_environment)
                output = Path(command[command.index("--output") + 1])
                job = json.loads(Path(command[command.index("--_worker") + 1]).read_text())
                self.assertEqual(job["installed_deepopt"], identity)
                benchmark.write_json(output, {"status": "ok", "source_before": {
                    "installed_deepopt": {**identity, "version": "different"}}, "source_after": {
                    "installed_deepopt": identity}, "results": []})
                return SimpleNamespace(returncode=0)

            with (tempfile.TemporaryDirectory() as temporary,
                  patch.object(installed_runtime, "prepare_imports"),
                  patch.object(installed_runtime, "installed_identity", return_value=identity),
                  patch.object(installed_runtime, "child_environment", return_value=child_environment),
                  patch.object(benchmark.subprocess, "run", side_effect=child)):
                output = Path(temporary) / "results.json"
                status = benchmark.main(["--device", "cpu", "--roles", "forward", "--shapes", "7x5x3",
                                         "--trials", "1", "--output", str(output)])
                result = json.loads(output.read_text())
            self.assertEqual(status, 1)
            self.assertFalse(result["passed"])
            self.assertEqual(result["installed_deepopt"], identity)
            self.assertIn("differs from its supervisor", result["results"][0]["error"])
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
