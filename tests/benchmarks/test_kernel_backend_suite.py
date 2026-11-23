"""Backend matrix, report validity and isolated-worker failure handling."""

import importlib.util
import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_kernel_backends.py"
spec = importlib.util.spec_from_file_location("kernel_backend_suite", SCRIPT)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)

ACTIVATION_FIXTURE_CLI = """import argparse
fixture_parser = argparse.ArgumentParser()
fixture_parser.add_argument('--shape')
fixture_parser.add_argument('--activations', nargs='+')
fixture_parser.add_argument('--device')
fixture_parser.add_argument('--warmup')
fixture_parser.add_argument('--iterations')
fixture_parser.add_argument('--repeats')
fixture_parser.add_argument('--deepopt-backend')
fixture_parser.add_argument('--deepopt-kernel-policy')
fixture_parser.add_argument('--time-budget')
fixture_parser.add_argument('--autotune', action=argparse.BooleanOptionalAction)
fixture_parser.add_argument('--output')
fixture_arguments = fixture_parser.parse_args()
"""


class KernelBackendSuiteTests(unittest.TestCase):
    def test_cli_preflight_rejects_the_stale_six_worker_bundle_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / "stale"
            stale.mkdir()
            source_files = ("fp8/benchmark_fp8_eager.py", "normalization/benchmark_normalization.py",
                            "pooling/benchmark_pooling.py")
            for relative in source_files:
                original = SCRIPT.parent / relative
                text = original.read_text()
                removed = set()
                for node in ast.walk(ast.parse(text)):
                    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "add_argument"
                            and any(isinstance(argument, ast.Constant) and argument.value in ("--time-budget", "--autotune")
                                    for argument in node.value.args)):
                        removed.update(range(node.lineno, node.end_lineno + 1))
                # An executable probe would fail; the preflight must only read
                # the exact old argparse declarations that caused six exits.
                old_source = "raise AssertionError('preflight executed a worker')\n" + "".join(
                    line for number, line in enumerate(text.splitlines(keepends=True), 1) if number not in removed)
                (stale / original.name).write_text(old_source)
            options = ["--benchmark-root", str(stale), "--preset", "representative", "--workloads",
                       "linear", "layer_norm", "pooling", "--backends", "triton", "tilelang",
                       "--warmup", "10", "--iterations", "50", "--repeats", "5", "--trials", "1"]
            self.assertEqual(len(suite.matrix(suite.parser().parse_args(options))), 6)
            with patch.object(suite, "run_provenance") as provenance, patch.object(suite, "_lease_scheduler") as gpu, \
                 patch.object(suite, "execute") as execute, patch.object(suite.subprocess, "Popen") as launch, \
                 patch.dict(sys.modules, {"torch": None}):
                for dry_run in (False, True):
                    output = root / ("dry" if dry_run else "run")
                    diagnostic = StringIO()
                    with self.subTest(dry_run=dry_run), redirect_stderr(diagnostic), self.assertRaises(SystemExit) as failure:
                        suite.main([*options, "--output", str(output), *(["--dry-run"] if dry_run else [])])
                    self.assertEqual(failure.exception.code, 2)
                    message = diagnostic.getvalue()
                    self.assertIn("--time-budget", message)
                    self.assertIn("--no-autotune", message)
                    self.assertIn("complete benchmark bundle", message)
                    for relative in source_files:
                        self.assertEqual(message.count(str(stale / Path(relative).name)), 1)
                    self.assertFalse(output.exists())
                for operation in (provenance, gpu, execute, launch):
                    operation.assert_not_called()

    def test_cli_preflight_checks_each_selected_file_once_and_accepts_fresh_commands(self):
        with patch.dict(sys.modules, {"torch": None}):
            for preset in ("smoke", "representative"):
                cases = suite.matrix(suite.parser().parse_args(["--preset", preset, "--trials", "2"]))
                original = suite._script_contract
                with patch.object(suite, "_script_contract", wraps=original) as inspect:
                    suite.preflight_cli(cases)
                self.assertEqual(inspect.call_count, len({case["command"][1] for case in cases}))
            # LayerNorm, BatchNorm and GroupNorm share one script even though
            # their backend/trial commands differ.
            cases = suite.matrix(suite.parser().parse_args(["--workloads", "layer_norm", "batch_norm", "group_norm"]))
            with patch.object(suite, "_script_contract", wraps=original) as inspect:
                suite.preflight_cli(cases)
            inspect.assert_called_once()

    def test_cli_capabilities_support_boolean_aliases_and_ignore_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "worker.py"
            script.write_text("""import argparse as ap
from argparse import BooleanOptionalAction as Bool
raise AssertionError('inspection must not execute the worker')
p = ap.ArgumentParser()
p.add_argument('--autotune', action=Bool)
p.add_argument('--bias', action=ap.BooleanOptionalAction)
p.add_argument('--store', action='store_true')
text = "p.add_argument('--not-real')"
""")
            capabilities = suite.cli_capabilities(script)
            self.assertEqual(capabilities, {"--autotune", "--no-autotune", "--bias", "--no-bias", "--store"})

    def test_cli_preflight_requires_only_selected_build_helper_apis(self):
        from deepopt import _benchmarking

        with patch.dict(sys.modules, {"torch": None}), patch.dict(_benchmarking.__dict__):
            for name in ("plan_timing", "timing_plan_from_cost", "capture_warmup_steps", "share_timing_counts"):
                _benchmarking.__dict__.pop(name, None)
            # Linear has its local timing implementation and needs none of
            # these APIs; a stale installed helper must not reject this case.
            suite.preflight_cli(suite.matrix(suite.parser().parse_args(["--workloads", "linear"])))
            cases = suite.matrix(suite.parser().parse_args(["--workloads", "layer_norm", "convolution", "compiler"]))
            with self.assertRaisesRegex(ValueError, "selected DeepOpt build is missing") as failure:
                suite.preflight_cli(cases)
            for name in ("plan_timing", "timing_plan_from_cost", "capture_warmup_steps", "share_timing_counts"):
                self.assertIn(name, str(failure.exception))
            with patch.object(suite, "run_provenance") as provenance, patch.object(suite, "_lease_scheduler") as gpu, \
                 redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                suite.main(["--workloads", "layer_norm", "--dry-run"])
            provenance.assert_not_called()
            gpu.assert_not_called()

    def test_pack_preflight_validates_its_selected_adjacent_timing_dependency(self):
        from deepopt import _benchmarking

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": None}):
            root = Path(directory)
            pack_source = SCRIPT.parent / suite.WORKLOADS["pack"][0]
            (root / pack_source.name).write_bytes(pack_source.read_bytes())
            helper = root / Path(suite.WORKLOADS["unpack"][0]).name

            def helper_source(omit=()):
                return ("raise AssertionError('preflight executed a helper')\n"
                        "from deepopt._benchmarking import plan_timing as shared_timing\n" + "\n".join(
                            f"def {name}(*args, **kwargs): pass" for name in sorted(suite.PACK_HELPER_API - set(omit))))

            args = suite.parser().parse_args(["--benchmark-root", str(root), "--workloads", "pack"])
            cases = suite.matrix(args)
            helper.write_text(helper_source(omit=("codec_schedule",)))
            with self.assertRaisesRegex(ValueError, "missing shared functions codec_schedule") as missing:
                suite.preflight_cli(cases)
            self.assertIn(str(helper), str(missing.exception))
            helper.write_text(helper_source())
            suite.preflight_cli(cases)
            with patch.dict(_benchmarking.__dict__):
                _benchmarking.__dict__.pop("plan_timing")
                with self.assertRaisesRegex(ValueError, "plan_timing"):
                    suite.preflight_cli(cases)
            helper.unlink()
            with self.assertRaisesRegex(ValueError, "cannot inspect CLI declarations"):
                suite.preflight_cli(cases)

    def test_representative_budget_keeps_requested_matrix_and_exhaustive_caps(self):
        options = ["--preset", "representative", "--workloads", "linear", "layer_norm", "pooling",
                   "--backends", "triton", "tilelang", "--warmup", "10", "--iterations", "50",
                   "--repeats", "5", "--trials", "3", "--timeout", "900"]
        args = suite.parser().parse_args(options)
        cases = suite.matrix(args)
        self.assertEqual(len(cases), 18)
        plan = suite.planned_work(args, cases)
        self.assertEqual(plan["estimated_measurement_calls"], 144)
        self.assertEqual(plan["estimated_measurement_budget_seconds"], 288)
        self.assertEqual(plan["requested_steps_per_measurement_cap"], 260)
        for case in cases:
            command = case["command"]
            for flag, value in (("--warmup", "10"), ("--iterations", "50"), ("--repeats", "5"), ("--time-budget", "2.0")):
                self.assertEqual(command[command.index(flag) + 1], value)
            if case["workload"] == "linear":
                self.assertIn("4096x768x3072", command)
                self.assertIn("4096x3072x768", command)
                self.assertIn("--no-autotune", command)
        exhaustive = suite.matrix(suite.parser().parse_args([*options, "--exhaustive", "--autotune"]))
        self.assertEqual(len(exhaustive), 18)
        self.assertTrue(all(row["command"][row["command"].index("--time-budget") + 1] == "0.0" for row in exhaustive))
        self.assertTrue(all("--autotune" in row["command"] for row in exhaustive if row["workload"] == "linear"))

    def test_matrix_keeps_policies_seeded_order_and_independent_trials(self):
        args = suite.parser().parse_args(["--workloads", "activation", "--trials", "3"])
        first, second = suite.matrix(args), suite.matrix(args)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertEqual({item["backend"] for item in first}, {"triton", "tilelang", "mixed"})
        self.assertNotEqual([item["backend"] for item in first[:3]], [item["backend"] for item in first[3:6]])
        for row in first:
            self.assertEqual(row["policy"]["backend"], row["backend"])
            self.assertIn(row["backend"], row["command"])
            self.assertIn("--output", row["command"])
            if row["backend"] == "mixed":
                self.assertEqual(row["policy"]["families"]["gemm"], "triton")
                self.assertIn("--deepopt-kernel-policy", row["command"])
        compiler = suite.matrix(suite.parser().parse_args(["--workloads", "compiler"]))
        self.assertTrue(all("--steps" in row["command"] and "--repeats" not in row["command"] for row in compiler))

    def test_cpu_matrix_is_reference_and_never_initializes_torch(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": None}):
            output = Path(directory) / "dry"
            self.assertEqual(suite.main(["--device", "cpu", "--dry-run", "--output", str(output)]), 0)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["execution"], "cpu_reference")
            self.assertEqual(report["results"], [])
            self.assertFalse(any(item["gpu_only"] for item in report["matrix"]))
            self.assertEqual({item["backend"] for item in report["matrix"]}, {"triton", "tilelang", "mixed"})

    def test_failed_or_nonfinite_reports_cannot_supply_valid_timings(self):
        for report in ({"results": [{"status": "error"}]}, {"cases": [{"exact": False}]},
                       {"numerical_parity": False}, {"error": float("nan")}, {"metrics": {"finite": False}},
                       {"error": "compiler failure", "timing": {"wall_ms_median": 1.}}):
            with self.subTest(report=report):
                self.assertTrue(suite.invalid_report(report))
        self.assertFalse(suite.invalid_report({"results": [{"exact": True, "timing": {"wall_ms_median": 1.}}]}))

    def test_comparison_matches_semantic_cases_and_excludes_native_rows(self):
        def report(shape, latency):
            return {"results": [
                {"implementation": "native_bf16", "timing": {"wall_ms_median": 100.}},
                {"implementation": "deepopt", "shape": shape, "compression": False,
                 "timing": {"eager": {"wall_ms_median": latency}}}]}
        baseline = {"workload": "activation", "trial": 0, "backend": "triton", "status": "ok",
                    "measurements": suite.measurements(report([3, 33], 1.))}
        candidate = {**baseline, "backend": "tilelang", "measurements": suite.measurements(report([3, 33], 2.))}
        self.assertEqual(len(baseline["measurements"]), 1)
        comparison = suite.comparisons([baseline, candidate])[0]
        self.assertEqual(comparison["warm_wall_ratio_geomean"], 2.)
        self.assertEqual(comparison["warm_wall_cases_above_1_25"], 1)
        mismatch = {**candidate, "measurements": suite.measurements(report([4, 33], 2.))}
        comparison = suite.comparisons([baseline, mismatch])[0]
        self.assertEqual(comparison["matching_metrics"], 0)
        self.assertTrue(comparison["unmatched_candidate_metrics"])
        self.assertEqual(suite.comparisons([baseline, {**candidate, "status": "error"}]), [])
        repeated = report([3, 33], 1.)
        repeated["results"].append(repeated["results"][-1])
        with self.assertRaisesRegex(ValueError, "Ambiguous repeated"):
            suite.measurements(repeated)

    def test_real_worker_success_nonzero_exit_and_timeout_keep_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text("""import json, sys, time
from pathlib import Path
if sys.argv[2] == 'timeout':
    time.sleep(10)
Path(sys.argv[1]).write_text(json.dumps({'results': [{'timing': {'wall_ms_median': 1.}}]}))
print('worker diagnostics', flush=True)
raise SystemExit(3 if sys.argv[2] == 'error' else 0)
""")
            for mode in ("ok", "error", "timeout"):
                case = {"report": str(root / (mode + ".json")),
                        "command": [sys.executable, str(worker), str(root / (mode + ".json")), mode]}
                row = suite.execute(case, .2 if mode == "timeout" else 10)
                self.assertEqual(row["status"], mode)
                self.assertTrue(Path(row["stdout"]).is_file())
                self.assertTrue(Path(row["stderr"]).is_file())
                if mode == "ok":
                    self.assertTrue(row["measurements"])
                else:
                    self.assertIn("error", row)

    @unittest.skipUnless(os.name == "posix", "GPU scheduler owns POSIX worker process groups")
    def test_gpu_adapter_inherits_one_lease_rebases_device_and_combines_logs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": None}):
            root = Path(directory)
            worker, report = root / "worker.py", root / "report.json"
            worker.write_text("""import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from analysis import scheduler
assert scheduler.inherited_lease()
def detached_again(*args, **kwargs):
    raise AssertionError('an inherited lease must not create another coordinator')
scheduler.run_external = detached_again
scheduler.discover_devices = detached_again
assert scheduler.ensure_lease(sys.argv[2:], __file__, 'cuda:0') is None
Path(sys.argv[2]).write_text(json.dumps({
    'timing': {'wall_ms_median': 1.}, 'device': sys.argv[4],
    'visible': os.environ['CUDA_VISIBLE_DEVICES'],
    'lease': os.environ['DEEPOPT_BENCHMARK_LEASE']}))
print('standard output', flush=True)
print('standard error', file=sys.stderr, flush=True)
""")
            scheduler = suite._lease_scheduler(SCRIPT.parent)
            device = scheduler.Device("GPU-ADAPTER-TEST", "7", 2**30, "control-plane fixture")
            case = {"device": "cuda:7", "benchmark_root": str(SCRIPT.parent), "report": str(report),
                    "command": [sys.executable, str(worker), str(SCRIPT.parent), str(report), "--device", "cuda:7"]}
            broker_path = root / "leases.sqlite3"
            with patch.object(scheduler, "discover_devices", return_value=[device]) as discover, \
                 patch.object(scheduler.Broker, "telemetry", return_value=(2**30, False)), \
                 patch.dict(os.environ, {"DEEPOPT_BENCHMARK_LEASE_DB": str(broker_path)}):
                row = suite.execute(case, 10)
            self.assertEqual(row["status"], "ok", row)
            discover.assert_called_once_with(None, "cuda:7")
            self.assertEqual(case["command"][-1], "cuda:7")
            self.assertEqual(row["executed_command"][-1], "cuda:0")
            self.assertEqual(row["device_binding"]["physical_uuid"], device.uuid)
            self.assertEqual(row["log_format"], "combined_stdout_stderr")
            self.assertIsNone(row["stdout"])
            self.assertIsNone(row["stderr"])
            self.assertIn("standard output", Path(row["combined_log"]).read_text())
            self.assertIn("standard error", Path(row["combined_log"]).read_text())
            actual = json.loads(report.read_text())
            self.assertEqual((actual["device"], actual["visible"]), ("cuda:0", device.uuid))
            self.assertTrue(actual["lease"])
            with scheduler.Broker(broker_path).connect() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0], 0)
            self.assertEqual(suite._logical_cuda_command(["worker", "--device=cuda:7"]),
                             ["worker", "--device=cuda:0"])

    @unittest.skipUnless(os.name == "posix", "GPU scheduler owns POSIX worker process groups")
    def test_gpu_adapter_timeout_reaps_nested_worker_before_releasing_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker, marker = root / "worker.py", root / "nested.pid"
            worker.write_text("""import signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
def terminate(signum, frame):
    child.wait(timeout=3)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, terminate)
Path(sys.argv[1]).write_text(str(child.pid))
print('nested worker started', flush=True)
time.sleep(30)
""")
            scheduler = suite._lease_scheduler(SCRIPT.parent)
            device = scheduler.Device("GPU-TIMEOUT-TEST", "3", 2**30, "control-plane fixture")
            broker_path = root / "leases.sqlite3"
            case = {"device": "cuda:3", "benchmark_root": str(SCRIPT.parent), "report": str(root / "report.json"),
                    "command": [sys.executable, str(worker), str(marker), "--device=cuda:3"]}
            with patch.object(scheduler, "discover_devices", return_value=[device]), \
                 patch.object(scheduler.Broker, "telemetry", return_value=(2**30, False)), \
                 patch.dict(os.environ, {"DEEPOPT_BENCHMARK_LEASE_DB": str(broker_path)}):
                row = suite.execute(case, 1)
            self.assertEqual(row["status"], "timeout", row)
            self.assertLess(row["process_wall_seconds"], 7)
            self.assertTrue(marker.exists(), Path(row["combined_log"]).read_text())
            nested_pid = int(marker.read_text())
            try:
                with self.assertRaises(ProcessLookupError):
                    os.kill(nested_pid, 0)
            finally:
                try:
                    os.kill(nested_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            with scheduler.Broker(broker_path).connect() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0], 0)

    def test_gpu_control_plane_failure_is_retained_as_a_failed_row(self):
        scheduler = suite._lease_scheduler(SCRIPT.parent)
        failures = (RuntimeError("nvidia-smi failed"), subprocess.TimeoutExpired("nvidia-smi", 10))
        with tempfile.TemporaryDirectory() as directory:
            case = {"device": "cuda:0", "benchmark_root": str(SCRIPT.parent),
                    "report": str(Path(directory) / "report.json"), "command": ["unreachable"]}
            for failure in failures:
                with self.subTest(error=type(failure).__name__), \
                     patch.object(scheduler, "discover_devices", side_effect=failure), \
                     patch.object(scheduler, "run_external") as launch:
                    row = suite.execute(case, 1)
                self.assertEqual(row["status"], "error")
                self.assertIn("nvidia-smi", row["error"])
                launch.assert_not_called()

    def test_runner_reuses_matching_inherited_lease_without_releasing_its_owner(self):
        scheduler = suite._lease_scheduler(SCRIPT.parent)
        device = scheduler.Device("GPU-INHERITED-TEST", "5", 2**30)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, broker_path = root / "report.json", root / "leases.sqlite3"
            broker = scheduler.Broker(broker_path)
            token = broker.acquire(device, None, True, 1, 0)
            case = {"device": "cuda:0", "benchmark_root": str(SCRIPT.parent), "report": str(report),
                    "command": ["worker", "--device", "cuda:0"]}

            def run_command(command, *, env, log, timeout, cancel_file):
                self.assertEqual(env["DEEPOPT_BENCHMARK_LEASE"], token)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], device.uuid)
                report.write_text(json.dumps({"timing": {"wall_ms_median": 1.}}))
                return 0

            try:
                with patch.dict(os.environ, {"DEEPOPT_BENCHMARK_LEASE": token,
                                              "DEEPOPT_BENCHMARK_LEASE_DB": str(broker_path)}), \
                     patch.object(scheduler, "discover_devices", return_value=[device]), \
                     patch.object(scheduler, "run_external") as acquire_again, \
                     patch.object(scheduler, "run_command", side_effect=run_command) as launch:
                    row = suite.execute(case, 1)
                    self.assertEqual(row["status"], "ok", row)
                    self.assertEqual(row["gpu_lease"], "inherited")
                    launch.assert_called_once()
                    acquire_again.assert_not_called()
                    self.assertEqual(broker.members(device), [token])
                    with patch.object(scheduler, "discover_devices", return_value=[scheduler.Device("GPU-OTHER", "6", 2**30)]):
                        mismatch = suite.execute(case, 1)
                    self.assertEqual(mismatch["status"], "error")
                    self.assertIn("does not cover", mismatch["error"])
                    self.assertEqual(launch.call_count, 1)
            finally:
                broker.release(token)

    def test_linear_nested_timings_and_compiler_step_phase_metrics(self):
        linear = {"results": [{"implementation": "deepopt", "status": "ok", "cases": [{
            "shape_m_k_n": [32, 64, 96], "mode": "forward-backward", "status": "ok",
            "timing": {"wall_ms": {"median": 2., "samples": [2.]},
                       "cuda_stream_ms": {"median": 1., "samples": [1.]}}}]}]}
        actual = suite.measurements(linear)
        self.assertEqual(len(actual), 2)
        self.assertTrue(any(key.endswith("/wall_ms/median") for key in actual))
        baseline = {"workload": "linear", "trial": 0, "backend": "triton", "status": "ok", "measurements": actual}
        candidate = {**baseline, "backend": "tilelang", "measurements": {key: value * 2 for key, value in actual.items()}}
        self.assertEqual(suite.comparisons([baseline, candidate])[0]["warm_wall_ratio_geomean"], 2.)
        compiler = {"cases": [{"backend": "eager", "timings": {"step_phases_ms": {"median": 100.}}},
                              {"backend": "deepopt", "timings": {"step_phases_ms": {"median": 2.}}}]}
        metrics = suite.measurements(compiler)
        self.assertEqual(len(metrics), 1)
        baseline = {**baseline, "workload": "compiler", "measurements": metrics}
        candidate = {**baseline, "backend": "tilelang", "measurements": {key: value * 2 for key, value in metrics.items()}}
        self.assertEqual(suite.comparisons([baseline, candidate])[0]["warm_step_phase_ratios"], [2.])

    def test_budget_limited_comparisons_retain_actual_counts_and_caution(self):
        def report(counts, exhausted, latency):
            return {"results": [{"implementation": "deepopt", "activation_compression": True,
                                 "timing": {"eager": {"wall_ms_median": latency, "sampling": {
                                     "requested_iterations": 50, "requested_repeats": 5,
                                     "actual_iterations_per_repeat": counts, "actual_repeats": len(counts),
                                     "actual_warmup": 10, "time_budget_seconds": 2., "budget_exhausted": exhausted}}}}]}
        baseline_report, candidate_report = report([50] * 5, False, 1.), report([2, 1], True, 2.)
        baseline = {"workload": "convolution", "trial": 0, "backend": "triton", "status": "ok",
                    "measurements": suite.measurements(baseline_report), "sampling": suite.sampling_records(baseline_report)}
        candidate = {**baseline, "backend": "tilelang", "measurements": suite.measurements(candidate_report),
                     "sampling": suite.sampling_records(candidate_report)}
        comparison = suite.comparisons([baseline, candidate])[0]
        self.assertEqual(comparison["warm_wall_ratio_geomean"], 2.)
        self.assertEqual(len(comparison["sampling"]["different_effective_counts"]), 1)
        self.assertEqual(comparison["sampling"]["candidate"]["budget_exhausted_measurements"], 1)
        self.assertEqual(comparison["sampling"]["candidate"]["minimum_actual_repeats"], 2)
        both = {"results": [baseline_report["results"][0], {**baseline_report["results"][0], "activation_compression": False}]}
        self.assertEqual(len(suite.measurements(both)), 2)

    def test_resume_retries_failed_workers_and_verifies_completed_report_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root, output = root / "benchmarks", root / "results"
            benchmark_root.mkdir()
            worker = benchmark_root / "benchmark_activations.py"
            worker.write_text(ACTIVATION_FIXTURE_CLI + """import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path)
p.add_argument('--deepopt-backend')
a, _ = p.parse_known_args()
root = Path(__file__).parent
with (root / 'launches.log').open('a') as f: f.write(a.deepopt_backend + '\\n')
if a.deepopt_backend == 'tilelang' and not (root / 'failed.marker').exists():
    (root / 'failed.marker').touch()
    raise SystemExit(3)
a.output.write_text(json.dumps({'timing': {'wall_ms_median': 1.}}))
""")
            options = ["--benchmark-root", str(benchmark_root), "--device", "cpu", "--workloads", "activation",
                       "--backends", "triton", "tilelang", "--output", str(output)]
            messages = StringIO()
            with redirect_stdout(messages):
                self.assertEqual(suite.main(options), 1)
                first_identity = json.loads((output / "summary.json").read_text())["run_identity"]
                self.assertEqual(suite.main([*options, "--resume", "--heartbeat-interval", "0"]), 0)
            launches = (benchmark_root / "launches.log").read_text().splitlines()
            self.assertEqual((launches.count("triton"), launches.count("tilelang")), (1, 2))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["run_identity"], first_identity)
            rows = {row["backend"]: row for row in summary["results"]}
            self.assertTrue(rows["triton"]["reused"])
            self.assertFalse(rows["tilelang"]["reused"])
            self.assertIn("reused verified success", messages.getvalue())
            self.assertIn(": error in", messages.getvalue())
            self.assertEqual(summary["progress"]["pending_workers"], 0)
            # A complete per-worker checkpoint can outlive the coordinator's
            # final summary update; neither successful worker is relaunched.
            summary["results"] = []
            (output / "summary.json").write_text(json.dumps(summary))
            with redirect_stdout(StringIO()):
                self.assertEqual(suite.main([*options, "--resume"]), 0)
            self.assertEqual((benchmark_root / "launches.log").read_text().splitlines(), launches)
            # Even finite, valid JSON must rerun after report-byte tampering.
            Path(rows["triton"]["report"]).write_text(json.dumps({"timing": {"wall_ms_median": 999.}}))
            with redirect_stdout(StringIO()):
                self.assertEqual(suite.main([*options, "--resume"]), 0)
            launches = (benchmark_root / "launches.log").read_text().splitlines()
            self.assertEqual((launches.count("triton"), launches.count("tilelang")), (2, 2))

    def test_heartbeat_reports_live_worker_log_and_stops_on_completion_and_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root, output = root / "benchmarks", root / "results"
            benchmark_root.mkdir()
            release = benchmark_root / "release.marker"
            (benchmark_root / "benchmark_activations.py").write_text(ACTIVATION_FIXTURE_CLI + """import json, sys, time
from pathlib import Path
root = Path(__file__).parent
print('cold compile fixture ready', file=sys.stderr, flush=True)
deadline = time.monotonic() + 3
while not (root / 'release.marker').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('the heartbeat did not release the waiting fixture')
    time.sleep(.005)
Path(sys.argv[sys.argv.index('--output') + 1]).write_text(json.dumps({'timing': {'wall_ms_median': 1.}}))
""")

            class LiveOutput(StringIO):
                def write(self, text):
                    size = super().write(text)
                    if "still running:" in text and "cold compile fixture ready" in text:
                        release.touch()
                    return size

            messages = LiveOutput()
            with redirect_stdout(messages):
                self.assertEqual(suite.main(["--benchmark-root", str(benchmark_root), "--device", "cpu",
                                             "--workloads", "activation", "--backends", "triton",
                                             "--heartbeat-interval", ".01", "--output", str(output)]), 0)
                captured = messages.getvalue()
                threading.Event().wait(.04)
                self.assertEqual(messages.getvalue(), captured)
            self.assertIn("latest log: cold compile fixture ready", captured)
            self.assertLess(captured.index("still running:"), captured.index(": ok in"))
            self.assertFalse(any(thread.name == "deepopt-worker-heartbeat" for thread in threading.enumerate()))
            case = {"id": "exception", "report": str(root / "exception.json"), "device": "cpu"}
            with self.assertRaisesRegex(RuntimeError, "fixture exception"):
                with suite.worker_heartbeat(case, .01):
                    raise RuntimeError("fixture exception")
            self.assertFalse(any(thread.name == "deepopt-worker-heartbeat" for thread in threading.enumerate()))
            with patch.object(suite.threading, "Thread") as start:
                with suite.worker_heartbeat(case, 0):
                    pass
                start.assert_not_called()

    def test_heartbeat_log_tail_is_short_and_tolerates_missing_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"report": str(root / "worker.json"), "device": "cuda:0"}
            self.assertIsNone(suite.worker_log_tail(case))
            (root / "worker.combined.log").write_bytes(b"old content\n" * 1000 + b"latest " + b"x" * 200 + b"\n")
            latest = suite.worker_log_tail(case)
            self.assertTrue(latest.startswith("latest "))
            self.assertEqual(len(latest), 180)

    def test_resume_rejects_changes_before_launch_or_summary_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            options = ["--device", "cpu", "--workloads", "activation", "--backends", "triton",
                       "--output", str(output), "--dry-run"]
            provenance = {"resume_eligible": True, "deepopt": "build-a", "packages": {"torch": "a"}, "device": "cpu-a"}
            with patch.object(suite, "run_provenance", return_value=provenance), redirect_stdout(StringIO()):
                self.assertEqual(suite.main(options), 0)
            original = (output / "summary.json").read_bytes()
            changes = (["--iterations", "11"], ["--time-budget", "1"], ["--backends", "tilelang"],
                       ["--backends", "mixed", "--mixed-policy", '{"preferred":"triton"}'])
            with patch.object(suite.subprocess, "Popen") as launch, redirect_stdout(StringIO()):
                for changed in changes:
                    with self.subTest(changed=changed), patch.object(suite, "run_provenance", return_value=provenance), \
                         self.assertRaises(SystemExit):
                        suite.main([*options, "--resume", *changed])
                for field, value in (("deepopt", "build-b"), ("packages", {"torch": "b"}), ("device", "cpu-b")):
                    with self.subTest(changed=field), patch.object(suite, "run_provenance", return_value={**provenance, field: value}), \
                         self.assertRaises(SystemExit):
                        suite.main([*options, "--resume"])
                launch.assert_not_called()
            self.assertEqual((output / "summary.json").read_bytes(), original)

    def test_resume_retry_cannot_reuse_stale_json_when_new_worker_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root, output = root / "benchmarks", root / "results"
            benchmark_root.mkdir()
            (benchmark_root / "benchmark_activations.py").write_text(ACTIVATION_FIXTURE_CLI + """import json, sys
from pathlib import Path
if not (Path(__file__).parent / 'skip.marker').exists():
    Path(sys.argv[sys.argv.index('--output') + 1]).write_text(json.dumps({'timing': {'wall_ms_median': 1.}}))
""")
            options = ["--benchmark-root", str(benchmark_root), "--device", "cpu", "--workloads", "activation",
                       "--backends", "triton", "--output", str(output)]
            with redirect_stdout(StringIO()):
                self.assertEqual(suite.main(options), 0)
            checkpoint_path = output / "activation-triton-00.checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            checkpoint["status"] = "error"
            checkpoint_path.write_text(json.dumps(checkpoint))
            (benchmark_root / "skip.marker").touch()
            with redirect_stdout(StringIO()):
                self.assertEqual(suite.main([*options, "--resume"]), 1)
            row = json.loads((output / "summary.json").read_text())["results"][0]
            self.assertEqual(row["status"], "error")
            self.assertIn("Invalid worker report", row["error"])
            self.assertFalse(Path(row["report"]).exists())

    def test_source_identity_detects_edits_and_excludes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "helper.py", root / "results"
            source.write_text("VERSION = 1\n")
            before = suite._source_fingerprint(root, exclude=(output,))
            output.mkdir()
            (output / "summary.json").write_text('{"results": []}')
            self.assertEqual(suite._source_fingerprint(root, exclude=(output,)), before)
            source.write_text("VERSION = 2\n")
            self.assertNotEqual(suite._source_fingerprint(root, exclude=(output,))["sha256"], before["sha256"])
            with suite.output_lock(output), self.assertRaisesRegex(ValueError, "Another benchmark coordinator"):
                with suite.output_lock(output):
                    self.fail("a second coordinator acquired the same output directory")

    def test_invalid_measurement_budget_fails_before_worker_launch(self):
        with patch.object(suite.subprocess, "Popen") as launch:
            for option in ("--time-budget", "--heartbeat-interval"):
                for budget in ("-1", "nan", "inf"):
                    with self.subTest(option=option, budget=budget), self.assertRaises(SystemExit):
                        suite.main([option, budget])
            launch.assert_not_called()

    def test_invalid_mixed_routes_fail_before_launch_and_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(suite.subprocess, "Popen") as launch:
            with self.assertRaises(SystemExit):
                suite.main(["--workloads", "activation", "--mixed-policy", '{"families":{"bad":"tilelang"}}',
                            "--output", str(Path(directory) / "new")])
            with self.assertRaises(SystemExit):
                suite.main(["--workloads", "activation", "--output", directory])
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
