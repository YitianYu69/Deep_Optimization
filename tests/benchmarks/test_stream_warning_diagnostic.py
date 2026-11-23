"""Diagnostic orchestration/reporting contracts; CPU is not CUDA validation."""

from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/cuda_graph/analyze_stream_warning.py"
SPEC = importlib.util.spec_from_file_location("stream_warning_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class StreamWarningDiagnosticTests(unittest.TestCase):
    def test_copied_help_requires_only_standard_library(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, target)
            result = subprocess.run([sys.executable, "-S", str(target), "--help"], cwd=directory,
                                    text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--cases", "--launcher", "--node-trace", "--profile", "--config", "--input-grad"):
            self.assertIn(option, result.stdout)
        self.assertNotIn("sys.path.insert", SCRIPT.read_text())

    def test_warning_phase_stack_and_node_records_are_json_metadata(self):
        recorder = diagnostic.Recorder(torch, torch.device("cpu"))
        node = SimpleNamespace(name=lambda: "torch::autograd::AccumulateGrad", variable=torch.ones(2))
        with redirect_stderr(io.StringIO()), recorder.warning_capture():
            with recorder.phase("candidate.train.backward"):
                warnings.warn(diagnostic.WARNING_TEXT, UserWarning)
                recorder.node_created(node)
        self.assertEqual(recorder.current, "startup")
        self.assertEqual(recorder.warnings[0]["phase"], "candidate.train.backward")
        self.assertTrue(recorder.warnings[0]["stack"])
        self.assertEqual(recorder.nodes[0]["shape"], [2])
        json.dumps({"warnings": recorder.warnings, "nodes": recorder.nodes, "events": recorder.events})
        self.assertNotIn("variable", recorder.nodes[0])

    def test_error_report_rejects_nonfinite_or_missing_gradients(self):
        same = {"weight": torch.tensor([1., 2.])}
        self.assertTrue(diagnostic.error_summary(torch, same, same, 0, 0)["passed"])
        changed = diagnostic.error_summary(torch, {"weight": torch.tensor([1., 3.])}, same, 0, 0)
        self.assertFalse(changed["passed"])
        self.assertEqual(changed["max_abs"], 1)
        self.assertFalse(diagnostic.error_summary(torch, {"weight": None}, same, 0, 0)["passed"])
        nan = {"weight": torch.tensor([float("nan")])}
        self.assertFalse(diagnostic.error_summary(torch, nan, nan, 0, 0)["passed"])

    def test_child_environment_removes_checkout_and_parent_rendezvous(self):
        with patch.dict(os.environ, {"PYTHONPATH": "checkout", "RANK": "0", "WORLD_SIZE": "1",
                                     "MASTER_PORT": "1234", "CUDA_VISIBLE_DEVICES": "2,3"}):
            result = diagnostic.child_environment()
        self.assertNotIn("PYTHONPATH", result)
        self.assertNotIn("RANK", result)
        self.assertNotIn("MASTER_PORT", result)
        self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_summary_does_not_attribute_reference_warning_to_candidate(self):
        row = {"rank": 0, "status": "ok", "warnings": [
            {"phase": "reference.train.backward", "message": diagnostic.WARNING_TEXT},
            {"phase": "candidate.train.backward", "message": diagnostic.WARNING_TEXT},
        ]}
        result = diagnostic.case_summary({"case": "eager", "pass": "diagnostic", "exit_code": 0, "ranks": [row]})
        self.assertEqual(result["ranks"][0]["first_mismatch_phase"], "candidate.train.backward")

    def test_invalid_options_fail_before_launching(self):
        for options, message in ((["--steps", "0"], "steps must be positive"),
                                 (["--ranks", "2"], "require --launcher torchrun"),
                                 (["--device", "cpu"], "require CUDA"),
                                 (["--rtol", "nan"], "tolerances must be nonnegative"),
                                 (["--retain-graph"], "manual-loop control"),
                                 (["--zero-grad", "zero"], "own gradient clearing")):
            with self.subTest(options=options), redirect_stderr(io.StringIO()) as output, self.assertRaises(SystemExit):
                diagnostic.main(options)
            self.assertIn(message, output.getvalue())

    def test_controller_captures_missing_reports_without_reusing_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            args = {"launcher": "python", "ranks": 1, "timeout": 10, "pass": "diagnostic"}
            worker = target / "empty.py"
            worker.write_text("raise SystemExit(0)\n")
            result = diagnostic.launch_job(worker, {"case": "eager", "args": args}, target)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["ranks"][0]["status"], "error")
        self.assertIn("report missing", result["ranks"][0]["error"])

    def test_controller_stops_timed_out_process(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            worker = target / "sleep.py"
            worker.write_text("import time\ntime.sleep(60)\n")
            args = {"launcher": "python", "ranks": 1, "timeout": .25, "pass": "diagnostic"}
            result = diagnostic.launch_job(worker, {"case": "eager", "args": args}, target)
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["exit_code"], 0)

    def test_cpu_eager_runtime_and_trainer_reference_paths(self):
        # No capture is attempted here: this verifies orchestration, loss
        # normalization, optimizer hooks and snapshot lifecycle on CPU.
        base = vars(diagnostic.parser().parse_args([]))
        base.update(device="cpu", seed=42, accumulation=2, warmup=3, bits=2, group_size=1024, recenter=True,
                    steps=2, validation_transition=True, input_grad=True, execution_mode="max_speed",
                    loaded_config={"training": {"optimizer": "sgd", "learning_rate": .001}},
                    **{"pass": "diagnostic"})
        for case in ("runtime", "trainer"):
            with self.subTest(case=case):
                recorder = diagnostic.Recorder(torch, torch.device("cpu"))
                result = diagnostic.execute_pass(torch, base, case, recorder, torch.device("cpu"), "reference")
                snapshots = result["snapshots"]
                self.assertEqual(result["optimizer_updates"], 2)
                self.assertEqual(set(snapshots["outputs"]), {"0", "1", "2", "3", "validation"})
                self.assertTrue(all(value is not None for value in snapshots["gradients_last_update"].values()))
                self.assertEqual(len(snapshots["input_gradients"]), 4)
                self.assertIsNone(result["statistics"])

    def test_fp8_trainer_reference_preserves_compute_and_compression_settings(self):
        import deepopt

        real_settings = deepopt.FP8Config

        def reference_settings(**options):
            return real_settings(**{**options, "backend": "reference", "backward_backend": "inherit"})

        args = vars(diagnostic.parser().parse_args([]))
        args.update(device="cpu", seed=42, accumulation=1, warmup=3, bits=2, group_size=1024,
                    recenter=True, steps=2, input_grad=True, execution_mode="max_speed", loaded_config={},
                    **{"pass": "diagnostic"})
        for case in ("fp8", "fp8_compressed"):
            with self.subTest(case=case), patch.object(deepopt, "FP8Config", reference_settings):
                result = diagnostic.execute_pass(torch, args, case, diagnostic.Recorder(torch, torch.device("cpu")),
                                                 torch.device("cpu"), "reference")
                self.assertEqual(result["orchestrator"], "trainer")
                config = result["effective_config"]["runtime"]
                self.assertTrue(config["fp8"]["enabled"])
                self.assertEqual(config["activation_compression"]["enabled"], case == "fp8_compressed")
                self.assertEqual(result["optimizer_updates"], 2)
                self.assertTrue(all(torch.isfinite(value).all() for value in result["snapshots"]["gradients_last_update"].values()))

    def test_bf16_probe_capture_policy_matches_runtime_compute(self):
        # Policy scopes are device keyed but independent of CUDA execution.
        # Exercise the real Runtime/Trainer scopes on CPU, without graph mocks.
        from deepopt.opt import _autocast as scopes

        def policy():
            return (scopes._get_fp8_plan("cpu"), scopes._get_activation_compression("cpu"),
                    *(scope.get().get("cpu", True) for scope in (
                        scopes._active_interception, scopes._active_convolution_interception,
                        scopes._active_normalization_interception, scopes._active_pooling_interception)))

        args = vars(diagnostic.parser().parse_args([]))
        args.update(device="cpu", seed=42, accumulation=1, warmup=3, bits=2, group_size=1024,
                    recenter=True, execution_mode="max_speed", loaded_config={}, **{"pass": "diagnostic"})
        for case in ("runtime", "trainer"):
            with self.subTest(case=case):
                run = diagnostic.Run(torch, args, case, diagnostic.Recorder(torch, torch.device("cpu")),
                                     torch.device("cpu"), "reference")
                try:
                    # CPU fp32 normally uses a null context. Enable BF16 on
                    # this already-constructed policy to exercise the same
                    # scope branch as CUDA BF16, without CUDA or any forward.
                    run.runtime._precision.precision = "bf16"
                    run.runtime._precision.configure(fp8=None, config=run.runtime.config)
                    with run.autocast(capture=True):
                        captured_policy = policy()
                    training_policy = run.runtime.compute(run.batch(0), lambda batch: policy())
                    self.assertEqual(captured_policy, training_policy)
                finally:
                    run.close()

    def test_copied_cpu_runner_correctness_fresh_processes_and_profiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            config = root / "config.toml"
            config.write_text('seed=42\n[application]\napi_token="private-app-value"\n'
                              '[training]\nlearning_rate=0.0002\naccumulation_steps=2\n'
                              '[runtime.activation_compression]\nbits=4\ngroup_size=1024\nnorm_xhat_recenter=false\n')
            result = subprocess.run([sys.executable, str(script), "--device", "cpu", "--cases", "eager",
                                     "--config", str(config), "--steps", "2", "--input-grad", "--node-trace",
                                     "--validation-transition", "--timing-steps", "1", "--timing-warmup", "0",
                                     "--profile", "--profile-steps", "1", "--output-dir", str(root / "reports")],
                                    cwd=root, text=True, capture_output=True, timeout=90,
                                    env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads(result.stdout)
            report = json.loads(Path(summary["report"]).read_text())
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(report["configuration"]["accumulation"], 2)
            self.assertEqual(report["configuration"]["bits"], 4)
            self.assertEqual(report["configuration"]["group_size"], 1024)
            self.assertNotIn("private-app-value", json.dumps(report))
            rows = [job["ranks"][0] for job in report["jobs"]]
            self.assertEqual(len({row["environment"]["pid"] for row in rows}), 3)
            self.assertTrue(rows[0]["correctness"]["passed"])
            self.assertGreater(rows[0]["correctness"]["comparisons"]["input_gradients"]["elements"], 0)
            self.assertFalse(rows[0]["native_case_imported_deepopt"])
            self.assertFalse(rows[0]["environment"]["cuda_available"] and rows[0]["environment"]["gpu"] is not None)
            self.assertTrue(all(event["stream"] is None for event in rows[0]["events"]))
            self.assertEqual(rows[0]["candidate"]["effective_config"]["training"]["learning_rate"], .0002)
            self.assertGreater(rows[1]["candidate"]["timing"]["wall_ms_per_update"], 0)
            self.assertIsNone(rows[1]["candidate"]["timing"]["cuda_event_ms_per_update"])
            self.assertTrue(Path(rows[2]["candidate"]["profile"]["trace"]).exists())


if __name__ == "__main__":
    unittest.main()
