"""Native-only physical execution contracts; large CUDA runs are opt-in."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples.vae import benchmark_vae as benchmark
from examples.vae import native_execution


ROOT = Path(__file__).resolve().parents[2]


def _receipt(directory, *, tiny=False):
    config = Path(benchmark.__file__).with_name("h200_bf16_storage_diagnostics.toml")
    raw = benchmark.settings(benchmark.parser().parse_args(
        ["--config", str(config), "--bf16-policy", "bf16_storage"]))
    if tiny:
        raw["application"]["model"] = dict(base_channels=2, max_channels=8,
            latent_channels=1, blocks_per_stage=2)
        raw["application"]["data"].update(image_size=16, resize_size=20)
        raw["training"]["batch_size"] = 1
        raw["loader"]["num_workers"] = 0
    path = Path(directory) / "original.json"
    path.write_text(json.dumps({"settings": raw, "optimizer_factory": benchmark.CUSTOM_OPTIMIZER,
                               "model_factory": None}))
    return path, raw


def _arguments(directory, *, tiny=False, extra=()):
    receipt, raw = _receipt(directory, tiny=tiny)
    args = native_execution.parser().parse_args(["--from-result", str(receipt),
        "--output", str(Path(directory) / "execution.json"), *extra])
    return args, raw


class VAENativeExecutionPlanningTests(unittest.TestCase):
    def test_stdlib_dry_run_preserves_original128_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            args, raw = _arguments(directory)
            original = args.from_result.read_bytes()
            command = [sys.executable, "-S", str(Path(native_execution.__file__).resolve()),
                "--from-result", str(args.from_result), "--dry-run"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads(completed.stdout)
            self.assertEqual(args.from_result.read_bytes(), original)
        self.assertEqual(plan["target_settings"], raw)
        self.assertEqual(plan["settings"]["training"]["batch_size"], 128)
        self.assertEqual(plan["settings"]["training"]["accumulation_steps"], 1)
        self.assertEqual(plan["settings"]["application"], raw["application"])
        self.assertEqual(plan["settings"]["runtime"]["activation_compression"],
                         raw["runtime"]["activation_compression"])
        self.assertEqual(plan["settings"]["runtime"]["activation_memory"],
                         raw["runtime"]["activation_memory"])
        self.assertEqual(plan["optimizer_factory"], benchmark.CUSTOM_OPTIMIZER)
        self.assertFalse(plan["qualification_eligible"])

    def test_recorded_mode_and_native_options_normalize_without_mutating_envelope(self):
        from deepopt.config import ConfigError, config_from_dict

        with tempfile.TemporaryDirectory() as directory:
            args, raw = _arguments(directory)
            frozen = deepcopy(raw)
            self.assertIsNotNone(raw["runtime"]["compile"]["mode"])
            self.assertTrue(raw["runtime"]["compile"]["options"]["use_native_autodiff"])
            with self.assertRaisesRegex(ConfigError, "Choose runtime.compile.mode"):
                config_from_dict(raw)
            plan = native_execution.execution_plan(args)
        self.assertEqual(raw, frozen)
        self.assertEqual(plan["target_settings"], frozen)
        native = plan["settings"]["runtime"]["compile"]
        self.assertIsNone(native["mode"])
        self.assertTrue(native["enabled"])
        self.assertEqual(native["backend"], "deepopt")
        self.assertTrue(native["fullgraph"])
        self.assertFalse(native["dynamic"])
        self.assertTrue(native["options"]["use_native_autodiff"])
        self.assertTrue(native["options"]["require_full_aot"])
        config_from_dict(plan["settings"])

    def test_invalid_execution_counts_reject_before_allocating_a_model(self):
        for extra in (("--batch-size", "0"), ("--warmup", "-1"), ("--updates", "0")):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises((ValueError, SystemExit)):
                    args, _ = _arguments(directory, extra=extra)
                    native_execution.execution_plan(args)


class VAENativeExecutionCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        from importlib.metadata import version

        # These CPU fixtures exercise execution contracts, not one historical
        # qualification wheel; the arm-manifest pinning tests keep that gate.
        expected = patch.object(native_execution, "EXPECTED_VERSION", version("deep-optimization"))
        expected.start()
        self.addCleanup(expected.stop)
        # Execution plans install process defaults for the real command path;
        # restore them so an in-process fixture cannot affect later tests.
        from deepopt import reset_config
        reset_config()
        self.addCleanup(reset_config)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def assert_no_tensor_files(self, directory, report):
        self.assertEqual(report["tensor_files_written"], 0)
        forbidden = {".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz", ".safetensors", ".ckpt"}
        self.assertFalse([str(path) for path in Path(directory).rglob("*") if path.suffix in forbidden])
        json.dumps(report, allow_nan=False)

    def artifact_files(self, report, directory):
        found = []
        def visit(value):
            if isinstance(value, dict):
                if {"path", "sha256", "bytes"}.issubset(value):
                    path = Path(value["path"])
                    if not path.is_absolute():
                        matches = list(Path(directory).rglob(str(path)))
                        self.assertEqual(len(matches), 1, value)
                        path = matches[0]
                    data = path.read_bytes()
                    self.assertEqual(len(data), value["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), value["sha256"])
                    found.append(path)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(report)
        return found

    def run_fixture(self, directory, *, warmup=1, updates=2):
        args, raw = _arguments(directory, tiny=True, extra=("--device", "cpu", "--batch-size", "1",
            "--synthetic", "--optimizer", "sgd", "--warmup", str(warmup), "--updates", str(updates)))
        with redirect_stdout(io.StringIO()):
            code = native_execution.run_native_execution(args)
        return code, json.loads(args.output.read_text()), raw

    def test_actual_native_cpu_updates_use_one_route_without_a_paired_oracle(self):
        import torch

        original_step = torch.optim.SGD.step
        changed, gradients_present = [], []
        def counted_step(optimizer, *args, **kwargs):
            parameters = [value for group in optimizer.param_groups for value in group["params"]]
            gradients_present.append(all(value.grad is not None for value in parameters if value.requires_grad))
            before = [value.detach().clone() for value in parameters]
            result = original_step(optimizer, *args, **kwargs)
            changed.append(any(not torch.equal(old, value) for old, value in zip(before, parameters)))
            return result
        with tempfile.TemporaryDirectory() as directory:
            with patch("examples.vae.qualification.paired_trajectory",
                       side_effect=AssertionError("Native execution invoked a paired oracle")), \
                    patch("examples.vae.qualification.TensorOracle.__init__",
                          side_effect=AssertionError("Native execution instantiated a tensor oracle")), \
                    patch.object(torch.optim.SGD, "step", new=counted_step):
                code, report, raw = self.run_fixture(directory)
            self.assertEqual(code, 0, report.get("error"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["target_settings"], raw)
            self.assertEqual(report["input"]["shape"], [1, 3, 16, 16])
            self.assertEqual(report["model"]["parameters"], 14425)
            self.assertEqual(report["settings"]["training"]["optimizer"], "sgd")
            self.assertEqual(report["settings"]["runtime"]["precision"], "fp32")
            self.assertEqual(report["settings"]["runtime"]["bf16_policy"], "torch_amp")
            self.assertEqual(report["target_settings"]["runtime"]["precision"], "bf16")
            self.assertEqual(report["target_settings"]["runtime"]["bf16_policy"], "bf16_storage")
            self.assertEqual(report["gradient_mode"], "trainable_parameters")
            self.assertTrue(report["native_ownership_verified"])
            self.assertTrue(report["execution_completed"])
            self.assertEqual(report["warmup_completed"], 1)
            self.assertEqual(report["updates_completed"], 2)
            self.assertEqual([row["stage"] for row in report["steps"]], ["warmup", "update", "update"])
            self.assertEqual(len(changed), 3)
            self.assertTrue(all(changed))
            self.assertTrue(all(gradients_present))
            self.assertEqual(report["capacity_status"], "CPU_diagnostic_only")
            self.assertFalse(report["physical_target_executed"])
            self.assertFalse(report["numerical_comparisons_performed"])
            self.assertTrue(report["artifacts"]["ordinary_artifact_fn_used"])
            self.assertFalse(report["artifacts"]["numerical_boundary_observation"])
            self.assertGreaterEqual(report["artifacts"]["execution_calls"]["forward"], 3)
            self.assertGreaterEqual(report["artifacts"]["execution_calls"]["backward"], 3)
            for row in report["steps"]:
                for key in ("forward_completed", "backward_completed", "optimizer_completed",
                            "finite_loss", "finite_gradients", "finite_parameters"):
                    self.assertTrue(row[key], (key, row))
            for key in ("qualification_eligible", "qualified", "correctness_assessed", "timing_eligible"):
                self.assertFalse(report[key])
            self.assertTrue(report["correctness_not_assessed"])
            self.assert_no_tensor_files(directory, report)
            files = self.artifact_files(report, directory)
            self.assertTrue(any(path.suffix == ".py" for path in files))
            self.assertTrue(any(path.suffix == ".json" for path in files))
            compiler_name = Path(report["compiler_report_artifact"]["path"]).name
            compiler_file, = {path for path in files if path.name == compiler_name}
            saved = json.loads(compiler_file.read_text())
            compiler = saved.get("compiler_report") or saved
            self.assertGreater(compiler["captured_graphs"], 0)
            self.assertGreater(compiler["native_autodiff_operations"], 0)
            self.assertGreater(compiler["native_autodiff_regions"], 0)
            self.assertEqual(compiler["runtime_fallbacks"], 0)
            for graph in compiler["graphs"]:
                self.assertFalse(graph["eager_autograd_operations"])
                self.assertFalse(graph["legacy_backward_operations"])
            compilation = report["settings"]["runtime"]["compile"]
            self.assertTrue(compilation["fullgraph"])
            self.assertTrue(compilation["options"]["require_full_aot"])
            self.assertTrue(compilation["options"]["use_native_autodiff"])

    def test_actual_native_failure_retains_program_context_and_generated_code(self):
        from deepopt.opt.nn.primitive_state import Nearest2dGeometry

        original = Nearest2dGeometry.forward
        def failed_nearest(geometry, value):
            original(geometry, value)
            raise RuntimeError("Injected native nearest execution failure")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Nearest2dGeometry, "forward", new=failed_nearest):
                code, report, _ = self.run_fixture(directory, warmup=0, updates=1)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["execution_completed"])
            self.assertFalse(report["qualification_eligible"])
            self.assertEqual(report["updates_completed"], 0)
            self.assertIn("Injected native nearest execution failure", report["error"])
            self.assertIn("failed_nearest", report["traceback"])
            self.assertTrue(report["failure_context"])
            self.assertEqual(report["failure_context"]["native_artifact_phase"], "forward")
            self.assertTrue(report["failure_context"]["native_artifact_source_sha256"])
            files = self.artifact_files(report, directory)
            self.assertTrue(any(path.suffix == ".py" for path in files))
            source_hash = report["failure_context"]["native_artifact_source_sha256"]
            self.assertTrue(any(hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
                                for path in files if path.suffix == ".py"))
            nodes = [node for path in files if path.suffix == ".json"
                     for boundary in json.loads(path.read_text()).get("boundary_metadata", [])
                     for node in boundary.get("nodes", [])]
            self.assertTrue(any(node.get("operation") == "autodiff_nearest2d"
                                and node.get("module") == "decoder.0" for node in nodes))
            self.assertTrue(report["compiler_report_artifact"])
            self.assert_no_tensor_files(directory, report)


@unittest.skipUnless(os.environ.get("DEEPOPT_RUN_VAE_NATIVE_B128") == "1",
                     "Original physical B128 CUDA execution requires explicit opt-in")
class VAENativeExecutionLargeCudaTests(unittest.TestCase):
    def test_original224_physical_batch128_native_execution(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("Actual CUDA is required")
        receipt = os.environ.get("DEEPOPT_VAE_NATIVE_FROM_RESULT")
        if not receipt:
            self.fail("Set DEEPOPT_VAE_NATIVE_FROM_RESULT to the original workload receipt")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(os.environ.get("DEEPOPT_VAE_NATIVE_B128_OUTPUT",
            str(ROOT / "work" / f"vae-native-b128-test-{stamp}.json")))
        args = native_execution.parser().parse_args(["--from-result", receipt, "--batch-size", "128",
            "--device", os.environ.get("DEEPOPT_VAE_NATIVE_DEVICE", "cuda:0"),
            "--warmup", "1", "--updates", "2", "--output", str(output)])
        code = native_execution.run_native_execution(args)
        report = json.loads(output.read_text())
        self.assertEqual(code, 0, report.get("error"))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["input"]["shape"], [128, 3, 224, 224])
        self.assertEqual(report["model"]["parameters"], 32317999)
        self.assertTrue(report["native_ownership_verified"])
        self.assertTrue(report["execution_completed"])
        self.assertEqual(report["updates_completed"], 2)
        self.assertEqual(report["warmup_completed"], 1)
        self.assertFalse(report["qualification_eligible"])
        self.assertFalse(report["correctness_assessed"])


if __name__ == "__main__":
    unittest.main()
