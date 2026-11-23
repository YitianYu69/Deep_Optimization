"""Portable benchmark policy/worker wiring checks; no GPU timings are measured."""

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/fp8/benchmark_fp8_eager.py"
SPEC = importlib.util.spec_from_file_location("kernel_backend_eager_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class KernelBackendBenchmarkTests(unittest.TestCase):
    def test_triton_defaults_keep_the_legacy_configuration_path(self):
        args = benchmark.parser().parse_args([])
        self.assertEqual(args.deepopt_backend, "triton")
        self.assertIsNone(args.deepopt_kernel_policy)
        with patch.dict(sys.modules, {"deepopt.kernel_policy": None}):
            self.assertIsNone(benchmark.deepopt_kernel_configuration(args))
            args.deepopt_backend = "reference"
            self.assertIsNone(benchmark.deepopt_kernel_configuration(args))

    def test_custom_backends_resolve_to_canonical_policies(self):
        for backend in ("triton", "tilelang", "mixed"):
            with self.subTest(backend=backend):
                args = benchmark.parser().parse_args(["--deepopt-backend", backend])
                policy = benchmark.deepopt_kernel_configuration(args)
                if backend == "triton":
                    self.assertIsNone(policy)
                else:
                    self.assertEqual(policy["backend"], backend)
                    self.assertEqual(policy["fallback"], "error")
                    self.assertEqual(policy["families"], {})

    def test_preparation_and_autocast_share_the_canonical_policy(self):
        import deepopt
        from deepopt.opt import policy as model_policy

        mapping = {"preferred": "tilelang", "roles": {"linear.dw": "triton"},
                   "families": {"activation_codec": "triton"}}
        for backend in ("tilelang", "mixed"):
            arguments = ["--deepopt-backend", backend]
            if backend == "mixed":
                arguments += ["--deepopt-kernel-policy", json.dumps(mapping)]
            args = benchmark.parser().parse_args(arguments)
            with self.subTest(backend=backend), (
                patch.object(model_policy, "prepare_fp8_model", wraps=model_policy.prepare_fp8_model)
            ) as prepare, patch.object(deepopt, "autocast", wraps=deepopt.autocast) as autocast:
                case = benchmark.LinearCase(args, "deepopt", (32, 16, 48), "forward-backward", device="cpu")
                canonical = benchmark.deepopt_kernel_configuration(args)
                expected = {"runtime": {"kernels": canonical}}
                self.assertEqual(prepare.call_args.kwargs["config"], expected)
                self.assertEqual(autocast.call_args.kwargs["config"], expected)
                self.assertEqual(case.configuration["backend"], "inherit" if backend == "mixed" else backend)
                self.assertEqual(case.configuration["kernel_policy"], canonical)
                self.assertEqual(case.model.plan["fp8_training"]["kernel_policy"], canonical)
                self.assertEqual(case.amp.kernel_policy.as_dict(), canonical)
                self.assertEqual(case.model._deepopt_kernel_policy, canonical)

    def test_invalid_json_and_routes_fail_before_worker_launch(self):
        for raw in ("[1, 2]", "null", "not-json"):
            with self.subTest(raw=raw), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parser().parse_args(["--deepopt-kernel-policy", raw])
        for mapping, message in (
            ({"roles": {"linear.dw": "native_fast"}}, "routes must"),
            ({"families": {"made_up": "tilelang"}}, "unknown routes"),
            ({"unknown": True}, "unknown fields"),
            ({"backend": "tilelang"}, "must match"),
        ):
            with self.subTest(mapping=mapping):
                args = benchmark.parser().parse_args(["--deepopt-backend", "mixed",
                    "--deepopt-kernel-policy", json.dumps(mapping)])
                with self.assertRaisesRegex(ValueError, message):
                    benchmark.deepopt_kernel_configuration(args)

    def test_json_mapping_reaches_the_worker_and_report_unchanged(self):
        mapping = {"preferred": "tilelang", "roles": {"linear.dw": "triton"}}
        raw = json.dumps(mapping)
        arguments = ["--implementations", "deepopt", "--deepopt-backend", "mixed",
                     "--deepopt-kernel-policy", raw]
        args = benchmark.parser().parse_args(arguments)
        observed = []

        def launch(command, **kwargs):
            parsed = benchmark.parser().parse_args(command[2:])
            observed.append(parsed)
            result = {"implementation": parsed._worker, "status": "ok", "cases": []}
            return SimpleNamespace(returncode=0, stderr="", stdout=benchmark.RESULT_PREFIX + json.dumps(result))

        with patch.object(sys, "argv", [str(SCRIPT), *arguments]), (
            patch.object(benchmark.subprocess, "run", side_effect=launch)
        ), patch("sys.stdout", new=io.StringIO()):
            report = benchmark.run(args)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].deepopt_backend, "mixed")
        self.assertEqual(observed[0].deepopt_kernel_policy, mapping)
        self.assertEqual(report["settings"]["deepopt_kernel_policy"], mapping)
        self.assertEqual(report["results"][0]["status"], "ok")

    def test_default_report_settings_do_not_gain_a_null_policy_field(self):
        args = benchmark.parser().parse_args(["--implementations", "bf16"])
        result = {"implementation": "bf16", "status": "ok", "cases": []}
        completed = SimpleNamespace(returncode=0, stderr="", stdout=benchmark.RESULT_PREFIX + json.dumps(result))
        with patch.object(sys, "argv", [str(SCRIPT), "--implementations", "bf16"]), (
            patch.object(benchmark.subprocess, "run", return_value=completed)
        ), patch("sys.stdout", new=io.StringIO()):
            report = benchmark.run(args)
        self.assertNotIn("deepopt_kernel_policy", report["settings"])
        self.assertEqual(report["settings"]["deepopt_backend"], "triton")

    def test_worker_records_tilelang_package_only_for_requested_routes(self):
        original_matmul, original_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", original_matmul)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", original_cudnn)
        for backend in ("triton", "tilelang", "mixed"):
            args = benchmark.parser().parse_args(["--deepopt-backend", backend, "--_worker", "deepopt"])
            if backend == "mixed":
                args.deepopt_kernel_policy = {"roles": {"linear.dw": "tilelang"}}
            args.shapes = []
            with self.subTest(backend=backend), patch.object(torch.cuda, "is_available", return_value=True), (
                patch.object(torch.cuda, "set_device")
            ), patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), (
                patch.object(torch.cuda, "get_device_name", return_value="mock CUDA device")
            ), patch.object(benchmark, "package_info", return_value={"version": "test", "path": "test"}) as packages:
                result = benchmark.worker(args)
            self.assertEqual(result["status"], "ok", result)
            names = [call.args[0] for call in packages.call_args_list]
            self.assertEqual("tilelang" in names, backend != "triton")
            self.assertEqual("tilelang" in result["environment"], backend != "triton")


if __name__ == "__main__":
    unittest.main()
