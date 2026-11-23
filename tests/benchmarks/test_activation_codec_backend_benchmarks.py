"""Codec benchmark backend selection and resume isolation, without GPU timing."""

from contextlib import ExitStack
from hashlib import sha256
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

from deepopt._benchmarking import BenchmarkBackend
from deepopt.kernel_policy import current_kernel_policy


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "benchmarks/activation_compression" / f"benchmark_activation_{name}.py"
    spec = importlib.util.spec_from_file_location(f"codec_backend_benchmark_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PACK, UNPACK = load("pack"), load("unpack")


class ActivationCodecBackendBenchmarkTests(unittest.TestCase):
    def test_help_and_json_syntax_validation_need_only_stdlib(self):
        for module in (PACK, UNPACK):
            for extra, code in ((["--help"], 0), (["--deepopt-kernel-policy", "{"], 2)):
                with self.subTest(module=module.__name__, extra=extra):
                    result = subprocess.run([sys.executable, "-S", module.__file__, "--output", "unused", *extra],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, code, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    if code == 0:
                        self.assertIn("--deepopt-backend", result.stdout)
                        self.assertIn("--deepopt-kernel-policy", result.stdout)

    def test_invalid_mappings_fail_before_gpu_dependencies_or_lease(self):
        original_import = __import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "triton", "tilelang"):
                raise AssertionError("Policy validation imported GPU dependency " + name)
            return original_import(name, *args, **kwargs)

        for module in (PACK, UNPACK):
            for raw in ("null", "[]", '{"roles":{"linear.dw":"native_fast"}}',
                        '{"families":{"unknown":"tilelang"}}', '{"backend":"triton"}'):
                args = ["--output", "unused", "--deepopt-backend", "mixed", "--deepopt-kernel-policy", raw]
                with self.subTest(module=module.__name__, raw=raw), (
                    patch("builtins.__import__", new=guarded)
                ), patch.object(module, "_lease_cli") as lease, patch.object(module, "run_selected") as run, (
                    patch("sys.stderr", new=io.StringIO())
                ), self.assertRaises(SystemExit) as failure:
                    module.main(args)
                self.assertEqual(failure.exception.code, 2)
                lease.assert_not_called()
                run.assert_not_called()

    def test_preparation_and_measurement_are_inside_selected_scope(self):
        before = current_kernel_policy()
        for module in (PACK, UNPACK):
            options = ["--output", "unused", "--deepopt-backend", "mixed", "--deepopt-kernel-policy",
                       '{"preferred":"triton","families":{"activation_codec":"tilelang"}}']
            observed = []

            def run(*arguments):
                selection = arguments[-1]
                observed.append(current_kernel_policy())
                self.assertEqual(current_kernel_policy(), selection.policy)
                self.assertEqual(selection.policy.route("activation_codec"), "tilelang")
                return 0

            with self.subTest(module=module.__name__), patch.object(module, "run_selected", side_effect=run):
                self.assertEqual(module.main(options), 0)
            self.assertEqual(len(observed), 1)
            self.assertIs(current_kernel_policy(), before)

    def test_selected_codec_dispatch_requires_no_gpu(self):
        for backend, mapping, expected in (("triton", {}, "triton"), ("tilelang", {}, "tilelang"),
                ("mixed", {"families": {"activation_codec": "tilelang"}}, "tilelang")):
            args = SimpleNamespace(deepopt_backend=backend, deepopt_kernel_policy=mapping)
            selection = BenchmarkBackend.from_args(args)
            with self.subTest(backend=backend), selection.scope(), (
                patch("deepopt.opt.backends.dispatch.import_module", side_effect=lambda name: SimpleNamespace(__name__=name))
            ) as imported:
                module = UNPACK.codec_module()
            self.assertEqual(module.__name__, f"deepopt.opt.backends.{expected}.quantization.activation.affine")
            imported.assert_called_once_with(module.__name__)

    def test_tilelang_deduplicates_inactive_knobs_but_retains_grouped_control(self):
        module = SimpleNamespace(__name__="deepopt.opt.backends.tilelang.quantization.activation.affine")
        pack_args = PACK.parser().parse_args(["--output", "unused", "--num-warps", "4", "8"])
        unpack_args = UNPACK.parser().parse_args(["--tiles", "16x32", "32x64", "--warps", "4", "8"])
        for operation, args, actual in (("pack", pack_args, "serial_per_group"),
                                        ("unpack", unpack_args, "flat_elementwise")):
            with self.subTest(operation=operation):
                requested = {"implementation": "tiled", "num_warps": [4, 8]}
                schedule = UNPACK.codec_schedule(module, operation, requested)
                choices = UNPACK.codec_choices(args, operation, schedule)
                self.assertEqual(len(choices), 1)
                self.assertEqual(choices[0][0], "grouped")
                self.assertEqual(schedule["actual_schedule"], actual)
                self.assertFalse(schedule["tuning_controls_active"])
                self.assertEqual(schedule["accepted_but_inactive_controls"], requested)

    def test_triton_tuning_grid_is_preserved(self):
        module = SimpleNamespace(__name__="deepopt.opt.backends.triton.quantization.activation.affine")
        args = PACK.parser().parse_args(["--output", "unused"])
        schedule = UNPACK.codec_schedule(module, "pack", {})
        self.assertEqual(UNPACK.codec_choices(args, "pack", schedule),
                         [("grouped", 1, 4), ("batched", 4, 4), ("batched", 8, 4),
                          ("serial", 4, 4), ("serial", 8, 4)])


class ActivationCodecResumeTests(unittest.TestCase):
    def test_identity_tracks_policy_backend_version_sources_and_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "deepopt"
            package.mkdir()
            (package / "__init__.py").write_text("# fake installed DeepOpt\n")
            helper = package / "_benchmarking.py"
            helper.write_text("# shared backend helper v1\n")
            timing_helper = root / "timing.py"
            timing_helper.write_text("# timing v1\n")
            backend = root / "tilelang"
            backend.mkdir()
            backend_source = backend / "__init__.py"
            backend_source.write_text("# backend source v1\n")
            backend_template = backend / "conversion.cuh"
            backend_template.write_text("// backend template v1\n")
            backend_runtime = backend / "runtime.so"
            backend_runtime.write_bytes(b"runtime-v1")
            deepopt = SimpleNamespace(__file__=str(package / "__init__.py"))
            gpu = SimpleNamespace(uuid="fake-uuid", name="fake GPU")
            torch = SimpleNamespace(__version__="fake-torch", version=SimpleNamespace(cuda="fake-cuda"),
                cuda=SimpleNamespace(get_device_properties=lambda device: gpu, get_device_name=lambda device: "fake GPU"))
            versions = {"triton": "t1", "tilelang": "t1"}

            def metadata(args):
                return BenchmarkBackend.from_args(args).metadata(args.device)

            def identity(module, args):
                resolved = metadata(args)
                if module is PACK:
                    value = PACK.resume_identity(args, deepopt, torch, timing_helper,
                                                 return_metadata=True, kernel_backend=resolved)
                else:
                    value = UNPACK.resume_identity(args, deepopt, torch, resolved)
                return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

            spec = SimpleNamespace(origin=str(backend_source))
            driver = SimpleNamespace(returncode=0, stdout="fake-uuid, fake-driver")
            with ExitStack() as stack:
                stack.enter_context(patch("deepopt._benchmarking.version", side_effect=lambda name: versions[name]))
                stack.enter_context(patch.object(importlib.util, "find_spec", return_value=spec))
                stack.enter_context(patch.object(subprocess, "run", return_value=driver))
                stack.enter_context(patch.object(PACK.platform, "platform", return_value="fake platform"))
                for module in (PACK, UNPACK):
                    with self.subTest(module=module.__name__):
                        args = module.parser().parse_args(["--output", str(root / "report.json"), "--deepopt-backend", "tilelang"])
                        first = identity(module, args)
                        self.assertEqual(first, identity(module, args))
                        args.deepopt_backend = "triton"
                        self.assertNotEqual(first, identity(module, args))
                        args.deepopt_backend = "mixed"
                        args.deepopt_kernel_policy = {"families": {"activation_codec": "tilelang"}}
                        self.assertNotEqual(first, identity(module, args))
                        args.deepopt_backend, args.deepopt_kernel_policy = "tilelang", {}
                        versions["tilelang"] = "t2"
                        self.assertNotEqual(first, identity(module, args))
                        versions["tilelang"] = "t1"
                        backend_source.write_text("# backend source v2\n")
                        self.assertNotEqual(first, identity(module, args))
                        backend_source.write_text("# backend source v1\n")
                        backend_template.write_text("// backend template v2\n")
                        self.assertNotEqual(first, identity(module, args))
                        backend_template.write_text("// backend template v1\n")
                        backend_runtime.write_bytes(b"runtime-v2")
                        self.assertNotEqual(first, identity(module, args))
                        backend_runtime.write_bytes(b"runtime-v1")
                        helper.write_text("# shared backend helper v2\n")
                        self.assertNotEqual(first, identity(module, args))
                        helper.write_text("# shared backend helper v1\n")
                        if module is PACK:
                            timing_helper.write_text("# timing v2\n")
                            self.assertNotEqual(first, identity(module, args))
                            timing_helper.write_text("# timing v1\n")
                            legacy = SimpleNamespace(**{key: value for key, value in vars(args).items()
                                if key not in ("deepopt_backend", "deepopt_kernel_policy")})
                            legacy_metadata = PACK.resume_identity(legacy, deepopt, torch, timing_helper, return_metadata=True)
                            self.assertEqual(legacy_metadata["kernel_backend"]["policy"]["backend"], "triton")

    def test_resume_reuses_only_matching_backend_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            row = {"case_id": "control", "exact": True,
                   "timing": {"eager": {"wall_ms_median": 1., "wall_ms_samples": [1.]}}}
            path.write_text(json.dumps({"resume_key": "tilelang-source-v1", "results": [row]}))
            args = SimpleNamespace(resume=True, execution="eager", repeats=1, profile_dir=None)
            for key, expected in (("tilelang-source-v1", {"control": row}), ("triton-source-v1", {}),
                                  ("tilelang-source-v2", {})):
                with self.subTest(key=key):
                    self.assertEqual(PACK.cached_results(path, key, args, UNPACK), expected)
                    self.assertEqual(UNPACK.cached_results(path, key, args), expected)


if __name__ == "__main__":
    unittest.main()
