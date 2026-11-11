"""Package organization preserves lazy imports and existing recipe identities."""

from importlib import import_module
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch


CPU_ALIASES = (
    ("deepopt.opt.do_nn.functional", "deepopt.opt.nn.functional"),
    ("deepopt.opt.do_nn.compressed_linear", "deepopt.opt.nn.functional"),
    ("deepopt.opt.nn.compressed_linear", "deepopt.opt.nn.functional"),
    ("deepopt.opt.do_nn.linear_module", "deepopt.opt.nn.linear"),
    ("deepopt.opt.do_nn.functional_module", "deepopt.opt.nn.functional_linear"),
    ("deepopt.opt.do_nn.hf_conv1d_module", "deepopt.opt.nn.hf_conv1d"),
    ("deepopt.opt.do_nn.mha_module", "deepopt.opt.nn.attention"),
    ("deepopt.opt.do_nn._cached_weights", "deepopt.opt.nn._cached_weights"),
    ("deepopt.opt.do_nn.AOTAutograd_Func.linear_autograd_func", "deepopt.opt.nn.autograd.linear"),
    ("deepopt.opt.scaling_strategy.amax", "deepopt.opt.scaling.amax"),
    ("deepopt.opt.scaling_strategy.delayed", "deepopt.opt.scaling.delayed"),
    ("deepopt.opt.backends._cuda_launch", "deepopt.opt.backends._shared.cuda"),
    ("deepopt.opt.backends._gemm_launch", "deepopt.opt.backends.triton.gemm.launch"),
    ("deepopt.opt.backends._triton_autotune", "deepopt.opt.backends.triton.gemm.autotune"),
    ("deepopt.opt.backends._delayed_launch", "deepopt.opt.backends.triton.quantization.launch"),
    ("deepopt.opt.backends.native_fp8", "deepopt.opt.backends.native.fp8"),
)

TRITON_ALIASES = (
    ("deepopt.opt.backends.triton_fp8", "deepopt.opt.backends.triton.gemm.fp8"),
    ("deepopt.opt.backends.triton_quantization", "deepopt.opt.backends.triton.quantization.fp8"),
    ("deepopt.opt.backends.triton_delayed", "deepopt.opt.backends.triton.quantization.delayed"),
    ("deepopt.opt.backends.scaling_utils.fused_amax", "deepopt.opt.backends.triton.scaling.amax"),
    ("deepopt.opt.backends.act.triton_act_affine_min_max_quantization",
     "deepopt.opt.backends.triton.quantization.activation.affine"),
    ("deepopt.opt.backends.act.triton_act_fp8_affine_min_max_quantization",
     "deepopt.opt.backends.triton.quantization.activation.fp8"),
    ("deepopt.opt.backends.act.triton_act_amax_quantization",
     "deepopt.opt.backends.triton.quantization.activation.amax"),
)


class PackageLayoutTests(unittest.TestCase):
    def run_fresh(self, code):
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_old_and_current_imports_share_one_module_in_both_import_orders(self):
        for old_first in (True, False):
            with self.subTest(old_first=old_first):
                self.run_fresh(f"""
from importlib import import_module
import sys

for old, current in {CPU_ALIASES!r}:
    first, second = (old, current) if {old_first!r} else (current, old)
    first_module = import_module(first)
    second_module = import_module(second)
    assert first_module is second_module, (old, current)
    assert sys.modules[old] is sys.modules[current]
    assert '_legacy' not in first_module.__file__, first_module.__file__
""")

    def test_package_imports_do_not_load_torch_or_cuda_kernels(self):
        self.run_fresh("""
import sys
import deepopt.config
import deepopt.opt
import deepopt.opt.nn
import deepopt.opt.nn.autograd
import deepopt.opt.scaling
import deepopt.opt.quantization
import deepopt.opt.do_nn
import deepopt.opt.do_nn.AOTAutograd_Func
import deepopt.opt.scaling_strategy

assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
""")

    def test_backend_packages_do_not_load_triton_implementations(self):
        self.run_fresh("""
import sys
import deepopt.opt.backends.native
import deepopt.opt.backends.reference
import deepopt.opt.backends.triton
import deepopt.opt.backends.triton.gemm
import deepopt.opt.backends.triton.quantization
import deepopt.opt.backends.triton.quantization.activation
import deepopt.opt.backends.triton.scaling
import deepopt.opt.backends.act
import deepopt.opt.backends.scaling_utils

assert 'triton' not in sys.modules
assert 'deepopt.opt.backends.native.fp8' not in sys.modules
assert 'deepopt.opt.backends.reference.gemm' not in sys.modules
assert 'deepopt.opt.backends.triton.gemm.fp8' not in sys.modules
assert 'deepopt.opt.backends.triton.quantization.fp8' not in sys.modules
assert 'deepopt.opt.backends.triton.quantization.delayed' not in sys.modules
assert 'deepopt.opt.backends.triton.quantization.activation.fp8' not in sys.modules
""")

    def test_former_autograd_import_and_patch_affect_the_actual_projection(self):
        import torch
        from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func
        from deepopt.opt.nn.functional import do_linear

        self.assertIs(linear_autograd_func, import_module("deepopt.opt.nn.autograd.linear"))
        plan = {
            "fp8_training": {
                "backend": "reference",
                "fp8_dtypes": {role: torch.float8_e4m3fn for role in ("activation", "weight", "grad")},
            },
            "block_shapes": {role: None for role in ("activation", "weight", "grad")},
        }
        target = "deepopt.opt.do_nn.AOTAutograd_Func.linear_autograd_func.quantize_fp8"
        with patch(target, side_effect=RuntimeError("legacy alias patch reached")):
            with self.assertRaisesRegex(RuntimeError, "legacy alias patch reached"):
                do_linear(torch.ones(2, 4), torch.ones(3, 4), plan=plan)

    def test_current_and_previous_projection_pickles_load_the_same_class(self):
        import torch
        from deepopt.opt.nn.linear import DOLinear
        from deepopt.opt.policy import fp8_plan

        model = DOLinear(torch.nn.Linear(4, 3), plan=fp8_plan({"enabled": True}, config={}))
        inputs = torch.randn(2, 4)
        expected = model(inputs)
        payloads = [pickle.dumps(model)]
        import_module("deepopt.opt.do_nn.linear_module")
        with patch.object(DOLinear, "__module__", "deepopt.opt.do_nn.linear_module"):
            payloads.append(pickle.dumps(model))
        for payload in payloads:
            restored = pickle.loads(payload)
            self.assertIs(type(restored), DOLinear)
            self.assertEqual(tuple(restored.state_dict()), tuple(model.state_dict()))
            torch.testing.assert_close(restored(inputs), expected)

    def test_current_and_previous_scaling_pickles_load_the_same_class(self):
        from deepopt.opt.scaling.amax import AmaxScaling

        settings = AmaxScaling((1, 32))
        payloads = [pickle.dumps(settings)]
        import_module("deepopt.opt.scaling_strategy.amax")
        with patch.object(AmaxScaling, "__module__", "deepopt.opt.scaling_strategy.amax"):
            payloads.append(pickle.dumps(settings))
        for payload in payloads:
            restored = pickle.loads(payload)
            self.assertIs(type(restored), AmaxScaling)
            self.assertEqual(restored, settings)

    def test_compatibility_files_are_isolated_behind_canonical_package_paths(self):
        import deepopt.opt
        import deepopt.opt.backends
        import deepopt.opt.nn

        packages = (
            (deepopt.opt, Path(deepopt.opt.__file__).parent / "_legacy"),
            (deepopt.opt.backends, Path(deepopt.opt.backends.__file__).parent / "_legacy"),
            (deepopt.opt.nn, Path(deepopt.opt.__file__).parent / "_legacy" / "nn"),
        )
        for package, legacy in packages:
            with self.subTest(package=package.__name__):
                paths = [Path(value) for value in package.__path__]
                self.assertEqual(paths[0], Path(package.__file__).parent)
                self.assertEqual(paths[-1], legacy)
                self.assertTrue((paths[-1] / "__init__.py").is_file())
        opt_root = Path(deepopt.opt.__file__).parent
        backend_root = Path(deepopt.opt.backends.__file__).parent
        for old, current in CPU_ALIASES + TRITON_ALIASES:
            base = "deepopt.opt.backends" if old.startswith("deepopt.opt.backends.") else "deepopt.opt"
            root = backend_root if base.endswith("backends") else opt_root
            relative = Path(*old.removeprefix(base + ".").split(".")).with_suffix(".py")
            with self.subTest(alias=old):
                self.assertTrue((root / "_legacy" / relative).is_file())
                self.assertFalse((root / relative).is_file())
                current_path = opt_root.parent.parent / Path(*current.split(".")).with_suffix(".py")
                self.assertTrue(current_path.is_file(), current_path)

    def test_reference_package_preserves_its_gemm_export(self):
        from deepopt.opt.backends.reference import gemm_nt
        from deepopt.opt.backends.reference.gemm import gemm_nt as implementation

        self.assertIs(gemm_nt, implementation)

    @unittest.skipUnless(importlib.util.find_spec("triton"), "Triton is not installed")
    def test_previous_triton_paths_resolve_to_the_same_kernel_modules(self):
        for old, current in TRITON_ALIASES:
            with self.subTest(alias=old):
                self.assertIs(import_module(old), import_module(current))


if __name__ == "__main__":
    unittest.main()
