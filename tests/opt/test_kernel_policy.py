"""Backend choices are explicit, immutable and independent of installed compilers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
import subprocess
import sys
import unittest

from deepopt.kernel_policy import (
    KernelConfig, current_kernel_policy, kernel_backend, kernel_backward,
    kernel_config_from_dict, kernel_forward, kernel_policy_scope,
    resolve_kernel_policy, select_kernel_backend,
)


class KernelPolicyTests(unittest.TestCase):
    def test_default_and_explicit_pure_modes(self):
        self.assertEqual(current_kernel_policy().backend, "triton")
        for backend in ("triton", "tilelang"):
            policy = resolve_kernel_policy(backend)
            self.assertEqual(policy, resolve_kernel_policy(policy.as_dict()))
            self.assertEqual(kernel_backend("gemm", backend=policy), backend)
        self.assertEqual(resolve_kernel_policy(fp8_backend="tilelang").backend, "tilelang")

    def test_mixed_precedence_and_input_ownership(self):
        values = {"backend": "mixed", "preferred": "triton", "families": {"gemm": "tilelang"},
                  "roles": {"linear.dw": "triton"}}
        policy = resolve_kernel_policy(values)
        values["families"]["gemm"] = "triton"
        self.assertEqual(policy.route("gemm"), "tilelang")
        self.assertEqual(policy.route("gemm", "linear.dw"), "triton")
        self.assertEqual(policy.route("activation"), "triton")
        self.assertEqual(policy.fingerprint, resolve_kernel_policy(policy.as_dict()).fingerprint)
        with kernel_policy_scope(policy):
            self.assertEqual(kernel_backend("gemm", backend="mixed"), "tilelang")

    def test_invalid_policies_and_conflicts(self):
        for values in ({"backend": "auto"}, {"backend": "triton", "families": {"gemm": "tilelang"}},
                       {"backend": "mixed", "families": {"typo": "triton"}},
                       {"backend": "mixed", "roles": {"linear.typo": "triton"}},
                       {"backend": "mixed", "preferred": "reference"}, {"surprise": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                kernel_config_from_dict(values)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            resolve_kernel_policy("mixed", fp8_backend="triton")
        with kernel_policy_scope("tilelang"), self.assertRaisesRegex(ValueError, "conflicts"):
            kernel_backend("gemm", backend="triton")

    def test_only_metadata_eligibility_may_fallback_and_pins_are_strict(self):
        policy = resolve_kernel_policy({"backend": "mixed", "fallback": "other"})
        self.assertEqual(select_kernel_backend("gemm", {"triton": "layout", "tilelang": None}, policy=policy), "tilelang")
        pinned = resolve_kernel_policy({"backend": "mixed", "fallback": "other", "families": {"gemm": "triton"}})
        with self.assertRaisesRegex(ValueError, "layout"):
            select_kernel_backend("gemm", {"triton": "layout", "tilelang": None}, policy=pinned)
        with self.assertRaisesRegex(ValueError, "no capability decision"):
            select_kernel_backend("gemm", {}, policy=policy)

    def test_runtime_dispatch_applies_only_declared_metadata_fallback(self):
        from unittest.mock import patch
        from deepopt.opt.backends import dispatch

        first = SimpleNamespace(run=lambda value: value + 1,
                                unsupported_reason=lambda operation, value: "layout")
        second = SimpleNamespace(run=lambda value: value * 2)
        policy = resolve_kernel_policy({"backend": "mixed", "fallback": "other"})
        with kernel_policy_scope(policy), patch.object(dispatch, "find_spec", return_value=object()), \
             patch.object(dispatch, "import_module", side_effect=[first, second]):
            self.assertEqual(dispatch.call_kernel("quantization", "quantization.fp8.run", 3), 6)
        broken = SimpleNamespace(run=lambda value: (_ for _ in ()).throw(RuntimeError("launch failed")))
        with kernel_policy_scope(policy), patch.object(dispatch, "find_spec", return_value=object()), \
             patch.object(dispatch, "import_module", return_value=broken) as imported:
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                dispatch.call_kernel("quantization", "quantization.fp8.run", 3)
            self.assertEqual(imported.call_count, 1)

    def test_scope_exception_cleanup_and_thread_isolation(self):
        with kernel_policy_scope("tilelang"):
            self.assertEqual(kernel_backend("quantization"), "tilelang")
            with ThreadPoolExecutor(1) as pool:
                self.assertEqual(pool.submit(lambda: kernel_backend("quantization")).result(), "triton")
            with self.assertRaises(RuntimeError):
                with kernel_policy_scope("mixed"):
                    raise RuntimeError("user failure")
            self.assertEqual(current_kernel_policy().backend, "tilelang")
        self.assertEqual(current_kernel_policy().backend, "triton")

    def test_autograd_owner_survives_scope_exit_and_worker_thread(self):
        @kernel_forward
        def forward(ctx):
            return kernel_backend("activation")

        @kernel_backward
        def backward(ctx):
            return kernel_backend("activation")

        ctx = SimpleNamespace()
        with kernel_policy_scope("tilelang"):
            self.assertEqual(forward(ctx), "tilelang")
        with ThreadPoolExecutor(1) as pool:
            self.assertEqual(pool.submit(backward, ctx).result(), "tilelang")

    def test_config_import_and_selection_load_neither_torch_nor_compilers(self):
        code = """
import sys
from deepopt.config import RuntimeConfig
from deepopt.kernel_policy import resolve_kernel_policy
assert RuntimeConfig().kernels.backend == 'inherit'
assert resolve_kernel_policy('tilelang').backend == 'tilelang'
assert all(name not in sys.modules for name in ('torch', 'triton', 'tilelang'))
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_configuration_roundtrip_and_fp8_compatibility(self):
        from deepopt._config_resolution import resolve_config
        for backend in ("triton", "tilelang", "mixed"):
            config = resolve_config({"runtime": {"kernels": {"backend": backend},
                                                "fp8": {"enabled": True, "backend": "inherit"}}}, base=None)
            self.assertEqual(config["runtime"]["kernels"]["backend"], backend)
        self.assertEqual(asdict(KernelConfig())["backend"], "inherit")

    def test_explicit_mode_switch_discards_only_inherited_mixed_routes(self):
        from deepopt._config_resolution import resolve_config
        base = {"runtime": {"kernels": {"backend": "mixed", "preferred": "tilelang",
                    "families": {"gemm": "triton"}, "roles": {"linear.dw": "tilelang"}}}}
        for mode in ("inherit", "triton", "tilelang"):
            snapshot = resolve_config({"runtime": {"kernels": {"backend": mode}}}, base=base)
            self.assertEqual(snapshot["runtime"]["kernels"]["families"], {})
            self.assertEqual(snapshot["runtime"]["kernels"]["roles"], {})
        snapshot = resolve_config({"runtime": {"kernels": {"preferred": "triton"}}}, base=base)
        self.assertEqual(snapshot["runtime"]["kernels"]["families"], {"gemm": "triton"})
        with self.assertRaises(ValueError):
            resolve_config({"runtime": {"kernels": {"backend": "triton", "families": {"gemm": "tilelang"}}}}, base=base)


if __name__ == "__main__":
    unittest.main()
