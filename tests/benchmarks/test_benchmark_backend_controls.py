"""Shared backend controls preserve explicit routes without loading compilers."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepopt._benchmarking import BenchmarkBackend
from deepopt.kernel_policy import current_kernel_policy


class BenchmarkBackendControlTests(unittest.TestCase):
    def test_lazy_import_in_fresh_process(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-c", """
import sys
from deepopt._benchmarking import BenchmarkBackend
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
assert 'tilelang' not in sys.modules
"""], env={**os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_and_cpu_do_not_query_optional_distributions(self):
        selection = BenchmarkBackend.from_args(SimpleNamespace())
        with patch("deepopt._benchmarking.version", side_effect=AssertionError("package query")):
            metadata = selection.metadata("cpu")
        self.assertEqual(selection.policy.backend, "triton")
        self.assertEqual(selection.fp8_backend("cpu"), "reference")
        self.assertEqual(metadata["execution"], "cpu_reference")
        self.assertEqual(metadata["packages"], {})

    def test_modes_scope_restore_and_configuration_are_owned(self):
        original = current_kernel_policy()
        for mode in ("triton", "tilelang", "mixed"):
            mapping = {"preferred": "tilelang", "roles": {"linear.dw": "triton"}} if mode == "mixed" else {}
            selection = BenchmarkBackend.from_args(SimpleNamespace(deepopt_backend=mode, deepopt_kernel_policy=mapping))
            self.assertEqual(selection.fp8_backend("cuda:1"), "inherit")
            with selection.scope():
                self.assertEqual(current_kernel_policy(), selection.policy)
            self.assertEqual(current_kernel_policy(), original)
            config = selection.config
            config["runtime"]["kernels"]["backend"] = "broken"
            self.assertEqual(selection.config["runtime"]["kernels"]["backend"], mode)

    def test_mixed_metadata_includes_role_routes_and_queries_selected_packages_only(self):
        selection = BenchmarkBackend.from_args(SimpleNamespace(deepopt_backend="mixed", deepopt_kernel_policy=json.dumps({
            "preferred": "tilelang", "families": {"normalization": "triton"}, "roles": {"linear.dw": "triton"}})))
        with patch("deepopt._benchmarking.version", side_effect=lambda name: name + "-version") as query:
            metadata = selection.metadata("cuda:1")
        self.assertEqual(metadata["resolved_families"]["gemm"], "tilelang")
        self.assertEqual(metadata["resolved_families"]["normalization"], "triton")
        self.assertEqual(metadata["role_overrides"]["linear.dw"], "triton")
        self.assertEqual({call.args[0] for call in query.call_args_list}, {"tilelang", "triton"})
        self.assertEqual(metadata["route_evidence"], "policy_resolution")
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        with patch("deepopt._benchmarking.version", return_value="version") as query:
            BenchmarkBackend.from_args(SimpleNamespace()).metadata("cuda")
        query.assert_called_once_with("triton")
        fallback = BenchmarkBackend.from_args(SimpleNamespace(deepopt_backend="mixed", deepopt_kernel_policy={"fallback": "other"}))
        with patch("deepopt._benchmarking.version", return_value="version") as query:
            self.assertEqual(set(fallback.metadata("cuda")["packages"]), {"triton", "tilelang"})

    def test_invalid_policies_fail_before_device_work(self):
        for mode, raw in (("missing", None), ("mixed", []), ("mixed", "null"), ("mixed", "bad-json"),
                          ("triton", {"backend": "tilelang"}), ("tilelang", {"preferred": "tilelang"}),
                          ("mixed", {"families": {"unknown": "triton"}}),
                          ("mixed", {"roles": {"linear.dw": "auto"}}), ("mixed", {"extra": True})):
            with self.subTest(mode=mode, policy=raw), self.assertRaises(ValueError):
                BenchmarkBackend.from_args(SimpleNamespace(deepopt_backend=mode, deepopt_kernel_policy=raw))


if __name__ == "__main__":
    unittest.main()
