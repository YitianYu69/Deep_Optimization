"""Actual installed CPU training checks premeasurement failure routing only."""

from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.compiler.installed_runtime import installed_identity
from examples.vae import benchmark_packed_offload, native_execution
from tests.benchmarks import test_vae_packed_offload_benchmark as fixtures


class PackedOffloadBudgetPreflightTests(unittest.TestCase):
    def setUp(self):
        import torch
        from deepopt import reset_config

        reset_config()
        self.addCleanup(reset_config)
        self.identity = installed_identity(native_execution.ROOT, full=True)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def arguments(self, directory):
        factory = fixtures.PackedOffloadInstalledCPUTests()
        factory.identity = self.identity
        return factory.arguments(directory)

    def test_invalid_budget_snapshot_blocks_timing_after_real_native_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with patch.object(benchmark_packed_offload, "runtime_budget_errors",
                    return_value=["physical_pinned_budget_unverified_or_exceeded"]) as validate, \
                    patch.object(native_execution, "_benchmark_updates",
                        side_effect=AssertionError("Measured updates ran after failed budget preflight")) as measured, \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            row = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertTrue(row["native_ownership_verified"])
            self.assertEqual(row["warmup_completed"], 1)
            self.assertEqual(row["updates_completed"], 0)
            self.assertEqual(row["measurement"]["samples"], [])
            self.assertFalse(row["measurement"]["valid"])
            self.assertFalse(row["packed_offload_budget_preflight"]["valid"])
            self.assertEqual(row["failure_context"]["phase"], "packed_offload_budget_preflight")
            self.assertIn("before measurement", row["error"])
            validate.assert_called_once()
            self.assertIs(validate.call_args.args[0]["available"], True)
            measured.assert_not_called()

    def test_real_resident_snapshot_passes_preflight_and_native_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with patch.object(benchmark_packed_offload, "runtime_budget_errors",
                    wraps=benchmark_packed_offload.runtime_budget_errors) as validate, redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            row = json.loads(args.output.read_text())
            self.assertEqual(code, 0, row.get("traceback"))
            self.assertTrue(row["packed_offload_budget_preflight"]["valid"])
            self.assertEqual(row["packed_offload_budget_preflight"]["errors"], [])
            self.assertTrue(row["measurement"]["valid"])
            self.assertEqual(row["updates_completed"], 2)
            validate.assert_called_once()
            self.assertFalse(row["memory_audit"]["cuda_execution_evidence"])


if __name__ == "__main__":
    unittest.main()
