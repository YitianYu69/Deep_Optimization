"""Bootstrap receipt handling with the full recorded recipe; no CUDA claims."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.vae import benchmark_packed_offload_frontier as frontier
from examples.vae import execution_memory
from tests.benchmarks.test_vae_packed_offload_frontier import setup


class FrontierBootstrapFailureTests(unittest.TestCase):
    def invoke(self, directory, *, worker, code=0, inventory=None, mutate_inventory=None):
        args, plan, valid_inventory = setup(directory)
        original = args.from_result.read_bytes()
        raw = json.loads(original)["settings"]
        self.assertIsNotNone(raw["runtime"]["compile"]["mode"])
        self.assertTrue(raw["runtime"]["compile"]["options"]["require_full_aot"])
        self.assertEqual(plan["workers"]["demand_offload"]["requested_input_shape"], [128, 3, 224, 224])
        if inventory == "valid":
            inventory = valid_inventory
        if mutate_inventory is not None:
            inventory = mutate_inventory(deepcopy(valid_inventory))
        def run(command, folder, label, environment):
            output = Path(command[command.index("--output") + 1])
            output.write_text(worker if isinstance(worker, str) else json.dumps(worker))
            if inventory is not None:
                target = Path(command[command.index("--inventory-output") + 1])
                target.write_text(inventory if isinstance(inventory, str) else json.dumps(inventory))
            log = folder / (label + ".log")
            log.write_text("simulated child I/O; no model or CUDA execution\n")
            return code, frontier._ref(log)
        with patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                patch.object(frontier, "_run", side_effect=run):
            result = frontier._bootstrap(args, plan, Path(directory), {})
        self.assertEqual(args.from_result.read_bytes(), original)
        return result

    def successful(self):
        return {"status": "completed", "execution_completed": True, "native_ownership_verified": True,
            "failure_context": {"stage": "complete", "phase": "done", "route": "native"}}

    def test_child_primary_export_error_and_context_survive_completed_training(self):
        worker = {**self.successful(), "status": "failed", "error": "RuntimeError: compact inventory unavailable",
            "failure_context": {"stage": "finalization", "phase": "inventory_export", "route": "native"}}
        with tempfile.TemporaryDirectory() as directory:
            value, report = self.invoke(directory, worker=worker, code=1)
            self.assertIsNone(value)
            self.assertEqual(report["error"], worker["error"])
            self.assertEqual(report["failure_context"], worker["failure_context"])
            self.assertTrue(report["worker_execution_completed"])
            self.assertTrue(report["worker_native_ownership_verified"])
            self.assertEqual(report["status"], "failed")
            self.assertEqual(frontier._ref(Path(report["worker"]["path"])), report["worker"])

    def test_malformed_or_nonobject_worker_keeps_raw_reference_and_exit_status(self):
        for worker in ("{", "[]"):
            with self.subTest(worker=worker), tempfile.TemporaryDirectory() as directory:
                value, report = self.invoke(directory, worker=worker, code=7, inventory="valid")
                self.assertIsNone(value)
                self.assertEqual(report["worker_exit_code"], 7)
                self.assertEqual(report["failure_context"]["phase"], "worker_receipt")
                self.assertIn("exit=7", report["error"])
                self.assertIn("worker", report)
                self.assertIn("log", report)
                self.assertFalse(report.get("worker_execution_completed", False))

    def test_missing_invalid_oversized_or_stale_compact_inventory_fails_closed(self):
        cases = ((None, None, "inventory_receipt"), ("{", None, "inventory_validation"),
            ('{"noise":"' + "x" * 2**20 + '"}', None, "inventory_validation"),
            ("valid", lambda value: {**value, "installed": {}}, "inventory_validation"))
        for inventory, mutate, phase in cases:
            with self.subTest(phase=phase, inventory_type=type(inventory).__name__), tempfile.TemporaryDirectory() as directory:
                value, report = self.invoke(directory, worker=self.successful(), inventory=inventory, mutate_inventory=mutate)
                self.assertIsNone(value)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["failure_context"]["phase"], phase)
                self.assertEqual(report["worker_context"]["phase"], "done")
                self.assertTrue(report["worker_execution_completed"])
                self.assertTrue(report["worker_native_ownership_verified"])

    def test_completion_and_native_owner_gates_still_reject_an_existing_valid_inventory(self):
        for field, value in (("execution_completed", False), ("native_ownership_verified", False), ("status", "failed")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                worker = {**self.successful(), field: value}
                result, report = self.invoke(directory, worker=worker, inventory="valid")
                self.assertIsNone(result)
                self.assertEqual(report["status"], "failed")
                self.assertIn("worker_completion", report["receipt_diagnostics"][0]["phase"])

    def test_large_raw_receipt_keeps_existing_capacity_and_successful_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = {**self.successful(), "unrelated_metadata": "x" * (2**20 + 1)}
            value, report = self.invoke(directory, worker=worker, inventory="valid")
            self.assertEqual(report["status"], "completed")
            self.assertEqual(value["status"], "available")
            self.assertGreater(report["worker"]["bytes"], 2**20)
            self.assertNotIn("failure_context", report)

    def test_compact_diagnostics_bound_long_child_error_and_retain_original_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = {**self.successful(), "status": "failed", "error": "root cause " + "x" * 5000,
                "failure_context": {"stage": "backward", "phase": "native", "detail": "y" * 2000}}
            value, report = self.invoke(directory, worker=worker, code=1)
            self.assertIsNone(value)
            self.assertEqual(len(report["error"]), 4096)
            self.assertTrue(report["error_truncated"])
            self.assertLessEqual(len(report["failure_context"]["detail"]), 1024)
            original = json.loads(Path(report["worker"]["path"]).read_text())
            self.assertEqual(original["error"], worker["error"])

    def test_parent_terminal_reports_child_cause_without_creating_timing_points(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = setup(directory)
            args.inventory_json = None
            failed = {"status": "failed", "worker_exit_code": 1,
                "error": "RuntimeError: schema-directed inventory export failed",
                "failure_context": {"stage": "finalization", "phase": "inventory_export"},
                "worker": {"path": "raw.json", "bytes": 1, "sha256": "a" * 64},
                "log": {"path": "raw.log", "bytes": 1, "sha256": "b" * 64}}
            output = io.StringIO()
            with patch.object(frontier, "_bootstrap", return_value=(None, failed)), \
                    patch("benchmarks.compiler.installed_runtime.child_environment", return_value={}), \
                    redirect_stdout(output):
                self.assertEqual(frontier.run_frontier(args), 1)
            report = json.loads(args.output.read_text())
            terminal = json.loads(output.getvalue())
            self.assertEqual(report["inventory_bootstrap"], failed)
            self.assertEqual(report["points"], [])
            self.assertFalse(report["maximum_all_eligible_measured"])
            self.assertIn(failed["error"], terminal["error"])
            self.assertEqual(terminal["bootstrap_failure"]["failure_context"], failed["failure_context"])
            self.assertEqual(terminal["bootstrap_failure"]["worker"], failed["worker"])

    def test_interruption_is_not_converted_to_a_failed_worker_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, _ = setup(directory)
            with patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                    patch.object(frontier, "_run", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                frontier._bootstrap(args, plan, Path(directory), {})

    def test_missing_worker_retains_log_and_expected_receipt_path(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, _ = setup(directory)
            log = Path(directory) / "missing-worker.log"
            log.write_text("Child exited before producing JSON; simulated metadata only\n")
            with patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                    patch.object(frontier, "_run", return_value=(9, frontier._ref(log))):
                value, report = frontier._bootstrap(args, plan, Path(directory), {})
            self.assertIsNone(value)
            self.assertEqual(report["worker_exit_code"], 9)
            self.assertEqual(report["failure_context"]["phase"], "worker_receipt")
            self.assertIn("expected_worker_path", report)
            self.assertNotIn("worker", report)
            self.assertEqual(report["log"], frontier._ref(log))


if __name__ == "__main__":
    unittest.main()
