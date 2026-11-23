"""Fresh-process receipts qualify memory comparisons independently of timing."""

from copy import deepcopy
import unittest

from examples.vae.memory_comparison import CUDA_FIELDS, compare_memory_baselines, worker_memory_errors


def audited_row(*, device="cuda:0", free=140 * 2**30, arm="A", pid=100):
    snapshots = {name: dict.fromkeys(CUDA_FIELDS, 0) for name in ("before_model", "after_setup", "after_first_update",
        "after_warmup", "measurement_start", "measurement_end", "after_postchecks", "cleanup")}
    for stage in snapshots.values():
        stage.update(device_free_bytes=free, device_total_bytes=141 * 2**30, synchronized=True, complete=True)
        if device == "cpu":
            stage.update(dict.fromkeys(CUDA_FIELDS))
            stage["synchronized"] = False
    return {"device": device, "arm": arm, "block": 1, "memory_audit": {
        "schema_version": 1, "enabled": True, "complete": True, "status": "complete",
        "process": {"pid": pid, "ppid": 1},
        "snapshots": snapshots, "errors": [], "cleanup": {"attempted": True, "completed": True}}}


class MemoryComparisonTests(unittest.TestCase):
    def test_comparable_fresh_start_and_completed_cleanup(self):
        rows = [audited_row(arm=name, free=140 * 2**30 - index * 2**20, pid=100+index)
                for index, name in enumerate("ABC")]
        result = compare_memory_baselines(rows)
        self.assertTrue(result["valid"], result["invalid_reasons"])
        self.assertEqual(result["starting_free_range_bytes"], 2 * 2**20)
        self.assertIn("CUDA", result["evidence_scope"])
        self.assertFalse(result["worker_cleanup_is_context_exit"])

    def test_external_occupancy_drift_suppresses_a_comparison(self):
        rows = [audited_row(), audited_row(arm="C", free=138 * 2**30)]
        self.assertFalse(compare_memory_baselines(rows)["valid"])
        # A changed limit must be explicitly declared and retained in evidence.
        result = compare_memory_baselines(rows, tolerance_bytes=2 * 2**30)
        self.assertTrue(result["valid"])
        self.assertEqual(result["tolerance_bytes"], 2 * 2**30)

    def test_live_or_reserved_start_and_failed_cleanup_are_rejected(self):
        for failure in ("allocated_bytes", "reserved_bytes", "cleanup", "missing_stage", "disabled", "error",
                        "unsynchronized", "incomplete", "incomplete_stage", "missing_counter"):
            with self.subTest(failure=failure):
                row = audited_row()
                audit = row["memory_audit"]
                if failure in ("allocated_bytes", "reserved_bytes"):
                    audit["snapshots"]["before_model"][failure] = 2**20
                elif failure == "cleanup":
                    audit["cleanup"]["completed"] = False
                elif failure == "missing_stage":
                    audit["snapshots"].pop("measurement_end")
                elif failure == "disabled":
                    audit["enabled"] = False
                elif failure == "error":
                    audit["errors"].append({"stage": "cleanup", "error": "injected"})
                elif failure == "incomplete":
                    audit.update(complete=False, status="incomplete")
                elif failure == "incomplete_stage":
                    audit["snapshots"]["measurement_end"]["complete"] = False
                elif failure == "missing_counter":
                    audit["snapshots"]["measurement_end"].pop("allocation_retries")
                else:
                    audit["snapshots"]["before_model"]["synchronized"] = False
                self.assertTrue(worker_memory_errors(row))
                self.assertFalse(compare_memory_baselines([row])["valid"])

    def test_malformed_receipts_and_bool_byte_counts_do_not_qualify(self):
        row = audited_row()
        cases = [None, [], {"enabled": True}, deepcopy(row["memory_audit"])]
        cases[-1]["snapshots"]["before_model"]["allocated_bytes"] = False
        for value in cases:
            with self.subTest(value=value):
                self.assertTrue(worker_memory_errors({"device": "cuda:0", "memory_audit": value}))
        for tolerance in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                compare_memory_baselines([row], tolerance_bytes=tolerance)

    def test_cpu_diagnostic_does_not_fabricate_device_bytes(self):
        result = compare_memory_baselines([audited_row(device="cpu")])
        self.assertTrue(result["valid"])
        self.assertEqual(result["starting_free_bytes"], [])
        self.assertEqual(result["evidence_scope"], "CPU_diagnostic_no_CUDA_allocator_evidence")
        self.assertFalse(compare_memory_baselines([])["valid"])


if __name__ == "__main__":
    unittest.main()
