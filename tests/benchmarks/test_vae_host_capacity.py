"""Read-only host/job capacity fixtures, without CUDA or allocation claims."""

from pathlib import Path
import tempfile
import unittest

from examples.vae.execution_memory import host_capacity_admission, host_capacity_snapshot


class HostCapacityTests(unittest.TestCase):
    def fixture(self, temporary, *, available=1024, scope="/slurm/job_42/step_0"):
        root = Path(temporary)
        proc, cgroup = root / "proc", root / "cgroup"
        (proc / "self").mkdir(parents=True)
        (proc / "meminfo").write_text(f"MemAvailable: {available} kB\n")
        (proc / "self/cgroup").write_text("0::" + scope + "\n")
        cgroup.mkdir()
        (cgroup / "memory.max").write_text("max\n")
        return proc, cgroup

    def limit(self, folder, limit, current):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "memory.max").write_text(str(limit))
        (folder / "memory.current").write_text(str(current))

    def test_cgroup_ancestors_and_headroom_bound_physical_capacity_without_clamping(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc, cgroup = self.fixture(temporary, available=8192)
            self.limit(cgroup / "slurm/job_42", 6 * 2**20, 2 * 2**20)
            self.limit(cgroup / "slurm/job_42/step_0", 5 * 2**20, 2 * 2**20)
            snapshot = host_capacity_snapshot(proc_root=proc, cgroup_root=cgroup, environment={})
        self.assertEqual(snapshot["available_bytes"], 3 * 2**20)
        accepted = host_capacity_admission(2 * 2**20, headroom_bytes=2**20, snapshot=snapshot)
        rejected = host_capacity_admission(2 * 2**20 + 1, headroom_bytes=2**20, snapshot=snapshot)
        self.assertTrue(accepted["admitted"])
        self.assertEqual(rejected["status"], "resource_limited")
        self.assertFalse(rejected["requested_capacity_clamped"])
        self.assertEqual(rejected["required_physical_capacity_bytes"], 2 * 2**20 + 1)

    def test_slurm_uses_whole_job_usage_not_smaller_step_and_exact_job_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc, cgroup = self.fixture(temporary, available=16384)
            self.limit(cgroup / "slurm/job_42", 10 * 2**20, 5 * 2**20)
            self.limit(cgroup / "slurm/job_42/step_0", 10 * 2**20, 2**20)
            snapshot = host_capacity_snapshot(proc_root=proc, cgroup_root=cgroup,
                environment={"SLURM_JOB_ID": "42", "SLURM_MEM_PER_NODE": "8"})
            wrong_job = host_capacity_snapshot(proc_root=proc, cgroup_root=cgroup,
                environment={"SLURM_JOB_ID": "4", "SLURM_MEM_PER_NODE": "8"})
        self.assertEqual(snapshot["slurm"]["available_bytes"], 3 * 2**20)
        self.assertEqual(snapshot["available_bytes"], 3 * 2**20)
        self.assertEqual(wrong_job["status"], "unknown")
        self.assertIsNone(wrong_job["available_bytes"])

    def test_unreadable_finite_limit_missing_RAM_or_unknown_job_usage_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc, cgroup = self.fixture(temporary)
            (cgroup / "memory.max").write_text("1024")
            incomplete = host_capacity_snapshot(proc_root=proc, cgroup_root=cgroup, environment={})
            self.assertTrue(incomplete["unresolved_limits"])
            self.assertEqual(incomplete["status"], "unknown")
            (cgroup / "memory.max").write_text("max")
            (proc / "meminfo").unlink()
            missing = host_capacity_snapshot(proc_root=proc, cgroup_root=cgroup, environment={})
            self.assertEqual(missing["status"], "unknown")
            self.assertFalse(host_capacity_admission(1, headroom_bytes=1, snapshot=missing)["admitted"])
            (proc / "meminfo").write_text("MemAvailable: 100000 kB\n")
            unscoped = host_capacity_snapshot(proc_root=proc, cgroup_root=cgroup,
                environment={"SLURM_JOB_ID": "42", "SLURM_MEM_PER_CPU": "10", "SLURM_CPUS_ON_NODE": "2"})
            self.assertEqual(unscoped["status"], "unknown")
            self.assertEqual(unscoped["slurm"]["requested_limit_bytes"], 20 * 2**20)

    def test_negative_or_malformed_limits_and_invalid_budgets_are_not_capacity_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc, cgroup = self.fixture(temporary, available=-1)
            invalid = host_capacity_snapshot(proc_root=proc, cgroup_root=cgroup,
                environment={"SLURM_MEM_PER_NODE": "invalid"})
        self.assertEqual(invalid["status"], "unknown")
        self.assertIsNone(invalid["available_bytes"])
        for required, headroom in ((True, 0), (0, False), (-1, 0), (0, -1)):
            with self.subTest(required=required, headroom=headroom), self.assertRaises(ValueError):
                host_capacity_admission(required, headroom_bytes=headroom, snapshot={})


if __name__ == "__main__":
    unittest.main()
