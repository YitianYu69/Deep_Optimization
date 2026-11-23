"""Startup admission and real Gloo agreement; CPU metadata is not CUDA evidence."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import distributed as dist

from deepopt._packed_offload import resolve_packed_offload
from deepopt.distributed import _process, offload
from tests.distributed._packed_offload_policy_worker import descriptor
from tests.distributed.launcher import run_torchrun


def policy(**changes):
    return resolve_packed_offload(dict(version=2, enabled=True, policy="demand",
        host_pinned_bytes=32 << 30, device_staging_bytes=4 << 30, d2h_inflight_bytes=2 << 30) | changes)


def context(rank=0, size=1):
    return _process.DistributedContext(torch.device("cpu"), rank, rank, size, size,
                                       "gloo" if size > 1 else None, size > 1, False)


class PackedOffloadStartupPolicyTests(unittest.TestCase):
    def test_disabled_real_local_context_does_not_inspect_recipe_cuda_or_host_capacity(self):
        with patch.object(offload, "_host_capacity_snapshot", side_effect=AssertionError("disabled capacity probe")), \
                patch("torch.cuda.is_available", side_effect=AssertionError("disabled CUDA probe")):
            receipt = offload.prepare_packed_offload(None, context(), recipe={"callback": object()})
        self.assertEqual(receipt["status"], "disabled")
        self.assertTrue(receipt["admitted"])
        self.assertIsNone(receipt["recipe_sha256"])
        self.assertEqual(receipt["nodes"][0]["requested_host_bytes"], 0)
        self.assertFalse(receipt["cuda_execution_verified"])
        json.dumps(receipt, allow_nan=False)

    def test_enabled_cpu_execution_and_stale_context_fail_before_host_probe(self):
        with patch.object(offload, "_host_capacity_snapshot", side_effect=AssertionError("CPU must reject first")):
            with self.assertRaisesRegex(ValueError, "CUDA execution"):
                offload.prepare_packed_offload(policy(), context(), recipe={"compile": "native"})
        with self.assertRaisesRegex(ValueError, "active default process group"):
            offload.prepare_packed_offload(None, context(rank=1, size=2))

    def test_recipe_is_bounded_owned_metadata_without_tensor_or_callback_serialization(self):
        raw = dict(precision="bf16", compression={"bits": 2, "group_size": 1024}, dtype=torch.bfloat16)
        first = offload._recipe_json(raw, enabled=True)
        self.assertEqual(first, offload._recipe_json(deepcopy(raw), enabled=True))
        raw["compression"]["bits"] = 4
        self.assertNotEqual(first, offload._recipe_json(raw, enabled=True))
        for bad in (None, {}, {"payload": torch.ones(1)}, {"callback": lambda: None},
                    {"bad": float("nan")}, {"large": "x" * 65536}, {"many": list(range(8192))}):
            with self.subTest(kind=type(bad).__name__), self.assertRaises(ValueError):
                offload._recipe_json(bad, enabled=True)

    def test_rank_local_cuda_observation_requires_current_device_without_allocating(self):
        selected = policy(host_pinned_bytes=4096, device_staging_bytes=4096, d2h_inflight_bytes=4096)
        ctx = replace(context(), device=torch.device("cuda:1"))
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2), \
                patch("torch.cuda.current_device", return_value=0), \
                patch("torch.cuda.get_device_properties", side_effect=AssertionError("mismatch before query")):
            with self.assertRaisesRegex(ValueError, "current visible device"):
                offload._local_descriptor(selected, ctx)


class PackedOffloadHostCapacityTests(unittest.TestCase):
    def fixture(self, root):
        proc, cg = root / "proc", root / "cgroup"
        (proc / "self").mkdir(parents=True)
        (proc / "meminfo").write_text(f"MemAvailable: {(400 << 30)//1024} kB\n")
        (proc / "self/cgroup").write_text("0::/job_12/task_0\n")
        task = cg / "job_12/task_0"
        task.mkdir(parents=True)
        for folder, limit, current in ((cg, "max", 0), (cg / "job_12", 200 << 30, 20 << 30),
                                       (task, 120 << 30, 5 << 30)):
            (folder / "memory.max").write_text(str(limit))
            (folder / "memory.current").write_text(str(current))
        return proc, cg

    def test_actual_file_reader_preserves_scoped_remaining_and_job_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, cg = self.fixture(Path(directory))
            value = offload._host_capacity_snapshot(proc_root=proc, cgroup_root=cg,
                environment={"SLURM_JOB_ID": "12", "SLURM_MEM_PER_NODE": str(200 * 1024)})
        self.assertEqual(value["status"], "available")
        self.assertEqual(value["mem_available_bytes"], 400 << 30)
        rows = {row["scope_id"]: row for row in value["constraints"]}
        self.assertEqual(rows["memory.max:job_12/task_0"]["available_bytes"], 115 << 30)
        self.assertEqual(rows["memory.max:job_12"]["available_bytes"], 180 << 30)
        self.assertEqual(rows["slurm_job:12"]["current_bytes"], 20 << 30)
        self.assertFalse(value["snapshot_is_reservation"])

    def test_unreadable_observed_limits_fail_closed_and_unknown_job_usage_is_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, cg = self.fixture(Path(directory))
            unknown_job = offload._host_capacity_snapshot(proc_root=proc, cgroup_root=cg,
                environment={"SLURM_JOB_ID": "1", "SLURM_MEM_PER_NODE": str(200 * 1024)})
            job = next(row for row in unknown_job["constraints"] if row["kind"] == "slurm")
            self.assertIsNone(job["current_bytes"], "job_1 cannot match job_12")
            self.assertFalse(job["remaining_observed"])
            self.assertTrue(unknown_job["warnings"])
            (cg / "job_12/memory.current").unlink()
            invalid = offload._host_capacity_snapshot(proc_root=proc, cgroup_root=cg, environment={})
            self.assertEqual(invalid["status"], "unknown")
            self.assertTrue(invalid["errors"])
            missing = offload._host_capacity_snapshot(proc_root=Path(directory) / "missing", cgroup_root=cg, environment={})
            self.assertEqual(missing["status"], "unknown")

    def test_four_rank_200GiB_job_refuses_128GiB_each_but_accepts_32GiB_each(self):
        for budget, fits in ((128 << 30, False), (32 << 30, True)):
            with self.subTest(per_rank=budget), patch.object(_process, "world_size", return_value=4):
                selected = policy(host_pinned_bytes=budget)
                rows = [descriptor(selected, context(rank, 4), scope_limit=200 << 30) for rank in range(4)]
                if fits:
                    nodes = offload._validate_descriptors(rows, selected)
                    self.assertEqual(nodes[0]["requested_host_bytes"], 128 << 30)
                    self.assertEqual(nodes[0]["constraints"][0]["margin_bytes"], 72 << 30)
                else:
                    with self.assertRaisesRegex(ValueError, "requires.*observed bound"):
                        offload._validate_descriptors(rows, selected)

    def test_private_subgroups_are_not_charged_the_entire_node_request(self):
        selected = policy(host_pinned_bytes=64 << 30)
        rows = [descriptor(selected, context(rank, 4), scope_limit=300 << 30) for rank in range(4)]
        for rank, row in enumerate(rows):
            row["capacity"]["constraints"].append(dict(scope_id=f"private:{rank}", kind="cgroup",
                limit_bytes=100 << 30, current_bytes=0, available_bytes=100 << 30, remaining_observed=True))
        with patch.object(_process, "world_size", return_value=4):
            nodes = offload._validate_descriptors(rows, selected)
        self.assertEqual(nodes[0]["requested_host_bytes"], 256 << 30)
        self.assertTrue(all(row["requested_host_bytes"] == 64 << 30 for row in nodes[0]["constraints"] if row["scope_id"].startswith("private")))

    def test_uuid_is_authoritative_with_rank_private_visible_indices_and_nodes(self):
        selected = policy(host_pinned_bytes=32 << 30)
        rows = [descriptor(selected, context(rank, 2)) for rank in range(2)]
        for row in rows:
            row.update(device="cuda:0", device_index=0)
        with patch.object(_process, "world_size", return_value=2):
            self.assertEqual(len(offload._validate_descriptors(rows, selected)), 1)
            private_launchers = deepcopy(rows)
            for row in private_launchers:
                row.update(local_rank=0, local_world_size=1)
            self.assertFalse(offload._validate_descriptors(private_launchers, selected)[0]["launcher_local_topology_matches_node"])
            duplicate = deepcopy(rows)
            duplicate[1].update(device_uuid=duplicate[0]["device_uuid"], device_identity=duplicate[0]["device_identity"])
            with self.assertRaisesRegex(ValueError, "distinct CUDA devices"):
                offload._validate_descriptors(duplicate, selected)
            for rank, row in enumerate(rows):
                row.update(node_id=f"node-{rank}", local_rank=0, local_world_size=1)
                row["capacity"]["mem_available_bytes"] = 40 << 30
            self.assertEqual(len(offload._validate_descriptors(rows, selected)), 2)

    def test_pageable_and_pinned_are_both_charged_and_minimum_shared_observation_wins(self):
        selected = policy(host_pinned_bytes=32 << 30, host_pageable_bytes=32 << 30)
        rows = [descriptor(selected, context(rank, 2), scope_limit=130 << 30) for rank in range(2)]
        rows[1]["capacity"]["constraints"][0].update(current_bytes=4 << 30, available_bytes=126 << 30)
        with patch.object(_process, "world_size", return_value=2), self.assertRaisesRegex(ValueError, "requires.*observed bound"):
            offload._validate_descriptors(rows, selected)


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class PackedOffloadCollectivePolicyTests(unittest.TestCase):
    def test_actual_two_rank_gloo_agreement_rejection_and_scoped_budget_arithmetic(self):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}}
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(Path(__file__).with_name("_packed_offload_policy_worker.py")), directory],
                                 env=environment, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(Path(directory, f"rank-{rank}.json").read_text()) for rank in range(2)]
        for rank, rows in enumerate(records):
            self.assertEqual(len(rows), 10)
            for row in rows:
                with self.subTest(rank=rank, scenario=row["scenario"]):
                    self.assertTrue(row["cpu_gloo_only"])
                    self.assertFalse(row["CUDA_execution_evidence"])
                    passed = row["scenario"] in ("disabled", "metadata_admitted")
                    self.assertEqual(row["status"], "passed" if passed else "rejected", row)
                    if passed:
                        self.assertFalse(row["receipt"]["cuda_execution_verified"])
                        self.assertEqual(row["receipt"]["world_size"], 2)
                    elif row["scenario"] == "enabled_disabled_mismatch":
                        self.assertIn("matching packed activation offload policies", row["error"])
                    elif row["scenario"] == "enabled_CPU_rejected":
                        self.assertIn("CUDA execution", row["error"])
                    elif row["scenario"] == "local_descriptor_failure":
                        self.assertIn("injected local descriptor failure", row["error"])


if __name__ == "__main__":
    unittest.main()
