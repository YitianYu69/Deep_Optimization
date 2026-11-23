"""Two-rank native/reducer contracts; CUDA transport is a separate opt-in gate."""

import json
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import distributed as dist

import deepopt
from tests.distributed.launcher import run_torchrun


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("_packed_offload_worker.py")
CUDA_ENV = "DEEPOPT_TEST_PACKED_OFFLOAD_DDP_CUDA"


def _environment(*, cuda=False):
    source = Path(deepopt.__file__).resolve().is_relative_to(ROOT / "src")
    if source:
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    else:
        from benchmarks.compiler.installed_runtime import child_environment

        environment = child_environment(ROOT)
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                 "TORCHDYNAMO_REPRO_AFTER"):
        environment.pop(name, None)
    environment.update(OMP_NUM_THREADS="1", DEEPOPT_LOGGING_LEVEL="ERROR")
    if not cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment, source


def _read_records(directory, processes=2):
    return [json.loads(path.read_text()) for rank in range(processes)
            if (path := Path(directory, f"rank-{rank}.json")).is_file()]


def _assert_completed(test, records, *, cuda):
    test.assertEqual(len(records), 2)
    for rank, record in enumerate(records):
        test.assertEqual(record["status"], "completed", record.get("exception"))
        test.assertEqual((record["rank"], record["global_rank"], record["local_rank"], record["world_size"]),
                         (rank, rank, rank, 2))
        test.assertEqual(record["backend"], "nccl" if cuda else "gloo")
        test.assertEqual(record["cuda_execution_verified"], cuda)
        test.assertEqual(record["test_transport"], not cuda)
        test.assertFalse(record["group_initialized_after_shutdown"])
        test.assertEqual(record["tensor_files_saved"], 0)
        test.assertTrue(record["checks"])
        test.assertTrue(all(record["checks"].values()), record["checks"])
        test.assertEqual(record["installed"], record["installed_after"])
        cleanup = record["runtime_after_cleanup"]
        test.assertEqual(cleanup["managers"], [])
        test.assertTrue(cleanup["totals"]["lifecycle"]["quiescent"])
        test.assertNotIn("cleanup_exception", record)
        controls = record["algorithm_controls_before_numerics"]
        test.assertTrue(controls["cudnn_enabled"])
        test.assertTrue(controls["deterministic_algorithms"])
        test.assertFalse(controls["cudnn_benchmark"])
        test.assertTrue(controls["cudnn_deterministic"])
        test.assertFalse(controls["cudnn_allow_tf32"])
        test.assertFalse(controls["matmul_allow_tf32"])
        names = {case["name"] for case in record["cases"]}
        if cuda:
            test.assertTrue({"native_demand", "native_memory_first_overlap", "checkpoint_demand",
                "checkpoint_memory_first_overlap", "native_demand_torch_amp", "checkpoint_demand_torch_amp",
                "preflight", "forward_error"} <= names)
            test.assertFalse(any(case["test_transport"] for case in record["cases"]))
            test.assertTrue(record["checks"]["rank_local_cuda_rng_restored"])
        else:
            test.assertTrue({"native_resident", "checkpoint_resident", "generated_testing_transport",
                             "preflight", "forward_error"} <= names)
            test.assertTrue(record["checks"]["cpu_offload_rejected"])
            test.assertTrue(record["checks"]["generated_joint_saved_abi"])
        for name in ("native_fullgraph", "native_autodiff", "fp32_parameter_gradients",
                     "exact_manual_all_reduce_oracle", "no_sync_accumulation", "retained_backward_in_no_sync",
                     "gradient_bucket_views", "checkpoint_restore", "checkpoint_next_update",
                     "policy_mismatch_cooperative", "cooperative_forward_error", "no_backward_after_forward_error",
                     "multiple_ddp_buckets", "rank_local_rng_restored"):
            test.assertTrue(record["checks"][name], name)
        test.assertTrue(record["checks"]["whole_model_offload_plan_preserved" if cuda else "ddp_optimizer_graph_split"])


class PackedOffloadManagerMetadataTests(unittest.TestCase):
    """Actual CPU test-transport snapshots; these do not qualify CUDA copies."""

    def _metadata(self):
        import gc

        from deepopt.compiler.runtime.manager import CompiledGraphRuntimeManager
        from deepopt.compiler.runtime.packed_offload import TestingTransportBackend
        from tests.compiler.test_packed_offload_lifecycle import _SavedFunction, fixture
        with patch.dict(os.environ):
            from tests.distributed._packed_offload_worker import _audit_transport_managers

        plan = fixture(version=2, host_pinned_bytes=4096)
        backend = TestingTransportBackend()
        managers = [CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False) for _ in range(2)]
        idle = CompiledGraphRuntimeManager(plan.policy, start_worker=False)
        for manager in (*managers, idle):
            self.addCleanup(manager.close)
        for index, manager in enumerate(managers):
            frame = manager.begin_forward(plan)
            with frame.saved_token_scope():
                output = _SavedFunction.apply(frame, torch.ones(2, requires_grad=True), torch.ones(1))
            output.sum().backward()
            del output, frame
            gc.collect()
            manager.freeze_profiles()
            if index == 0:
                manager.close()
        idle.freeze_profiles()
        rows = [managers[0].snapshot(), managers[1].snapshot(), idle.snapshot()]
        self.assertTrue(all(row["testing_backend"] for row in rows[:2]))
        return _audit_transport_managers, {"managers": rows}, rows[1]

    def test_closed_history_shares_verified_arena_without_becoming_another_owner(self):
        audit, snapshot, owned = self._metadata()
        closed = snapshot["managers"][0]
        self.assertTrue(closed["plans"])
        self.assertTrue(closed["lifecycle"]["closed"])
        self.assertEqual(closed["host"]["pinned_physical_capacity_bytes"], 4096)
        manager, metadata = audit(snapshot, owned)
        self.assertIs(manager, snapshot["managers"][1])
        self.assertEqual(metadata, dict(unused_zero_resource_facades=1, closed_zero_resource_histories=0,
                                       closed_shared_allocator_histories=1))

    def test_closed_history_cannot_hide_current_resources_or_another_arena_domain(self):
        audit, snapshot, owned = self._metadata()
        mutations = (("lifecycle", "graph_leases", 1), ("lifecycle", "task_leases", 1),
            ("lifecycle", "dma_leases", 1), ("lifecycle", "quiescent", False),
            ("host", "pinned_active_bytes", 1), ("host", "pageable_cached_bytes", 1),
            ("host", "pinned_physical_capacity_bytes", 8192),
            ("host", "pinned_allocator_domain_id", owned["host"]["pinned_allocator_domain_id"] + 999),
            ("host", "pinned_accounting_verified", False), ("device", "staging_bytes", 1),
            ("device", "pending_retire_bytes", 1), ("device", "d2h_inflight_bytes", 1))
        for section, key, value in mutations:
            with self.subTest(section=section, field=key):
                changed = deepcopy(snapshot)
                changed["managers"][0][section][key] = value
                with self.assertRaises(AssertionError):
                    audit(changed, owned)

    def test_unused_facade_requires_complete_frozen_zero_resource_schema(self):
        audit, snapshot, owned = self._metadata()
        for section, key, value in (("host", "pinned_active_bytes", 1), ("device", "staging_peak_bytes", 1),
                                    ("lifecycle", "task_leases", 1)):
            with self.subTest(section=section, field=key):
                changed = deepcopy(snapshot)
                changed["managers"][2][section][key] = value
                with self.assertRaises(AssertionError):
                    audit(changed, owned)
        for key, value in (("counters", {"d2h_bytes": 1}), ("profile_frozen", False),
                           ("profile_learning", True), ("error", "failure")):
            with self.subTest(field=key):
                changed = deepcopy(snapshot)
                changed["managers"][2][key] = value
                with self.assertRaises(AssertionError):
                    audit(changed, owned)


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class PackedOffloadDistributedTests(unittest.TestCase):
    def test_two_rank_native_and_generated_testing_transport_contracts(self):
        environment, source = _environment()
        with tempfile.TemporaryDirectory() as directory:
            command = [str(WORKER), "--device", "cpu", "--output-dir", directory]
            if source:
                command.append("--allow-source")
            result = run_torchrun(command, env=environment, timeout=180)
            records = _read_records(directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + json.dumps(records))
            _assert_completed(self, records, cuda=False)
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["rank-0.json", "rank-1.json"])

    def test_four_rank_independent_bucket_shaped_native_oracle(self):
        environment, source = _environment()
        with tempfile.TemporaryDirectory() as directory:
            command = [str(WORKER), "--device", "cpu", "--case", "native", "--output-dir", directory]
            if source:
                command.append("--allow-source")
            result = run_torchrun(command, env=environment, timeout=180, processes=4)
            records = _read_records(directory, 4)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + json.dumps(records))
            self.assertEqual(len(records), 4)
            for rank, record in enumerate(records):
                self.assertEqual(record["status"], "completed", record.get("exception"))
                self.assertEqual((record["rank"], record["world_size"], record["backend"]), (rank, 4, "gloo"))
                self.assertFalse(record["cuda_execution_verified"])
                self.assertTrue(all(record["checks"].values()), record["checks"])
                native = next(case for case in record["cases"] if case["name"] == "native_resident")
                collective = [row for row in native["comparisons"] if "independent_bucket_collective_ABI" in row["role"]]
                self.assertEqual(len(collective), 2)
                for row in collective:
                    self.assertTrue(row["same_sized_zero_bucket_views_filled_only_from_reference_gradients"])
                self.assertTrue(any(len(row["buckets"]) >= 2 for row in collective))
                self.assertTrue(all(row["bitwise_equal"] for row in native["comparisons"] if "bitwise_equal" in row))
                self.assertEqual(record["runtime_after_cleanup"]["managers"], [])
                self.assertFalse(record["group_initialized_after_shutdown"])

    def test_explicit_cuda_requirement_never_substitutes_cpu(self):
        environment, source = _environment()
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(WORKER), "--device", "cpu", "--require-cuda", "--output-dir", directory]
            if source:
                command.append("--allow-source")
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(Path(directory, "rank-0.json").is_file(), result.stdout + result.stderr)
            record = json.loads(Path(directory, "rank-0.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("CPU substitution is forbidden", record["exception"]["message"])
            self.assertFalse(record["cuda_execution_verified"])
            self.assertEqual(record["cases"], [])


@unittest.skipUnless(os.environ.get(CUDA_ENV) == "1", f"set {CUDA_ENV}=1 for real two-GPU NCCL qualification")
class PackedOffloadDistributedCudaTests(unittest.TestCase):
    def test_two_rank_nccl_native_offload_checkpoint_and_failure_contracts(self):
        self.assertTrue(torch.cuda.is_available(), "Explicit CUDA qualification requires CUDA")
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "Explicit CUDA qualification requires two visible GPUs")
        self.assertTrue(dist.is_nccl_available(), "Explicit CUDA qualification requires NCCL")
        environment, source = _environment(cuda=True)
        self.assertFalse(source, "CUDA qualification must import the installed candidate wheel")
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(WORKER), "--device", "cuda", "--require-cuda", "--output-dir", directory],
                                 env=environment, timeout=300)
            records = _read_records(directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + json.dumps(records))
            _assert_completed(self, records, cuda=True)
            self.assertEqual([record["device"] for record in records], ["cuda:0", "cuda:1"])
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["rank-0.json", "rank-1.json"])


if __name__ == "__main__":
    unittest.main()
