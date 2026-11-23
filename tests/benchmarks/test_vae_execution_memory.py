"""Staged memory evidence and cleanup stay outside canonical VAE timing."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from examples.vae import benchmark_vae as benchmark
from examples.vae import execution_memory, native_execution
from tests.benchmarks.test_vae_execution_profile import _arm
from tests.benchmarks.test_vae_native_execution import _receipt


def _fake_cuda(events):
    def note(name, result=None):
        def call(*args):
            events.append(name)
            return result
        return call
    cuda = SimpleNamespace(
        get_device_properties=note("properties", SimpleNamespace(name="metadata_fixture", uuid="GPU-unit")),
        current_device=note("device_index", 0), memory=SimpleNamespace(get_allocator_backend=lambda: "native"),
        synchronize=note("synchronize"), empty_cache=note("empty_cache"),
        memory_stats=note("stats", {"inactive_split_bytes.all.current": 32, "num_alloc_retries": 2, "num_ooms": 1}),
        mem_get_info=note("free", (1000, 2000)), memory_allocated=note("allocated", 100),
        memory_reserved=note("reserved", 200), max_memory_allocated=note("peak_allocated", 300),
        max_memory_reserved=note("peak_reserved", 400))
    return SimpleNamespace(cuda=cuda), SimpleNamespace(type="cuda", index=0)


class ExecutionMemoryMetadataTests(unittest.TestCase):
    def test_cuda_metadata_preserves_peaks_counters_and_optional_process_omission(self):
        events = []
        torch, device = _fake_cuda(events)
        audit = execution_memory.ExecutionMemoryAudit()
        with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}), \
                patch.object(execution_memory, "_process_occupancy", return_value={"status": "unavailable", "processes": []}):
            audit.bind(torch, device)
            audit.prepare_baseline()
            audit.capture("after_first_update", update=1)
            audit.cleanup()
        audit.finalize(expected_stages=("before_model", "after_first_update", "cleanup"))
        row = audit.report["snapshots"]["after_first_update"]
        self.assertEqual((row["peak_allocated_bytes"], row["peak_reserved_bytes"]), (300, 400))
        self.assertEqual((row["allocation_retries"], row["ooms"], row["inactive_split_bytes"]), (2, 1, 32))
        self.assertEqual((row["device_free_bytes"], row["device_total_bytes"]), (1000, 2000))
        self.assertTrue(audit.report["complete"], audit.report)
        self.assertFalse(audit.report["errors"])
        self.assertNotIn("reset_peak_memory_stats", events)
        json.dumps(audit.report, allow_nan=False)

    def test_baseline_and_cleanup_order_and_failed_release_are_explicit(self):
        events = []
        torch, device = _fake_cuda(events)
        audit = execution_memory.ExecutionMemoryAudit()
        def fail():
            events.append("release_failure")
            raise RuntimeError("diagnostic cache release failed")
        with patch.object(execution_memory.gc, "collect", side_effect=lambda: events.append("gc")), \
                patch.object(execution_memory, "_process_occupancy", return_value={"status": "complete", "processes": []}):
            audit.bind(torch, device)
            events.clear()
            audit.prepare_baseline()
            self.assertEqual(events[:3], ["gc", "synchronize", "empty_cache"])
            events.clear()
            audit.cleanup(release_actions=(("compiler_reset", fail),))
        self.assertEqual(events[:4], ["release_failure", "gc", "synchronize", "empty_cache"])
        self.assertFalse(audit.report["cleanup"]["completed"])
        self.assertIn("cleanup", audit.report["snapshots"])
        self.assertEqual(audit.report["errors"][0]["stage"], "cleanup")
        self.assertIn("diagnostic cache release failed", audit.report["errors"][0]["traceback"])

    def test_cpu_snapshot_has_null_cuda_fields_and_missing_stages_remain_incomplete(self):
        audit = execution_memory.ExecutionMemoryAudit()
        audit.bind(SimpleNamespace(), SimpleNamespace(type="cpu", index=None))
        audit.prepare_baseline()
        audit.cleanup()
        audit.finalize(expected_stages=("before_model", "after_first_update", "cleanup"))
        self.assertFalse(audit.report["complete"])
        self.assertEqual(audit.report["missing_stages"], ["after_first_update"])
        self.assertFalse(audit.report["cuda_execution_evidence"])
        self.assertIsNone(audit.report["snapshots"]["before_model"]["device_free_bytes"])
        self.assertEqual(audit.report["scope"], "CPU_no_CUDA_allocator_evidence")

    def test_failed_device_free_query_retains_available_allocator_counters(self):
        torch, device = _fake_cuda([])
        audit = execution_memory.ExecutionMemoryAudit()
        audit.bind(torch, device)
        with patch.object(torch.cuda, "mem_get_info", side_effect=RuntimeError("free-memory query failed")), \
                patch.object(execution_memory, "_process_occupancy", return_value={"status": "unavailable", "processes": []}):
            row = audit.capture("after_first_update", update=1)
        self.assertFalse(row["complete"])
        self.assertIsNone(row["device_free_bytes"])
        self.assertEqual(row["peak_allocated_bytes"], 300)
        self.assertEqual(row["allocation_retries"], 2)
        self.assertEqual(audit.report["errors"][0]["action"], "device_free_memory")

    def test_process_inventory_is_selected_device_bounded_scalar_metadata(self):
        rows = [f"{index}, 10, GPU-unit" for index in range(66)] + ["1000, 99, GPU-other"]
        output = SimpleNamespace(returncode=0, stdout=("\n".join(rows)).encode())
        with patch.object(execution_memory.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                patch.object(execution_memory.subprocess, "run", return_value=output):
            result = execution_memory._process_occupancy("unit")
        self.assertEqual(len(result["processes"]), 64)
        self.assertEqual(result["omitted_processes"], 2)
        self.assertTrue(result["selected_device_filter_applied"])
        self.assertEqual(result["processes"][0]["used_memory_bytes"], 10 * 2**20)
        self.assertFalse(any(row["pid"] == 1000 for row in result["processes"]))

    def test_disabled_audit_does_not_query_or_clear_cuda_allocator(self):
        events = []
        torch, device = _fake_cuda(events)
        audit = execution_memory.ExecutionMemoryAudit(enabled=False)
        audit.bind(torch, device)
        audit.prepare_baseline()
        audit.capture("measurement_start")
        audit.cleanup()
        audit.finalize(expected_stages=("before_model",))
        self.assertFalse(events)
        self.assertEqual(audit.report["status"], "disabled")
        self.assertEqual(audit.report["snapshots"], {})

    def test_memory_default_and_explicit_disable_remain_stdlib_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, raw = _receipt(directory)
            for extra, enabled in (([], True), (["--no-memory-audit"], False)):
                result = subprocess.run([sys.executable, "-S", str(Path(native_execution.__file__)),
                    "--from-result", str(receipt), "--dry-run", *extra], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["memory_audit_options"]["enabled"], enabled)
                self.assertEqual(plan["target_settings"], raw)
                self.assertTrue(plan["memory_audit_options"]["first_update_profile_implemented"])
                self.assertTrue(plan["memory_audit_options"]["typed_storage_owner_ledger_implemented"])

    def test_measurement_boundary_captures_are_outside_all_timer_calls(self):
        clocks, captures = [], []
        class Schedules:
            def prepare_update(self, *args, **kwargs):
                pass
            def complete_update(self, *args):
                pass
            def __getitem__(self, key):
                return 0.1
        class Loss:
            def detach(self):
                return self
        def clock():
            clocks.append(len(clocks) + 1)
            return float(clocks[-1])
        audit = SimpleNamespace(capture=lambda stage, **kwargs: captures.append((stage, len(clocks))))
        report = {"measurement": {"samples": []}, "memory": {}, "steps": [], "updates_completed": 0}
        with patch.object(native_execution.time, "perf_counter", side_effect=clock):
            native_execution._benchmark_updates(torch=SimpleNamespace(), device=SimpleNamespace(type="cpu"),
                benchmark=SimpleNamespace(train_step=lambda *args, **kwargs: Loss()), model=object(),
                optimizer=object(), runtime=object(), images=SimpleNamespace(shape=(1,)), schedules=Schedules(),
                grad_clip_norm=None, warmup=1, updates=2, report=report, context={}, memory_audit=audit)
        self.assertEqual(captures, [("measurement_start", 0), ("measurement_end", 6)])
        self.assertEqual(report["measurement"]["wall_seconds"], 5.0)


class ExecutionMemoryInstalledCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        from deepopt import reset_config

        # In-process execution installs process defaults; restore them here.
        reset_config()
        self.addCleanup(reset_config)
        self.torch = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def arguments(self, directory, *, profile=False, execution="native"):
        from benchmarks.compiler.installed_runtime import installed_identity

        receipt, _ = _receipt(directory, tiny=True)
        arm = _arm(directory, identity=installed_identity(native_execution.ROOT, full=True))
        extra = ["--profile-after-measurement"] if profile else []
        return native_execution.parser().parse_args(["--from-result", str(receipt), "--device", "cpu",
            "--batch-size", "1", "--synthetic", "--optimizer", "sgd", "--benchmark", "--warmup", "2",
            "--updates", "2", "--execution-mode", execution, "--arm-manifest", str(arm),
            "--output", str(Path(directory) / "execution.json"), *extra])

    def test_real_installed_native_cpu_stage_order_and_tensor_owners_release(self):
        references = []
        original = benchmark.make_optimizer
        def optimizer(model, *args):
            value = original(model, *args)
            references.extend((weakref.ref(model), weakref.ref(value)))
            return value
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with patch.object(benchmark, "make_optimizer", new=optimizer), \
                    patch("examples.vae.qualification.TensorOracle.__init__", side_effect=AssertionError("oracle")), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0, report.get("error"))
            self.assertTrue(report["measurement"]["valid"])
            self.assertTrue(report["native_ownership_verified"])
            self.assertTrue(report["memory_audit"]["complete"], report["memory_audit"])
            self.assertEqual(list(report["memory_audit"]["snapshots"]), ["before_model", "after_setup",
                "after_first_update", "after_warmup", "measurement_start", "measurement_end", "after_postchecks", "cleanup"])
            self.assertEqual(report["memory_audit"]["snapshots"]["after_first_update"]["update"], 1)
            self.assertEqual(report["memory_audit"]["snapshots"]["after_warmup"]["update"], 2)
            self.assertIn("examples/vae/execution_memory.py", report["helper_sha256"])
            self.assertTrue(report["memory_audit"]["cleanup"]["completed"])
            self.assertFalse(report["memory_audit"]["cuda_execution_evidence"])
            self.assertFalse([path for path in Path(directory).rglob("*")
                if path.suffix in {".pt", ".pth", ".pkl", ".pickle", ".npy", ".safetensors"}])
        self.assertTrue(references)
        self.assertTrue(all(reference() is None for reference in references))

    def test_profile_and_cleanup_failure_preserve_frozen_measurement(self):
        import deepopt

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory, profile=True)
            with patch.object(deepopt, "shutdown_distributed", side_effect=RuntimeError("cleanup diagnostic failed")), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0, report.get("error"))
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["measurement"]["valid"])
            self.assertTrue(report["profile"]["complete"], report["profile"])
            frozen = json.loads(Path(report["measurement_frozen_artifact"]["path"]).read_text())
            self.assertEqual(report["measurement"], frozen["measurement"])
            self.assertEqual(report["memory"], frozen["memory"])
            self.assertEqual(report["cuda_allocator_diagnostics"], frozen["cuda_allocator_diagnostics"])
            self.assertFalse(report["memory_audit"]["cleanup"]["completed"])
            snapshots = report["memory_audit"]["snapshots"]
            self.assertLess(snapshots["after_postchecks"]["monotonic_ns"], snapshots["profile_start"]["monotonic_ns"])
            self.assertLess(snapshots["profile_end"]["monotonic_ns"], snapshots["cleanup"]["monotonic_ns"])
            self.assertEqual(report["memory_audit"]["errors"][-1]["action"], "shutdown_distributed")

    def test_original_forward_failure_survives_secondary_cleanup_failure(self):
        import deepopt

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with patch.object(benchmark, "train_step", side_effect=RuntimeError("original forward failure")), \
                    patch.object(deepopt, "shutdown_distributed", side_effect=RuntimeError("secondary cleanup failure")), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertIn("original forward failure", report["error"])
            self.assertIn("original forward failure", report["traceback"])
            self.assertFalse(report["execution_completed"])
            self.assertIn("execution_failure", report["memory_audit"]["snapshots"])
            self.assertFalse(report["memory_audit"]["complete"])
            self.assertIn("after_first_update", report["memory_audit"]["missing_stages"])
            self.assertIn("secondary cleanup failure", report["memory_audit"]["errors"][-1]["error"])

    def test_fresh_real_eager_cpu_worker_records_its_own_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory, execution="eager")
            command = [sys.executable, "-m", "examples.vae.native_execution", "--from-result", str(args.from_result),
                "--device", "cpu", "--batch-size", "1", "--synthetic", "--optimizer", "sgd", "--benchmark",
                "--warmup", "1", "--updates", "1", "--execution-mode", "eager", "--arm-manifest", str(args.arm_manifest),
                "--output", str(args.output)]
            result = subprocess.run(command, cwd=native_execution.ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(args.output.read_text())
            self.assertTrue(report["eager_ownership_verified"])
            self.assertTrue(report["memory_audit"]["complete"])
            self.assertNotEqual(report["memory_audit"]["process"]["pid"], os.getpid())
            self.assertEqual(report["memory_audit"]["process"]["ppid"], os.getpid())
            self.assertTrue(report["helper_source_stable"])
            self.assertTrue(report["installed_stable"])


@unittest.skipUnless(os.environ.get("DEEPOPT_RUN_H200_MEMORY_AUDIT") == "1",
                     "Original B128 CUDA memory audit is explicitly opt-in")
class ExecutionMemoryAuditOriginalShapeCudaTests(unittest.TestCase):
    def setUp(self):
        from deepopt import reset_config

        reset_config()
        self.addCleanup(reset_config)

    def test_original_first_update_and_cleanup_allocator_evidence(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        receipt = os.environ.get("DEEPOPT_VAE_MEMORY_TEST_RESULT")
        output = os.environ.get("DEEPOPT_VAE_MEMORY_TEST_OUTPUT")
        if not receipt or not output:
            self.fail("Set DEEPOPT_VAE_MEMORY_TEST_RESULT and DEEPOPT_VAE_MEMORY_TEST_OUTPUT to fresh target paths")
        extra = []
        arm = os.environ.get("DEEPOPT_VAE_MEMORY_TEST_ARM_MANIFEST")
        if arm:
            extra = ["--arm-manifest", arm]
        args = native_execution.parser().parse_args(["--from-result", receipt, "--output", output,
            "--device", "cuda:0", "--batch-size", "128", "--benchmark", "--warmup", "1", "--updates", "1", *extra])
        self.assertEqual(native_execution.run_native_execution(args), 0)
        report = json.loads(Path(output).read_text())
        self.assertTrue(report["physical_target_executed"])
        self.assertTrue(report["native_ownership_verified"])
        self.assertTrue(report["memory_audit"]["complete"])
        self.assertTrue(report["memory_audit"]["cuda_execution_evidence"])
        snapshots = report["memory_audit"]["snapshots"]
        self.assertEqual(snapshots["before_model"]["allocated_bytes"], 0)
        self.assertEqual(snapshots["before_model"]["reserved_bytes"], 0)
        self.assertGreater(snapshots["after_first_update"]["peak_allocated_bytes"], 0)
        self.assertTrue(report["memory_audit"]["cleanup"]["completed"])
        self.assertGreater(snapshots["cleanup"]["device_free_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
