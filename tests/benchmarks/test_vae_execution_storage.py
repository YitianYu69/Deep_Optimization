"""Real CPU diagnostic storage tracing; CUDA capacity remains a target gate."""

from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from examples.vae.execution_profile import run_execution_profile
from examples.vae.execution_storage import NativeStorageTrace, join_allocator_trace


class ExecutionStorageTests(unittest.TestCase):
    def setUp(self):
        from deepopt.config import reset_config

        reset_config()
        self.addCleanup(reset_config)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_actual_lazy_native_VAE_first_update_is_profiled_without_preprofile_forward(self):
        import deepopt
        from deepopt.compiler import compilation_report
        from examples.vae import benchmark_vae
        from examples.vae.mbconv_vae import MBConvVAE_V2

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        model = MBConvVAE_V2(base_channels=2, max_channels=8, latent_channels=1, blocks_per_stage=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        raw = {"runtime": {"device": "cpu", "precision": "fp32", "bf16_policy": "torch_amp", "checks": False,
            "activation_compression": {"enabled": True, "bits": 2, "group_size": 1024},
            "activation_memory": {"multiply_backward": "materialized", "pack_reuse": "off", "reduction_scratch_bytes": 4096},
            "compile": {"enabled": True, "backend": "deepopt", "fullgraph": True,
                "options": {"use_native_autodiff": True, "require_full_aot": True}}}}
        runtime = deepopt.TrainingRuntime(model, optimizer, device="cpu", precision="fp32", config=raw)
        model = runtime.model
        before = compilation_report(model)
        self.assertTrue(before is None or not before.graphs)
        images = torch.randn(1, 3, 16, 16)
        trace = NativeStorageTrace(torch, images.device)
        trace.observe_persistent(model, optimizer, images)
        calls = []
        def update(number, progress):
            calls.append(number)
            benchmark_vae.train_step(model, optimizer, runtime, images, 0., device_type="cpu", progress=progress)
        with tempfile.TemporaryDirectory() as temporary:
            result = run_execution_profile(torch=torch, device=images.device, directory=Path(temporary) / "cold",
                update=update, first_update=1, updates=1, record_memory=True, storage_trace=trace,
                diagnostic_phase="cold", allow_initial_compilation=True)
            self.assertTrue(result["complete"], result.get("traceback") or result.get("omissions"))
            self.assertEqual(calls, [1])
            self.assertGreater(trace.invocations, 1)
            self.assertFalse(result["cuda_execution_evidence"])
            self.assertFalse(images.requires_grad)
            self.assertIsNone(images.grad)
            ownership = benchmark_vae.native_compilation_evidence(model)
            self.assertEqual(ownership["captured_graphs"], 1)
            self.assertEqual(ownership["runtime_fallbacks"], 0)
            self.assertEqual(ownership["legacy_backward_operations"], [])

    def test_real_native_CPU_forward_backward_profiler_and_weak_storage_ranges(self):
        torch.manual_seed(723)
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.SiLU(), nn.Conv2d(4, 3, 1))
        graph = torch.fx.symbolic_trace(deepcopy(model))
        optimizer = torch.optim.SGD(graph.parameters(), lr=.01)
        images = torch.randn(2, 3, 5, 7)
        trace = NativeStorageTrace(torch, images.device)
        trace.observe_persistent(graph, optimizer, images)
        report = CompilationReport()
        ShapeProp(graph).propagate(images)
        executable = compile_fx(graph, [images], config=BackendConfig(require_full_aot=True), report=report)
        def update(number, progress):
            progress({"phase": "forward_and_loss"})
            output = executable(images)
            progress({"phase": "backward"})
            output.sum().backward()
            progress({"phase": "optimizer"})
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as temporary:
            result = run_execution_profile(torch=torch, device=images.device,
                directory=Path(temporary) / "cold", update=update, first_update=1, updates=1,
                record_memory=True, storage_trace=trace, diagnostic_phase="cold", allow_initial_compilation=True)
            self.assertTrue(result["complete"], result.get("traceback"))
            self.assertFalse(result["cuda_execution_evidence"])
            self.assertEqual(result["first_update"], 1)
            ledger = json.loads(Path(result["storage_ledger"]["path"]).read_text())
            self.assertGreater(ledger["native_invocations"], 1)
            self.assertEqual({row["phase"] for row in ledger["boundaries"]}, {"forward", "backward"})
            self.assertTrue(any(row["physical_address"] for row in ledger["ledger"]["storages"]))
            self.assertTrue(all(type(row["native_region_index"]) is int for row in ledger["boundaries"]))
            self.assertTrue(all(len(row["native_forward_signature"]) == 64 for row in ledger["boundaries"]))
            self.assertTrue(any(row["storage_byte_span"][1] > 0 for row in ledger["ledger"]["logical_values"]))
            self.assertFalse(ledger["attribution_complete"])
            self.assertEqual(report.runtime_fallbacks, 0)
            self.assertFalse(images.requires_grad)
            self.assertIsNone(images.grad)
            self.assertFalse([path for path in Path(temporary).rglob("*") if path.suffix in (".pt", ".pkl", ".npy")])
            events = json.loads((Path(temporary) / "cold/trace.json").read_text())["traceEvents"]
            self.assertTrue(any(str(row.get("name", "")).startswith("deepopt.storage.backward.") for row in events))

    def test_observer_does_not_retain_persistent_tensor_owners_and_aliases_deduplicate(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        images = torch.randn(4, 3)
        trace = NativeStorageTrace(torch, images.device)
        trace.observe_persistent(model, optimizer, images)
        trace.ledger.observe(images[:, :1], owner="alias")
        image_ref, model_ref = weakref.ref(images), weakref.ref(model)
        del images, model, optimizer
        gc.collect()
        self.assertIsNone(image_ref())
        self.assertIsNone(model_ref())
        trace.finish()
        self.assertEqual(trace.as_dict()["ledger"]["live_visible_bytes_by_device"], {})

    def test_real_CPU_transport_bridge_forwards_runtime_kwargs_and_captured_backward_observer(self):
        from deepopt.compiler.autodiff.joint import _OffloadedJointFunction
        from tests.compiler.test_compiled_packed_offload import _cpu_fixture

        source, region, _, plan, manager = _cpu_fixture()
        trace = NativeStorageTrace(torch, source.device, runtime_snapshot=manager.snapshot)
        try:
            frame = manager.begin_forward(plan)
            with trace.capture(), frame.saved_token_scope():
                output, = _OffloadedJointFunction.apply(region, frame, source)
            # The observer is captured by the real bridge, rather than relying
            # on a forward-only ContextVar propagating to the engine thread.
            output.sum().backward()
            manager.drain()
            trace.finish()
            report = trace.as_dict()
            self.assertEqual(report["native_invocations"], 2)
            self.assertEqual({row["phase"] for row in report["boundaries"]}, {"forward", "backward"})
            self.assertFalse(report["cuda_execution_evidence"])
            self.assertIsNotNone(source.grad)
            self.assertTrue(manager.snapshot()["lifecycle"]["quiescent"])
        finally:
            manager.close()

    def test_allocator_join_requires_address_generation_and_native_range_and_keeps_unknowns(self):
        storage = {"ledger": {"storages": [dict(storage_id=1, generation=1, device="cuda:0",
            physical_address=4096, birth_phase="deepopt.storage.backward.1", owners=["role:native_adjoint_or_temporary"]),
            dict(storage_id=2, generation=2, device="cuda:0", physical_address=4096,
                 birth_phase="deepopt.storage.backward.2", owners=["role:parameter_gradient"])]}}
        events = [{"name": "deepopt.storage.backward.1", "ph": "X", "ts": 10, "dur": 5},
                  {"name": "deepopt.storage.backward.2", "ph": "X", "ts": 30, "dur": 5}]
        for timestamp, address, total in ((12, 4096, 100), (32, 4096, 200), (50, 9999, 300)):
            events.append({"name": "[memory]", "ts": timestamp,
                "args": {"Device Type": 1, "Device Id": 0, "Addr": address, "Bytes": 100, "Total Allocated": total}})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            result = join_allocator_trace(path, storage)
        self.assertEqual([row["storage_generations"] for row in result["events"]], [[1], [2], []])
        self.assertEqual(result["unattributed_events"], 1)
        self.assertEqual(result["peak_allocation_event"]["address"], 9999)
        self.assertFalse(result["attribution_complete"])

    def test_byte_bound_is_an_explicit_incomplete_receipt(self):
        trace = NativeStorageTrace(torch, torch.device("cpu"))
        value = torch.empty(2)
        trace.ledger.observe(value)
        with tempfile.TemporaryDirectory() as temporary, patch("examples.vae.execution_storage.MAX_JSON_BYTES", 1):
            receipt = trace.save(temporary)
            result = json.loads(Path(receipt["path"]).read_text())
        self.assertEqual(result["omission"], "storage_ledger_byte_limit")
        self.assertEqual(receipt["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
