"""Failure attribution and host-only telemetry contracts; no CUDA qualification."""

from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from examples.vae import benchmark_vae as benchmark
from examples.vae import qualification


class VAEFailureDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        from deepopt import reset_config

        # In-process execution installs process defaults; restore them here.
        reset_config()
        self.addCleanup(reset_config)

    def test_allocator_snapshot_is_failure_only_best_effort_and_never_synchronizes(self):
        stats = {"inactive_split_bytes.all.current": 17, "inactive_split_bytes.all.peak": 23,
                 "num_alloc_retries": 2, "num_ooms": 1}
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_name", return_value="mock H200"), \
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
                patch.object(torch.cuda, "mem_get_info", return_value=(300, 400)), \
                patch.object(torch.cuda, "memory_allocated", side_effect=RuntimeError("allocator unavailable")), \
                patch.object(torch.cuda, "memory_reserved", return_value=200), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=150), \
                patch.object(torch.cuda, "max_memory_reserved", return_value=250), \
                patch.object(torch.cuda, "memory_stats", return_value=stats), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("must not synchronize")), \
                patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("must not reset")), \
                patch.dict("os.environ", {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}):
            result = benchmark.cuda_failure_diagnostics("cuda:0")
        self.assertIn("allocator unavailable", result["errors"]["allocated_bytes"])
        self.assertEqual(result["reserved_bytes"], 200)
        self.assertEqual(result["gpu_name"], "mock H200")
        self.assertEqual(result["device_memory"], {"free_bytes": 300, "total_bytes": 400})
        self.assertEqual(result["peak_allocated_bytes"], 150)
        self.assertEqual(result["allocator_stats"]["num_ooms"], 1)
        self.assertEqual(result["allocator_stats"]["inactive_split_bytes.all.current"], 17)
        self.assertEqual(result["allocator_environment"]["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
        self.assertIn("cudnn_benchmark", result["torch_flags"])
        self.assertIn("after exception stack unwind", result["counter_scope"])
        self.assertFalse(result["synchronized"])
        json.dumps(result)

    def test_gate_progress_identifies_eager_backward_failure_before_candidate(self):
        eager = torch.nn.Linear(2, 2)
        native = deepcopy(eager)
        optimizers = [torch.optim.SGD(model.parameters(), lr=.1) for model in (eager, native)]
        events, executed = [], []
        original = RuntimeError("simulated backward OOM")

        def fail(model, optimizer, images, beta, compression, sink, *, progress, **kwargs):
            executed.append(model)
            progress({"phase": "forward"})
            progress({"phase": "backward"})
            raise original

        with patch.object(qualification, "emit_update", side_effect=fail), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CPU gate cannot synchronize")):
            with self.assertRaises(RuntimeError) as caught:
                qualification.paired_trajectory(eager, native, *optimizers, torch.ones(1, 2), [0.],
                    dict(enabled=False), device="cpu", progress=events.append)
        self.assertIs(caught.exception, original)
        self.assertEqual(executed, [eager])
        self.assertEqual(events[-1], {"phase": "backward", "route": "eager", "update": 1})

    def test_update_progress_tracks_optimizer_and_does_not_query_cuda(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

            def forward(self, x):
                value = x * self.scale
                return value, value, value * .01

        model = Tiny()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        events = []
        with patch.object(torch.cuda, "memory_stats", side_effect=AssertionError("success cannot query allocator")), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("success cannot synchronize")):
            qualification.emit_update(model, optimizer, torch.ones(1, 3, 2, 2), 0., dict(enabled=False),
                                      lambda *_: None, progress=events.append)
        phases = [event["phase"] for event in events]
        self.assertLess(phases.index("forward"), phases.index("backward"))
        self.assertLess(phases.index("backward"), phases.index("optimizer"))
        self.assertEqual(phases[-1], "complete")

    def test_progress_observer_errors_do_not_change_update_behavior(self):
        qualification._notify_progress(Mock(side_effect=RuntimeError("observer failed")), "forward")

    def test_worker_gate_failure_records_route_source_notes_and_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parser().parse_args([])
            args._case, args._repeat = "deepopt-bf16-native-compressed", 0
            args.model_factory = "fixture:Tiny"
            args._snapshot, args._result = Path(directory) / "snapshot.json", Path(directory) / "result.json"
            args._snapshot.write_text(json.dumps(benchmark.settings(args)), encoding="utf-8")
            context = SimpleNamespace(device=torch.device("cuda:0"), is_main_process=True)
            original = RuntimeError("original eager failure")
            original.add_note("failure belongs to the eager oracle")
            provenance = {"source_sha256": "before-gate-source"}

            def fail(*args, progress, **kwargs):
                progress({"phase": "backward", "route": "eager", "update": 1})
                raise original

            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
                stack.enter_context(patch.object(torch.cuda, "is_bf16_supported", return_value=True))
                stack.enter_context(patch("benchmarks.compiler.installed_runtime.installed_identity", return_value={}))
                stack.enter_context(patch("deepopt.initialize_distributed", return_value=context))
                stack.enter_context(patch("deepopt.shutdown_distributed"))
                stack.enter_context(patch("deepopt.runtime.seed_everything"))
                stack.enter_context(patch.object(benchmark, "build_loader", return_value=[torch.ones(1, 3, 2, 2)]))
                stack.enter_context(patch.object(benchmark, "import_factory", return_value=lambda **_: torch.nn.Linear(2, 2)))
                stack.enter_context(patch.object(benchmark, "source_provenance", return_value=provenance))
                stack.enter_context(patch.object(benchmark, "make_optimizer", return_value=Mock()))
                stack.enter_context(patch("deepopt.opt.policy.prepare_fp8_model", side_effect=lambda model, *a, **k: (model, None)))
                stack.enter_context(patch("deepopt.engine.execution.compilation.compile_model"))
                stack.enter_context(patch.object(qualification, "preserve_rng", side_effect=nullcontext))
                stack.enter_context(patch.object(qualification, "eager_dtype_diagnostic", return_value={
                    "complete": True, "storage_policy_validation": {"valid": True}}))
                stack.enter_context(patch.object(qualification, "paired_trajectory", side_effect=fail))
                stack.enter_context(patch.object(benchmark, "cuda_failure_diagnostics", return_value={"errors": {"stats": "unavailable"}}))
                stack.enter_context(redirect_stdout(io.StringIO()))
                with self.assertRaises(RuntimeError) as caught:
                    benchmark.worker(args)
            self.assertIs(caught.exception, original)
            result = json.loads(args._result.read_text(encoding="utf-8"))
            self.assertEqual(result["failure_context"],
                             dict(stage="matching_recipe_gate", phase="backward", route="eager", update=1))
            self.assertEqual(result["source_sha256"], "before-gate-source")
            self.assertEqual(result["exception_notes"], ["failure belongs to the eager oracle"])
            self.assertIn("fail", result["traceback"])
            self.assertEqual(result["status"], "failed")

    def test_running_aggregate_does_not_label_unstarted_modes_as_failed(self):
        result = benchmark.aggregate_runs([], ["deepopt-bf16-native-compressed"], 1, running=True)
        self.assertEqual(result[0]["status"], "pending")
        self.assertEqual(benchmark.aggregate_runs([], ["deepopt-bf16-native-compressed"], 1)[0]["status"], "failed")
        failure = dict(mode="deepopt-bf16-native-compressed", repeat=0, status="failed")
        self.assertEqual(benchmark.aggregate_runs([failure], [failure["mode"]], 2, running=True)[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
