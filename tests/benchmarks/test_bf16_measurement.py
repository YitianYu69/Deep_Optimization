"""Saved-storage lifetime and BF16 benchmark control regressions."""

import gc
import importlib.util
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("bf16_measurement", ROOT / "benchmarks/analysis/measurement.py")
measurement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measurement)


class BF16MeasurementTests(unittest.TestCase):
    def test_memory_diagnostics_preserves_fullgraph_without_dispatch_observer(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        model = torch.nn.Linear(4, 3)
        call = torch.compile(model, backend="eager", fullgraph=True)
        value = torch.randn(2, 4, requires_grad=True)

        def clear():
            model.zero_grad(set_to_none=True)
            value.grad = None

        def forward():
            output = call(value)
            return output, output.square().mean()

        def step():
            clear()
            _, loss = forward()
            loss.backward()

        session = SimpleNamespace(model=model, device=torch.device("cpu"), inputs=(value,),
            runtime={"execution": "compile", "accumulation_steps": 1}, reset=clear, clear=clear,
            step=step, forward=forward, backward=lambda loss: loss.backward(), optimizer=None,
            after_step=None, last_output=None)
        result = measurement.memory_diagnostics(session)
        self.assertFalse(result["dispatch_observation_enabled"])
        self.assertFalse(result["all_intermediate_storage_observed"])
        self.assertIn("fullgraph", result["dispatch_observation_reason"])
        self.assertTrue(result["saved_tensors"])
        self.assertIsNotNone(result["invocation_storage_ledger"])

    def profile_model(self, *, execution="eager", in_place=False, detail="trace"):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.GELU(), torch.nn.Linear(6, 3))
        reference = deepcopy(model)
        inputs = torch.randn(2, 4, requires_grad=True)
        expected_inputs = inputs.detach().clone().requires_grad_()
        reference(expected_inputs).square().mean().backward()
        call = model
        if execution == "compile":
            if in_place:
                model.compile(backend="eager", fullgraph=True)
            else:
                call = torch.compile(model, backend="eager", fullgraph=True)

        def step(annotate=False):
            model.zero_grad(set_to_none=True)
            inputs.grad = None
            with torch.profiler.record_function("analysis.forward_loss") if annotate else nullcontext():
                loss = call(inputs).square().mean()
            with torch.profiler.record_function("analysis.backward") if annotate else nullcontext():
                loss.backward()

        session = SimpleNamespace(model=model, device=torch.device("cpu"),
                                  runtime={"execution": execution}, reset=lambda: None, step=step)
        with tempfile.TemporaryDirectory() as directory:
            result = measurement.profile(session, Path(directory), 2, detail=detail)
            self.assertTrue(Path(result["trace"]).is_file())
        self.assertEqual(result["module_annotations_enabled"], execution == "eager")
        torch.testing.assert_close(inputs.grad, expected_inputs.grad)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        for module in model.modules():
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._forward_hooks)
        ranges = {row["name"] for row in result["operator_ranges"]}
        self.assertIn("analysis.forward_loss", ranges)
        self.assertIn("analysis.backward", ranges)
        return ranges

    def test_profile_preserves_fullgraph_execution_and_gradients(self):
        for in_place in (False, True):
            for detail in ("trace", "stacks"):
                with self.subTest(in_place=in_place, detail=detail):
                    ranges = self.profile_model(execution="compile", in_place=in_place, detail=detail)
                    self.assertFalse(any(name.startswith("module::") for name in ranges))

    def test_eager_profile_retains_module_annotations(self):
        ranges = self.profile_model()
        self.assertIn("module::<root>::Sequential", ranges)
        self.assertIn("module::0::Linear", ranges)

    def test_aliases_count_once_and_release_after_last_saved_owner(self):
        value = torch.randn(5, 7)
        weight = torch.randn(7, 3)
        tracker = measurement.SavedStorageLifetimes((weight,))
        first, second, parameter = tracker.pack(value), tracker.pack(value[:, 1:]), tracker.pack(weight)
        self.assertEqual(tracker.snapshot()["live_nonpersistent_bytes"], value.untyped_storage().nbytes())
        tracker.unpack(first)
        tracker.unpack(first)
        del first
        gc.collect()
        self.assertGreater(tracker.snapshot()["live_nonpersistent_bytes"], 0)
        del second
        gc.collect()
        self.assertEqual(tracker.snapshot()["live_nonpersistent_bytes"], 0)
        self.assertGreater(tracker.snapshot()["live_unique_bytes"], 0)
        del parameter
        gc.collect()
        self.assertEqual(tracker.snapshot()["live_unique_bytes"], 0)

    def test_retained_graph_does_not_release_storage_on_first_unpack(self):
        x = torch.randn(13, requires_grad=True)
        tracker = measurement.SavedStorageLifetimes()
        with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
            loss = x.square().sum()
        loss.backward(retain_graph=True)
        self.assertGreater(tracker.snapshot()["live_unique_bytes"], 0)
        loss.backward()
        gc.collect()
        self.assertEqual(tracker.snapshot()["live_unique_bytes"], 0)

    def test_memory_diagnostic_integrates_owners_and_dispatch_without_retaining_tensors(self):
        model = torch.nn.Linear(4, 3)
        inputs = torch.randn(2, 4, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters())

        def clear():
            optimizer.zero_grad(set_to_none=True)
            inputs.grad = None

        def forward():
            output = model(inputs)
            return output, output.square().mean()

        def step():
            clear()
            _, loss = forward()
            loss.backward()
            optimizer.step()

        session = SimpleNamespace(model=model, optimizer=optimizer, inputs=inputs, kwargs={},
                                  device=torch.device("cpu"), runtime={"accumulation_steps": 1},
                                  reset=clear, clear=clear, step=step, forward=forward,
                                  backward=lambda loss: loss.backward(), after_step=None)
        result = measurement.memory_diagnostics(session)
        report = result["invocation_storage_ledger"]
        self.assertIsNone(report["opaque_workspace_bytes"])
        self.assertGreater(report["peak_visible_bytes_by_device"]["cpu"], 0)
        owners = {owner for row in report["storages"] for owner in row["owners"]}
        self.assertTrue({"model_parameter", "caller_input", "optimizer_state", "escaped_output",
                         "leaf_gradient", "saved_for_backward"}.issubset(owners))
        self.assertEqual(result["observed_saved_unique_storage_bytes"],
                         result["saved_storage_lifetimes"]["cumulative_unique_storage_bytes"])
        self.assertIsNone(result["peak_allocated_bytes"])
        import json
        json.dumps(result)

    def test_linked_launches_are_attributed_without_double_counting(self):
        kernel = SimpleNamespace(name="_bf16_gemm")
        child = SimpleNamespace(name="launch", kernels=[kernel], cpu_children=[])
        parent = SimpleNamespace(name="deepopt.compiler.fc1.forward.linear.forward", kernels=[kernel], cpu_children=[child])
        evidence = measurement.compiler_launch_evidence([parent, child], 1)
        self.assertEqual(evidence[parent.name]["linked_cuda_launches_per_step"], 1)
        self.assertEqual(evidence[parent.name]["kernel_names"], ["_bf16_gemm"])

    def test_selected_fusion_ranges_retain_cuda_correlation(self):
        kernel = SimpleNamespace(name="_bf16_gemm")
        launch = SimpleNamespace(name="launch", kernels=[kernel], cpu_children=[], correlation_id=29)
        candidate = SimpleNamespace(name="deepopt.fusion::candidate-id", id=17, kernels=[], cpu_children=[launch])
        evidence = measurement.compiler_launch_evidence([candidate, launch], 1)[candidate.name]
        self.assertEqual(evidence["linked_cuda_launches_per_step"], 1)
        self.assertEqual(evidence["range_event_ids"], [17])
        self.assertEqual(evidence["correlation_ids"], [29])


if __name__ == "__main__":
    unittest.main()
