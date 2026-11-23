"""Fresh backward add writes keep saved values, seeds and native VJPs intact."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import donation
from deepopt.compiler.autodiff.optimization import rebuild
from deepopt.compiler.ir import CallNode, InputNode, PointwiseNode, TensorMeta, TensorValue
from deepopt.compiler.trace.metadata_utils import contiguous_stride


class _Projections(nn.Module):
    def __init__(self, channels=4, *, swiglu=False, kernel=1, padding=0, groups=1):
        super().__init__()
        self.gate_proj = nn.Conv2d(channels, channels, kernel, padding=padding, groups=groups, bias=False)
        self.up_proj = nn.Conv2d(channels, channels, kernel, padding=padding, groups=groups, bias=False)
        self.swiglu = swiglu

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        return nn.functional.silu(gate) * up if self.swiglu else gate + up


def _compile(source, sample, enabled, *, modifier=None, compressed=False):
    graph = torch.fx.symbolic_trace(deepcopy(source))
    report = CompilationReport()
    captured = []
    original = donation.backward

    def select(backward_graph, context, **options):
        if modifier is not None:
            unsafe = modifier(backward_graph)
            if unsafe is not None:
                options["unsafe_values"] = unsafe
            rebuild(backward_graph)
        result = original(backward_graph, context, **options)
        captured.append(backward_graph)
        return result

    scope = (autocast(sample.device.type, dtype=torch.bfloat16, fp8=False,
                     activation_compression=dict(enabled=compressed, bits=2, group_size=1024),
                     bf16_policy="bf16_storage" if sample.device.type == "cuda" else "torch_amp")
             if sample.dtype == torch.bfloat16 else nullcontext())
    with patch.object(donation, "backward", new=select), scope:
        ShapeProp(graph).propagate(sample)
        executable = compile_fx(graph, [sample], config=BackendConfig(
            require_full_aot=True, donate_fresh_backward_add=enabled), report=report)
        output = executable(sample)
    return graph, output, report, captured


def _decisions(report):
    return [value for value in report.as_dict()["fusion_candidates"]
            if value["pattern"] == "fresh_backward_add_donation"]


def _merge(graph):
    return next(node for node in graph.nodes if isinstance(node, PointwiseNode) and node.op == "add"
                and any(getattr(value, "fresh_allocation", None) is not None for value in node.args))


class BackwardAddDonationTests(unittest.TestCase):
    def test_actual_native_vjp_reuses_storage_and_preserves_original_operand_order(self):
        torch.manual_seed(322)
        source = _Projections(swiglu=True)
        sample = torch.randn(2, 4, 3, 7)
        for dtype in (torch.float32, torch.bfloat16):
            results, observations = [], []
            for enabled in (False, True):
                x = sample.to(dtype).clone().requires_grad_()
                original = donation._add_out

                def observed(left, right, **options):
                    donor = (left, right)[options["donor_index"]]
                    expected = torch.add(left, right)
                    result = original(left, right, **options)
                    observations.append((result is donor, result.data_ptr() == donor.data_ptr(),
                                         options["donor_index"]))
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    return result

                with patch.object(donation, "_add_out", new=observed):
                    model, output, report, _ = _compile(source, x, enabled, compressed=dtype == torch.bfloat16)
                    seed = torch.linspace(-1.3, 0.9, output.numel(), dtype=output.dtype).reshape(output.shape)
                    seed_before = seed.clone()
                    gradients = torch.autograd.grad(output, (x, *model.parameters()), seed)
                    self.assertTrue(all(value.dtype == torch.float32 for value in gradients[1:]))
                    torch.testing.assert_close(seed, seed_before, rtol=0, atol=0)
                results.append((output.detach(), gradients, report))
            for actual, expected in zip((results[1][0], *results[1][1]), (results[0][0], *results[0][1])):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(observations)
            self.assertTrue(all(private and same for private, same, _ in observations))
            selected = [row for row in _decisions(results[1][2]) if row["selected"]]
            self.assertEqual(len(selected), 1)
            record = selected[0]
            self.assertFalse(record["kernel_fusion"])
            self.assertEqual(record["replacement_contract"]["effects"], "invocation_owned_add_out")
            self.assertEqual(record["estimated_removed_output_allocation_bytes"], sample.numel() * dtype.itemsize)
            self.assertEqual(results[1][2].runtime_fallbacks, 0)
            self.assertEqual(results[1][2].as_dict()["native_autodiff_regions"], 1)
            self.assertFalse(results[1][2].as_dict()["legacy_backward_operations"])
            json.dumps(record, allow_nan=False)

    def test_default_off_does_not_call_mutating_executor(self):
        source = _Projections()
        x = torch.randn(2, 4, 3, 7, requires_grad=True)
        with patch.object(donation, "_add_out", side_effect=AssertionError("default route mutated")):
            model, output, report, graphs = _compile(source, x, False)
            torch.autograd.grad(output, (x, *model.parameters()), torch.randn_like(output))
        self.assertTrue(_decisions(report))
        self.assertTrue(all(not row["selected"] for row in _decisions(report)))
        self.assertFalse(any(node.op == "memory.donated_add_out" for node in graphs[0].nodes
                             if hasattr(node, "op")))

    def test_right_operand_can_be_donated_without_swapping_inputs(self):
        def only_right(graph):
            _merge(graph).args[0].fresh_allocation = None

        observations = []
        original = donation._add_out

        def observed(left, right, **options):
            expected = torch.add(left, right)
            result = original(left, right, **options)
            observations.append((options["donor_index"], result is right))
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            return result

        x = torch.randn(2, 4, 3, 7, requires_grad=True)
        with patch.object(donation, "_add_out", new=observed):
            model, output, report, _ = _compile(_Projections(), x, True, modifier=only_right)
            torch.autograd.grad(output, (x, *model.parameters()), torch.randn_like(output))
        self.assertEqual(observations, [(1, True)])
        self.assertEqual(next(row for row in _decisions(report) if row["selected"])["donor_index"], 1)

    def test_retained_graph_repeated_backward_owns_fresh_destinations_each_time(self):
        x = torch.randn(2, 4, 3, 7, requires_grad=True)
        source = _Projections(swiglu=True)
        model, output, report, _ = _compile(source, x, True)
        seed = torch.randn_like(output)
        seed_before = seed.clone()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        first = torch.autograd.grad(output, (x, *model.parameters()), seed, retain_graph=True)
        snapshots = tuple(value.clone() for value in first)
        second = torch.autograd.grad(output, (x, *model.parameters()), seed, retain_graph=True)
        third = torch.autograd.grad(output, (x, *model.parameters()), seed)
        for values in (first, second, third):
            for actual, expected in zip(values, snapshots):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(seed, seed_before, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(any(row["selected"] for row in _decisions(report)))

    def test_fanout_alias_and_retained_value_reject_before_any_write(self):
        def change(kind):
            def apply(graph):
                node = _merge(graph)
                donor = node.args[0]
                node.args[1].fresh_allocation = None
                if kind == "alias":
                    alias = TensorValue("private_alias", donor.meta)
                    position = graph.nodes.index(node)
                    graph.nodes.insert(position, CallNode("test.alias_view", lambda value: value.view_as(value),
                                                          (donor,), {}, (alias,)))
                elif kind == "retained":
                    node.saved_tensors_for_bwd = (donor,)
                elif kind == "model":
                    donor.storage_source = "model.weight"
                elif kind == "unsafe":
                    donor.model_state_origin = ("model.weight", donor.meta)
                elif kind == "stale":
                    donor.fresh_allocation = replace(donor.fresh_allocation,
                        meta=replace(donor.meta, dtype=torch.bfloat16))
            return apply

        for kind, reason in (("alias", "donor_fanout_or_escape"), ("retained", "retained_backward_value"),
                             ("model", "protected_storage"), ("unsafe", "protected_storage"),
                             ("stale", "stale_fresh_allocation_metadata")):
            with self.subTest(kind=kind):
                x = torch.randn(2, 4, 3, 7, requires_grad=True)
                with patch.object(donation, "_add_out", side_effect=AssertionError("rejected route mutated")):
                    model, output, report, _ = _compile(_Projections(), x, True, modifier=change(kind))
                    torch.autograd.grad(output, (x, *model.parameters()), torch.randn_like(output))
                rows = _decisions(report)
                self.assertFalse(any(row["selected"] for row in rows))
                self.assertIn(reason, rows[0]["operand_rejections"])

    def test_supplied_seed_never_acquires_ownership_from_copied_proof(self):
        def replace_donor(graph):
            node = _merge(graph)
            seed = next(item.value for item in graph.nodes if isinstance(item, InputNode) and item.index == 0)
            seed.meta = node.args[0].meta
            seed.fresh_allocation = node.args[0].fresh_allocation
            node.args[1].fresh_allocation = None
            node.args = (seed, node.args[1])

        x = torch.randn(2, 4, 3, 7, requires_grad=True)
        _, _, report, _ = _compile(_Projections(), x, True, modifier=replace_donor)
        row = _decisions(report)[0]
        self.assertFalse(row["selected"])
        self.assertIn("fresh_allocation_producer_mismatch", row["operand_rejections"])

    def test_unsafe_input_disables_the_replacement(self):
        def unsafe(graph):
            return frozenset((_merge(graph).args[0],))

        x = torch.randn(2, 4, 3, 7, requires_grad=True)
        with patch.object(donation, "_add_out", side_effect=AssertionError("unsafe route mutated")):
            model, output, report, _ = _compile(_Projections(), x, True, modifier=unsafe)
            torch.autograd.grad(output, (x, *model.parameters()), torch.randn_like(output))
        self.assertFalse(any(row["selected"] for row in _decisions(report)))
        self.assertEqual(_decisions(report)[0]["reason"], "unsafe_value")

    def test_cropped_grouped_and_channels_last_routes_keep_allocating_add(self):
        for kind in ("spatial", "grouped", "channels_last"):
            with self.subTest(kind=kind):
                source = _Projections(kernel=3, padding=1) if kind == "spatial" else _Projections(groups=2) if kind == "grouped" else _Projections()
                x = torch.randn(2, 4, 3, 7)
                if kind == "channels_last":
                    x = x.contiguous(memory_format=torch.channels_last)
                x.requires_grad_()
                with patch.object(donation, "_add_out", side_effect=AssertionError("unqualified geometry mutated")):
                    model, output, report, _ = _compile(source, x, True)
                    torch.autograd.grad(output, (x, *model.parameters()), torch.randn_like(output))
                self.assertFalse(any(row["selected"] for row in _decisions(report)))

    def test_runtime_checks_precede_mutation_for_alias_views_metadata_and_higher_order(self):
        shape = (2, 4, 3, 7)
        metadata = TensorMeta(shape, torch.float32, "cpu", contiguous_stride(shape))
        for kind in ("alias", "view", "metadata", "higher_order"):
            left, right = torch.randn(shape), torch.randn(shape)
            if kind == "alias":
                right = left.view_as(left)
            elif kind == "view":
                left = torch.randn((3, *shape[1:]))[:2]
            elif kind == "metadata":
                right = right.transpose(2, 3).contiguous()
            else:
                left.requires_grad_()
            before = left.detach().clone()
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                donation._add_out(left, right, donor_index=0, metadata=metadata)
            torch.testing.assert_close(left, before, rtol=0, atol=0)

    def test_donation_is_an_effect_barrier_with_explicit_result_alias(self):
        x = torch.randn(2, 4, 3, 7, requires_grad=True)
        _, _, report, graphs = _compile(_Projections(), x, True)
        from deepopt.compiler.scheduler import Scheduler

        graph = graphs[0]
        replacement = next(node for node in graph.nodes if getattr(node, "op", None) == "memory.donated_add_out")
        schedule = Scheduler(graph, backward=True).run()
        self.assertIn(id(replacement), schedule.effect_barriers)
        self.assertIn(replacement, schedule.items)
        self.assertEqual(replacement.scheduling_constraints["mutation"], "invocation_owned_add_out")
        self.assertEqual(replacement.scheduling_constraints["owned_output_alias"]["result"], replacement.values[0].name)
        self.assertIsNone(replacement.values[0].fresh_allocation)
        self.assertTrue(any(row["selected"] for row in _decisions(report)))


@unittest.skipUnless(torch.cuda.is_available(), "Backward donation CUDA execution requires a GPU")
class BackwardAddDonationCudaTests(unittest.TestCase):
    def test_native_cuda_projection_vjp_and_storage_identity(self):
        for dtype in (torch.float32, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            _cuda_projection_check(self, (2, 8, 7, 11), dtype)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA_DONATION") == "1",
                     "Original VAE projection geometry is an opt-in CUDA capacity test")
class BackwardAddDonationOriginalShapeCudaTests(unittest.TestCase):
    def test_original_encoder_one_mlp_projection_geometry(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 CUDA support is required")
        _cuda_projection_check(self, (128, 224, 224, 224), torch.bfloat16)


def _cuda_projection_check(test, shape, dtype):
    torch.manual_seed(183)
    source = _Projections(shape[1], swiglu=True).cuda()
    sample = torch.randn(shape, device="cuda", dtype=dtype)
    seed = torch.linspace(-0.7, 1.1, sample.numel(), device="cuda", dtype=dtype).reshape(shape)
    gradients, observations = [], []
    original = donation._add_out

    def observed(left, right, **options):
        donor = (left, right)[options["donor_index"]]
        result = original(left, right, **options)
        observations.append(result is donor and result.data_ptr() == donor.data_ptr())
        return result

    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
        for enabled in (False, True):
            x = sample.clone().requires_grad_()
            with patch.object(donation, "_add_out", new=observed):
                model, output, report, _ = _compile(source, x, enabled, compressed=True)
                gradients.append(tuple(value.detach() for value in torch.autograd.grad(
                    output, (x, *model.parameters()), seed, retain_graph=enabled)))
                if enabled:
                    again = torch.autograd.grad(output, (x, *model.parameters()), seed)
                    for actual, expected in zip(again, gradients[-1]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            test.assertEqual(report.runtime_fallbacks, 0)
            test.assertEqual(any(row["selected"] for row in _decisions(report)), enabled)
        torch.cuda.synchronize()
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    test.assertTrue(observations)
    test.assertTrue(all(observations))


if __name__ == "__main__":
    unittest.main()
