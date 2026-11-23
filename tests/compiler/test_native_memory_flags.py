"""Default-off memory options preserve CUDA-metadata scheduling ownership.

These tests run the real proof/transform/scheduler on static CUDA metadata.
They never launch CUDA and do not establish target-device qualification.
"""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from deepopt._bf16_policy import BF16Policy
from deepopt.compiler import BackendConfig
from deepopt.compiler.autodiff import donation
from deepopt.compiler.autodiff.optimization import backward_layouts
from deepopt.compiler.autodiff.semantic import recorded
from deepopt.compiler.autodiff.spatial import Convolution
from deepopt.compiler.ir import CallNode, InputNode, IRGraph, OutputNode, PointwiseNode, TensorMeta, TensorValue
from deepopt.compiler.registry import KernelCodegenContext
from deepopt.compiler.scheduler import FusionGroup, Scheduler, SinglePointwiseOp
from deepopt.kernel_policy import KernelPolicy


def backward_fixture(enabled):
    context = KernelCodegenContext(kernel_policy=KernelPolicy("triton"), donate_fresh_backward_add=enabled)
    meta = TensorMeta((2, 4, 3, 7), torch.bfloat16, "cuda:0", (84, 21, 7, 1), True)
    weight_meta = TensorMeta((4, 4, 1, 1), torch.float32, "cuda:0", (4, 1, 1, 1), True)
    graph = IRGraph()
    gradient = TensorValue("seed", replace(meta, requires_grad=False, stride=None), provenance="upstream_gradient")
    weight = TensorValue("weight", weight_meta, provenance="parameter")
    graph.add(InputNode(0, gradient))
    graph.add(InputNode(1, weight))
    results = []
    specifications = []
    for index in range(2):
        module = nn.Conv2d(4, 4, 1, bias=False)
        module.plan = {"activation_compression": dict(enabled=True, bits=2, group_size=1024)}
        source = TensorValue(f"forward_input{index}", meta)
        public = TensorValue(f"forward_output{index}", meta)
        raw = SimpleNamespace(module=module, module_path=f"mlp.projection{index}", node_name=f"projection{index}",
            op="deepopt_conv", args=(source, weight, None), values=(public,), options={}, kwargs={},
            plan=module.plan, bf16_policy=BF16Policy("bf16_storage"))
        spec = Convolution(raw, context)
        result = TensorValue(f"dx{index}", replace(meta, requires_grad=False))
        result.fresh_allocation = spec.fresh_dx_contract()
        if result.fresh_allocation is not None:
            result.meta = result.fresh_allocation.meta
        graph.add(CallNode("convolution.native_dx_db", recorded(spec, spec.dx_db, "backward"),
                           (gradient, weight), {}, (result,)))
        results.append(result)
        specifications.append(spec)
    summed = TensorValue("sum", replace(meta, requires_grad=False))
    addition = PointwiseNode("add", tuple(results), {}, summed, torch.add)
    graph.add(addition)
    graph.add(OutputNode(summed))
    return graph, context, specifications, results, addition


class NativeMemoryFlagTests(unittest.TestCase):
    def test_defaults_and_validation_are_explicit(self):
        config = BackendConfig()
        self.assertFalse(config.fuse_norm_cat_body)
        self.assertFalse(config.donate_fresh_backward_add)
        for name in ("fuse_norm_cat_body", "donate_fresh_backward_add"):
            self.assertFalse(getattr(KernelCodegenContext(), name))
            with self.assertRaisesRegex(ValueError, "boolean"):
                BackendConfig(**{name: 1})
            with self.assertRaisesRegex(ValueError, "native AutoDiff"):
                BackendConfig(use_native_autodiff=False, **{name: True})

    def test_default_off_produces_no_fresh_proof_or_generated_cuda_add_partition(self):
        graph, context, specs, results, addition = backward_fixture(False)
        self.assertTrue(all(spec.fresh_dx_contract() is None for spec in specs))
        self.assertTrue(all(result.fresh_allocation is None for result in results))
        diagnostics = []
        self.assertEqual(donation.backward(graph, context, diagnostics=diagnostics), ())
        backward_layouts(graph)
        self.assertTrue(all(result.meta.stride is None for result in results))
        schedule = Scheduler(graph, kernel_policy=context.kernel_policy, backward=True).run()
        self.assertIn(addition, schedule.items)
        self.assertFalse(any(isinstance(item, (FusionGroup, SinglePointwiseOp)) for item in schedule.items))
        self.assertFalse(any(row["selected"] for row in diagnostics))

    def test_explicit_enable_proves_and_selects_owned_alias_without_global_layout_relaxation(self):
        graph, context, _, results, addition = backward_fixture(True)
        self.assertTrue(all(result.fresh_allocation is not None for result in results))
        diagnostics = []
        self.assertEqual(donation.backward(graph, context, diagnostics=diagnostics), ("fresh_backward_add_donation",))
        backward_layouts(graph)
        replacement = next(node for node in graph.nodes if getattr(node, "op", None) == "memory.donated_add_out")
        self.assertNotIn(addition, graph.nodes)
        self.assertEqual(replacement.values[0].meta.stride, (84, 21, 7, 1))
        self.assertIsNone(replacement.values[0].fresh_allocation)
        self.assertEqual(replacement.scheduling_constraints["mutation"], "invocation_owned_add_out")
        self.assertFalse(diagnostics[0]["kernel_fusion"])
        schedule = Scheduler(graph, kernel_policy=context.kernel_policy, backward=True).run()
        self.assertIn(replacement, schedule.items)
        self.assertFalse(any(isinstance(item, (FusionGroup, SinglePointwiseOp)) for item in schedule.items))


if __name__ == "__main__":
    unittest.main()
