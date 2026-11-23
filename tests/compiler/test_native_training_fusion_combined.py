"""Native normalization/SiLU and SwiGLU composites compose through convolutions."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt._bf16_policy import BF16Policy
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import norm_activation_residuals, swiglu_fusion
from deepopt.compiler.autodiff.optimization import rebuild
from deepopt.compiler.trace.metadata_utils import contiguous_stride
from deepopt.opt.nn.activation.elementwise_modules import adapter
from deepopt.opt.nn.normalization.modules import DOGroupNorm


class _TrainingBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        plan = dict(fp8=False, bf16_policy=BF16Policy(policy).as_dict(),
            activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True))
        self.norm = DOGroupNorm(nn.GroupNorm(8, 64), plan=plan)
        self.activation, self.gate_activation = adapter("silu", plan=plan), adapter("silu", plan=plan)
        self.gate = nn.Conv2d(64, 64, 1, bias=False)
        self.up = nn.Conv2d(64, 64, 1, bias=False)

    def forward(self, images):
        x = self.activation(self.norm(images))
        gate = self.gate_activation(self.gate(x))
        return gate * self.up(x)


class _PreparedTracer(torch.fx.Tracer):
    def is_leaf_module(self, module, name):
        return hasattr(module, "plan") or super().is_leaf_module(module, name)


def _compare(test, device):
    torch.manual_seed(1721)
    source = _TrainingBlock("bf16_storage" if device == "cuda" else "torch_amp").to(device)
    images = torch.randn(2, 64, 32, 32, dtype=torch.bfloat16, device=device)
    results, coverage = [], []
    for enabled in (False, True):
        current = deepcopy(source)
        model, report = torch.fx.GraphModule(current, _PreparedTracer().trace(current)), CompilationReport()
        with autocast(device, dtype=torch.bfloat16, fp8=False,
                bf16_policy="bf16_storage" if device == "cuda" else "torch_amp",
                activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)):
            ShapeProp(model).propagate(images)
            executable = compile_fx(model, [images], config=BackendConfig(require_full_aot=True,
                share_compatible_packs=True, fuse_norm_activation_residuals=enabled,
                norm_activation_fusion_stage="forward_backward" if enabled else "off",
                fuse_swiglu_residuals=enabled, swiglu_fusion_stage="forward_backward" if enabled else "off"),
                report=report)
            output = executable(images)
        # A scalar-loss seed is expanded, so the fused adjoints must read
        # logical indices instead of assuming a contiguous cotangent.
        seed = torch.tensor(.173, dtype=output.dtype, device=device).expand_as(output)
        gradients = torch.autograd.grad(output, tuple(model.parameters()), seed, retain_graph=True)
        repeated = torch.autograd.grad(output, tuple(model.parameters()), seed)
        for actual, expected in zip(repeated, gradients):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        test.assertFalse(images.requires_grad)
        test.assertIsNone(images.grad)
        test.assertEqual(len(gradients), len(tuple(source.parameters())))
        diagnostic = report.as_dict()
        test.assertEqual(report.runtime_fallbacks, 0)
        test.assertFalse(diagnostic["legacy_backward_operations"])
        selected = {row["pattern"] for row in diagnostic["fusion_candidates"] if row["selected"]}
        for pattern in (norm_activation_residuals.FORWARD, norm_activation_residuals.BACKWARD,
                        swiglu_fusion.FORWARD, swiglu_fusion.BACKWARD):
            test.assertEqual(pattern in selected, enabled and device == "cuda", pattern)
        coverage.append(diagnostic["native_autodiff_operations"])
        results.append((output.detach(), *gradients))
    test.assertGreater(coverage[0], 0)
    test.assertEqual(*coverage)
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class NativeTrainingFusionCombinedTests(unittest.TestCase):
    def test_cpu_device_ineligible_fusions_preserve_native_parameter_gradients(self):
        _compare(self, "cpu")

    def test_actual_CPU_graph_admits_both_passes_under_static_CUDA_metadata_only(self):
        """Selection regression only; the separate CUDA class executes kernels."""
        forwards, backwards = [], []
        original_forward, original_backward = norm_activation_residuals.forward, swiglu_fusion.backward

        def forward(graph, context):
            forwards.append((graph, context))
            return original_forward(graph, context)

        def backward(graph, context, diagnostics=None):
            backwards.append(graph)
            return original_backward(graph, context, diagnostics)

        with patch.object(norm_activation_residuals, "forward", side_effect=forward), \
                patch.object(swiglu_fusion, "backward", side_effect=backward):
            _compare(self, "cpu")
        graph, context = forwards[-1]
        adjoint = backwards[-1]
        graph.nodes[-1].result = graph.nodes[-1].result[0]
        rebuild(graph)
        for current in (graph, adjoint):
            for node in current.nodes:
                for value in node.outputs():
                    meta = value.meta
                    if not (meta.dtype == torch.int64 and meta.shape == ()):
                        value.meta = replace(meta, device=torch.device("cuda:0"),
                            stride=contiguous_stride(meta.shape) if meta.dtype == torch.bfloat16 else meta.stride)
        for node in graph.nodes:
            spec = getattr(node, "specification", None)
            if type(spec) is norm_activation_residuals.Normalization:
                spec.metas = tuple(replace(meta, device=torch.device("cuda:0")) if meta is not None else None
                                   for meta in spec.metas)
                spec.backend = "triton"
            elif type(spec) is norm_activation_residuals.NativeActivation:
                spec.input_meta = replace(spec.input_meta, device=torch.device("cuda:0"))
                spec.kernel_routes = {key: "triton" for key in spec.kernel_routes}
        records, norm_forward = norm_activation_residuals.forward(graph, context)
        rows, swiglu_forward = swiglu_fusion.forward(graph, context)
        records += rows
        norm_backward = norm_activation_residuals.backward(adjoint, context, records)
        swiglu_backward = swiglu_fusion.backward(adjoint, context, records)
        self.assertEqual(set((*norm_forward, *swiglu_forward, *norm_backward, *swiglu_backward)),
            {norm_activation_residuals.FORWARD, norm_activation_residuals.BACKWARD,
             swiglu_fusion.FORWARD, swiglu_fusion.BACKWARD})


@unittest.skipUnless(torch.cuda.is_available(), "requires actual combined CUDA native training fusion execution")
class NativeTrainingFusionCombinedCudaTests(unittest.TestCase):
    def test_actual_compiled_composites_preserve_retained_parameter_backward(self):
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
            _compare(self, "cuda")


if __name__ == "__main__":
    unittest.main()
