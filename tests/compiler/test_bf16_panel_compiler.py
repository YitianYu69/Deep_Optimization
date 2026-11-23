"""Panel decode option binding and truthful CPU semantic-reference coverage."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt._bf16_policy import BF16Policy
from deepopt.compiler import BackendConfig, compile_fx
from deepopt.compiler.autodiff.semantic import BF16Linear, recorded
from deepopt.compiler.ir import TensorMeta
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.backends._shared.bf16_resources import bf16_scratch_requirements
from deepopt.opt.quantization.activation import quantize_pack_activation


COMPRESSION = {"enabled": True, "bits": 2, "group_size": 1024}


class BF16PanelCompilerTests(unittest.TestCase):
    def test_panel_requires_generated_triton_dw_route(self):
        with self.assertRaisesRegex(ValueError, "requires bf16_gemm_backend=generated"):
            BackendConfig(bf16_decode="panel")
        with self.assertRaisesRegex(ValueError, "Triton linear.dw"):
            BackendConfig(bf16_decode="panel", bf16_gemm_backend="generated", kernel_backend="tilelang")
        valid = BackendConfig(bf16_decode="panel", bf16_gemm_backend="generated")
        self.assertEqual(valid.bf16_decode, "panel")

    def test_cpu_compiled_reference_matches_materialized_recipe_and_reports_scratch(self):
        torch.manual_seed(803)
        original = nn.Sequential(nn.Linear(37, 19)).bfloat16()
        control = deepcopy(original)
        values = torch.randn(33, 37, dtype=torch.bfloat16, requires_grad=True)
        reference_values = values.detach().clone().requires_grad_()
        graph = torch.fx.symbolic_trace(original)
        reference = torch.fx.symbolic_trace(control)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                      activation_compression=COMPRESSION, config={}):
            ShapeProp(graph).propagate(values)
            ShapeProp(reference).propagate(reference_values)
            compiled = compile_fx(graph, [values], config=BackendConfig(
                bf16_gemm_backend="generated", bf16_decode="panel", require_full_aot=True))
            baseline = compile_fx(reference, [reference_values], config=BackendConfig(require_full_aot=True))
            actual, expected = compiled(values), baseline(reference_values)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        # Both CPU references execute the same materialized recipe. This does
        # not execute or qualify a generated panel kernel.
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(values.grad, reference_values.grad)
        for parameter, control_parameter in zip(graph.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, control_parameter.grad)
        program, = compiled.backend_artifact.deepopt_programs
        self.assertFalse(program.generated_kernels)
        self.assertEqual(program.scratch_requirements[0]["name"], "decoded_operand")
        self.assertEqual(program.scratch_requirements[0]["bytes"], 33 * 37 * 2)
        self.assertEqual(compiled.compilation_report.program_schedules[0]["decode"], "panel")

    def test_bound_dw_dispatches_panel_without_materializing_before_backend(self):
        from deepopt.opt.backends.bf16 import product as actual_product

        source = torch.randn(33, 37, dtype=torch.bfloat16)
        packed = quantize_pack_activation(source, 1024, bits=2)
        gradient = torch.randn(33, 19, dtype=torch.bfloat16)
        spec = object.__new__(BF16Linear)
        spec.bf16_policy = BF16Policy()
        spec.need_saved_input, spec.compressed, spec.decode = True, True, "panel"
        spec.plan = {"activation_compression": COMPRESSION}
        spec.matrix_shape, spec.weight_meta = source.shape, TensorMeta((19, 37), torch.bfloat16, "cpu")
        spec.tile, spec.num_stages = 64, 3
        spec.backend = lambda role: "triton"
        with patch("deepopt.opt.backends.bf16.product", wraps=actual_product) as observed:
            result = spec.dw(gradient, packed.data, packed.step, packed.minimum)
        call = observed.call_args
        self.assertIsNone(call.args[1])
        self.assertIsNotNone(call.kwargs["packed_b"])
        self.assertEqual(call.kwargs["packed_route"], "panel")
        self.assertEqual(call.kwargs["panel_size"], 1024)
        self.assertEqual(tuple(result.shape), (19, 37))

    def test_scratch_annotation_belongs_to_backward_call(self):
        class Spec:
            module_path, node_name = "projection", "projection"
            precision, family, profile_ranges = "bf16", "linear", False
            kernel_policy = KernelPolicy()
            scratch_requirements = bf16_scratch_requirements(19, 37, 2051, route="panel")

            def dw(self, value):
                return value

            def forward(self, value):
                return value

        spec = Spec()
        backward = recorded(spec, spec.dw, "backward")
        forward = recorded(spec, spec.forward, "forward")
        self.assertEqual(backward.scratch_requirements, spec.scratch_requirements)
        self.assertFalse(hasattr(forward, "scratch_requirements"))
        self.assertEqual(sum(row["bytes"] for row in backward.scratch_requirements),
                         1024 * 37 * 2 + 19 * 37 * 4)


if __name__ == "__main__":
    unittest.main()
