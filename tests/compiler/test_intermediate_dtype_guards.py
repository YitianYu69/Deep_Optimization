"""Runtime intermediate dtype receipts cannot be hidden by later casts."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt._bf16_policy import BF16Policy, bf16_policy_scope
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import AuxiliaryTensorMeta, CallNode, IRGraph, InputNode, OutputNode, TensorMeta, TensorValue
from deepopt.compiler.runtime.physical_memory import PhysicalPlanObservation
from deepopt.compiler.scheduler import Scheduler


class IntermediateDtypeGuardTests(unittest.TestCase):
    @staticmethod
    def artifact(*, storage, actual_dtype=torch.float32, declared_dtype=torch.bfloat16, device="cpu"):
        events = []
        meta = TensorMeta((4,), torch.bfloat16, device, (1,))
        x, result = TensorValue("x", meta), TensorValue("result", meta)
        hidden = TensorValue("hidden_norm", TensorMeta((4,), declared_dtype, device, (1,)))
        graph = IRGraph()
        graph.add(InputNode(0, x))
        producer = CallNode("norm", lambda value: value.to(actual_dtype), (x,), {}, (hidden,))
        producer.bf16_policy = BF16Policy("bf16_storage" if storage else "torch_amp")
        producer.module_path = "decoder.0"
        graph.add(producer)

        def downstream(value):
            events.append("consumer")
            return value.bfloat16()

        graph.add(CallNode("downstream_conv_cast", downstream, (hidden,), {}, (result,)))
        graph.add(OutputNode(result))
        return WrapperCodegen(nn.Module(), Scheduler(graph).run(), enable_autodiff=False).run(), events

    def test_storage_rejects_hidden_fp32_before_downstream_can_recast(self):
        artifact, events = self.artifact(storage=True)
        with bf16_policy_scope("torch_amp", "cpu"), self.assertRaisesRegex(TypeError, "hidden_norm.*torch.bfloat16"):
            artifact.fn(torch.ones(4).bfloat16())
        self.assertEqual(events, [])
        self.assertNotIn(".to(", artifact.source)

    def test_diagnostic_checks_amp_without_changing_ordinary_amp_wrapper(self):
        artifact, events = self.artifact(storage=False)
        result = artifact.fn(torch.ones(4).bfloat16())
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(events, ["consumer"])
        self.assertNotIn("violates captured dtype", artifact.source)
        events.clear()
        with self.assertRaisesRegex(TypeError, "hidden_norm"):
            PhysicalPlanObservation(artifact).run(artifact, torch.ones(4).bfloat16())
        self.assertEqual(events, [])

    def test_declared_fp32_island_is_valid_under_storage_policy(self):
        artifact, events = self.artifact(storage=True, declared_dtype=torch.float32)
        artifact.fn(torch.ones(4).bfloat16())
        self.assertEqual(events, ["consumer"])

    def test_failure_has_metadata_and_never_formats_or_transfers_tensor_data(self):
        artifact, _ = self.artifact(storage=True)
        value = torch.ones(4).bfloat16()
        with patch.object(torch.Tensor, "__repr__", side_effect=AssertionError("tensor formatting")), \
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device transfer")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar transfer")), \
                self.assertRaises(TypeError) as caught:
            artifact.fn(value)
        details = caught.exception.dtype_contract
        self.assertEqual(details["value"], "hidden_norm")
        self.assertEqual(details["operation"], "norm")
        self.assertEqual(details["module"], "decoder.0")
        self.assertEqual(details["stage"], "forward")
        self.assertEqual(details["expected"]["dtype"], "torch.bfloat16")
        self.assertEqual(details["actual"], {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [4]})
        self.assertIn("actual=", str(caught.exception))
        self.assertEqual(details["bf16_policy"], {"name": "bf16_storage", "version": 1})

    def test_real_resize_module_origin_survives_native_lowering_and_dtype_failure(self):
        from deepopt import autocast
        from deepopt.compiler import BackendConfig, compile_fx
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.nn.primitive_state import Nearest2dGeometry
        from torch.fx.passes.shape_prop import ShapeProp

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

            def forward(self, value):
                return self.decoder(value)

        self.addCleanup(torch.compiler.reset)
        original = Nearest2dGeometry.forward
        def broken_resize(specification, value):
            return original(specification, value).float()

        for dynamo in (False, True):
            with self.subTest(dynamo=dynamo):
                torch.compiler.reset()
                model = Model()
                value = torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True)
                with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                              activation_compression=False, config={}):
                    if dynamo:
                        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                            options={"require_full_aot": True}), config={})
                        compiled = model
                    else:
                        graph = torch.fx.symbolic_trace(model)
                        ShapeProp(graph).propagate(value)
                        with patch.object(Nearest2dGeometry, "forward", broken_resize):
                            compiled = compile_fx(graph, [value], config=BackendConfig(require_full_aot=True))
                    with patch.object(Nearest2dGeometry, "forward", broken_resize), self.assertRaises(TypeError) as caught:
                        compiled(value)
                self.assertEqual(caught.exception.dtype_contract["module"], "decoder.0")
                self.assertEqual(caught.exception.dtype_contract["operation"], "autodiff_nearest2d")

    def test_non_tensor_failure_does_not_call_arbitrary_repr(self):
        from deepopt.compiler.codegen.wrapper import intermediate_dtype_error

        class Unprintable:
            def __repr__(self):
                raise AssertionError("result payload was formatted")

        contract = dict(value="resize", operation="nearest2d", module="decoder.0", stage="backward",
                        expected={"dtype": "torch.bfloat16"}, bf16_policy={}, primitive_recipe=3)
        error = intermediate_dtype_error(Unprintable(), contract)
        self.assertIsNone(error.dtype_contract["actual"]["dtype"])
        self.assertIn("Unprintable", error.dtype_contract["actual"]["type"])

    def test_native_backward_dtype_failure_keeps_forward_module_origin(self):
        from deepopt import autocast
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.nn.primitive_state import Nearest2dGeometry

        model = nn.Sequential()
        model.add_module("decoder", nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest")))
        value = torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True)
        original = Nearest2dGeometry.backward

        def broken_backward(specification, gradient):
            return original(specification, gradient).float()

        self.addCleanup(torch.compiler.reset)
        with patch.object(Nearest2dGeometry, "backward", broken_backward):
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                          activation_compression=False, config={}):
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    options={"require_full_aot": True}), config={})
                output = model(value)
            with self.assertRaises(TypeError) as caught:
                output.backward(torch.ones_like(output))
        contract = caught.exception.dtype_contract
        self.assertEqual(contract["module"], "decoder.0")
        self.assertEqual(contract["stage"], "backward")
        self.assertIsNotNone(contract["source_node"])
        self.assertEqual(contract["actual"]["dtype"], "torch.float32")

    def test_multiple_results_keep_fp32_statistics_and_optional_absence(self):
        graph = IRGraph()
        meta = TensorMeta((4,), torch.bfloat16, "cpu", (1,))
        x, y = TensorValue("x", meta), TensorValue("normalized", meta)
        stat = TensorValue("statistic", TensorMeta((), torch.float32, "cpu", ()))
        absent = TensorValue("absent", AuxiliaryTensorMeta(dtype=torch.int64, presence="optional"))
        graph.add(InputNode(0, x))
        node = CallNode("multi_norm", lambda value: (value, torch.tensor(1.), None),
                        (x,), {}, (y, stat, absent))
        node.bf16_policy = BF16Policy("bf16_storage")
        graph.add(node)
        graph.add(OutputNode((y, stat, absent)))
        artifact = WrapperCodegen(nn.Module(), Scheduler(graph).run(), enable_autodiff=False).run()
        output, statistic, optional = artifact.fn(torch.ones(4).bfloat16())
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(statistic.dtype, torch.float32)
        self.assertIsNone(optional)


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA tensors")
class IntermediateDtypeGuardCUDATests(unittest.TestCase):
    def test_cuda_fp32_intermediate_is_rejected_before_recast(self):
        artifact, events = IntermediateDtypeGuardTests.artifact(storage=True, device="cuda")
        with self.assertRaisesRegex(TypeError, "hidden_norm.*torch.bfloat16"):
            artifact.fn(torch.ones(4, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
        self.assertEqual(events, [])
