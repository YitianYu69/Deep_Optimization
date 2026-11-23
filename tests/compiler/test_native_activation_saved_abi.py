"""Native activation saves conform to their typed matrix ABI on every backend."""

from dataclasses import replace
from math import prod
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt._bf16_policy import resolve_bf16_policy
from deepopt.compiler.autodiff import semantic
from deepopt.compiler.autodiff.aot_auto_diff import AOTAutoDiff
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import InputNode, IRGraph, TensorMeta, TensorValue, register_uses
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.compiler.scheduler import Scheduler
from deepopt.compiler.trace.builder import GraphBuilder
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.backends.reference.activation import elementwise as reference_executor
from deepopt.opt.nn.activation.elementwise_modules import adapter


def activation_program(kind, source, *, compressed=False):
    """Build the same semantic forward and native-derived VJP used by capture."""
    policy = resolve_bf16_policy("bf16_storage")
    plan = {"fp8": False, "bf16_policy": policy.as_dict(),
            "activation_compression": {"enabled": compressed, "bits": 2, "group_size": 1024}}
    module = adapter(kind, plan=plan)
    graph, registry = IRGraph(), BackendRegistry()
    builder = GraphBuilder(graph)
    metadata = TensorMeta(tuple(source.shape), source.dtype, source.device, tuple(source.stride()), True)
    value = builder.input(0, metadata, name="activation_input").value
    output = TensorValue("activation_output", metadata)
    node = SimpleNamespace(module=module, plan=module.plan, args=(value, None, None), kwargs={},
        value=output, output=lambda: output, bf16_policy=policy, module_path="activation", node_name="activation",
        op="native_activation")
    specification = semantic.NativeActivation(node, KernelCodegenContext(kernel_policy=KernelPolicy()))
    graph.add(specification.lower(node))
    builder.output(output)
    semantic.register_rules(registry)
    trace = AOTAutoDiff(graph, registry=registry).diff()
    # JointRegion admits the caller's actual cotangent strides. Saved-state
    # input guards remain fully specified and are the contract under test.
    for input_node in trace.graph.nodes:
        if isinstance(input_node, InputNode) and input_node.index == 0:
            input_node.value.meta = replace(input_node.value.meta, stride=None)
    output_node = graph.nodes[-1]
    output_node.result = (output_node.result, trace.saved_tensors)
    register_uses(output_node)
    forward = WrapperCodegen(torch.nn.Module(), Scheduler(graph, registry=registry).run(),
        registry=registry, enable_autodiff=False).run().fn
    backward = WrapperCodegen(torch.nn.Module(), Scheduler(trace.graph, registry=registry).run(),
        registry=registry, enable_autodiff=False, diagnostic_phase="backward").run().fn
    return forward, backward, trace, specification, module


class NativeActivationSavedABITests(unittest.TestCase):
    def test_simulated_cuda_nd_recompute_snapshot_preserves_abi_storage_and_vjp(self):
        # CPU reference execution normally exports a matrix, which hid this
        # failure. Simulate only CUDA's N-D recompute-save shape; all forward
        # and backward arithmetic still uses the real reference executor.
        snapshots = []
        def forward_nd(source, geometry, **options):
            output, inverse, saved, snapshot, observations, info = reference_executor.forward(
                source, geometry, **options)
            if saved is not None and saved.recompute is not None:
                saved = replace(saved, data=saved.data.view(source.shape))
                snapshots.append({"pointer": saved.data.data_ptr(), "bytes": saved.data.untyped_storage().nbytes(),
                                  "shape": tuple(saved.data.shape)})
            return output, inverse, saved, snapshot, observations, info

        executor = SimpleNamespace(forward=forward_nd, backward=reference_executor.backward)
        torch.manual_seed(814)
        for kind in ("tanh", "sigmoid", "silu", "gelu"):
            shapes = ((1, 3, 5, 7), (3, 5, 7), (17,), ())
            if kind == "tanh":
                shapes += ((1, 3, 224, 224),)
            for shape in shapes:
                with self.subTest(kind=kind, shape=shape), torch.no_grad():
                    source = torch.randn(shape, dtype=torch.bfloat16)
                    forward, backward, trace, specification, _ = activation_program(kind, source)
                    gradient = torch.randn_like(source)
                    expected_output, expected_saved = forward(source)
                    expected_gradient, = backward(gradient, *expected_saved)
                    with patch("deepopt.opt.backends.activation.executor", return_value=executor):
                        actual_output, saved = forward(source)
                        initial_gradient, = backward(gradient, *saved)
                        self.assertEqual(tuple(saved[0].shape), trace.saved_tensors[0].meta.shape)
                        self.assertEqual(tuple(saved[0].stride()), trace.saved_tensors[0].meta.stride)
                        self.assertEqual(saved[0].data_ptr(), snapshots[-1]["pointer"])
                        self.assertEqual(saved[0].untyped_storage().nbytes(), snapshots[-1]["bytes"])
                        self.assertEqual(snapshots[-1]["shape"], shape)
                        self.assertEqual(tuple(saved[0].shape), specification.geometry.matrix_shape)
                        self.assertNotEqual(saved[0].data_ptr(), source.data_ptr())
                        source.add_(7)  # The exported view must still name the private forward snapshot.
                        actual_gradient, = backward(gradient, *saved)
                        repeated_gradient, = backward(gradient, *saved)
                    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
                    torch.testing.assert_close(initial_gradient, expected_gradient, rtol=0, atol=0)
                    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
                    torch.testing.assert_close(repeated_gradient, expected_gradient, rtol=0, atol=0)

    def test_backward_guards_continue_to_reject_wrong_shape_dtype_and_device(self):
        source = torch.randn(1, 3, 5, 7, dtype=torch.bfloat16)
        forward, backward, _, _, _ = activation_program("tanh", source)
        with torch.no_grad():
            _, saved = forward(source)
            gradient = torch.randn_like(source)
            invalid = (saved[0].view(source.shape), saved[0].float(),
                       torch.empty(tuple(saved[0].shape), dtype=saved[0].dtype, device="meta"))
            for value in invalid:
                with self.subTest(shape=tuple(value.shape), dtype=value.dtype, device=value.device):
                    with self.assertRaisesRegex(ValueError, "compiled tensor metadata"):
                        backward(gradient, value)

    def test_compressed_activation_components_keep_the_existing_codec_abi(self):
        source = torch.randn(2, 3, 17, 19, dtype=torch.bfloat16)
        for kind in ("tanh", "sigmoid", "silu", "gelu"):
            with self.subTest(kind=kind), torch.no_grad():
                forward, backward, trace, specification, _ = activation_program(kind, source, compressed=True)
                _, saved = forward(source)
                self.assertFalse(specification.recompute)
                self.assertEqual(len(saved), 3)
                self.assertEqual([value.dtype for value in saved], [torch.int32, torch.float32, torch.float32])
                self.assertEqual([tuple(value.shape) for value in saved], [value.meta.shape for value in trace.saved_tensors])
                actual, = backward(torch.randn_like(source), *saved)
                self.assertEqual((tuple(actual.shape), actual.dtype), (tuple(source.shape), source.dtype))
                self.assertEqual(saved[0].shape[0], (prod(source.shape) + 1023) // 1024)


@unittest.skipUnless(torch.cuda.is_available(), "Native activation saved-state CUDA regression requires a GPU")
class NativeActivationSavedABICudaTests(unittest.TestCase):
    def test_actual_cuda_nd_snapshots_match_declared_metadata_and_eager_vjp(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 CUDA support is required")
        for kind in ("tanh", "sigmoid", "silu", "gelu"):
            for layout in ("contiguous", "channels_last", "strided"):
                with self.subTest(kind=kind, layout=layout):
                    source = torch.randn(1, 3, 224, 448 if layout == "strided" else 224,
                                         dtype=torch.bfloat16, device="cuda")
                    if layout == "strided":
                        source = source[..., ::2]
                    elif layout == "channels_last":
                        source = source.contiguous(memory_format=torch.channels_last)
                    forward, backward, trace, _, eager = activation_program(kind, source)
                    control = source.detach().clone(memory_format=torch.preserve_format).requires_grad_()
                    expected_output = eager(control)
                    gradient = torch.randn_like(expected_output)
                    expected_output.backward(gradient)
                    with torch.no_grad():
                        actual_output, saved = forward(source)
                        self.assertEqual(tuple(saved[0].shape), trace.saved_tensors[0].meta.shape)
                        self.assertEqual(tuple(saved[0].stride()), trace.saved_tensors[0].meta.stride)
                        actual_gradient, = backward(gradient, *saved)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
                    torch.testing.assert_close(actual_gradient, control.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
