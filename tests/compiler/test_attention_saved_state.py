"""Typed optional SDPA state through the real native saved-variable bridge."""

from contextlib import ExitStack
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
import torch.distributed.tensor  # Register variants before mocking private dispatcher operators.
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.attention import SDPBackend

from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import AuxiliaryTensorMeta, IRGraph, TensorMeta, TensorValue
from deepopt.compiler.scheduler import Scheduler
from deepopt.compiler.trace.builder import GraphBuilder


class AttentionSavedStateTests(unittest.TestCase):
    def test_cuda_flash_padding_keeps_logical_saved_shapes_and_original_scale(self):
        self._check_cuda_flash_padding((0.,))

    def test_cuda_flash_padding_preserves_dropout_state_plumbing(self):
        self._check_cuda_flash_padding((.2,))

    def test_padded_flash_gradient_layout_is_unknown_to_flat_pointwise_codegen(self):
        from deepopt.compiler.autodiff.optimization import backward_layouts
        from deepopt.compiler.ir import CallNode, PointwiseNode
        from deepopt.compiler.scheduler import valid_same_shape_region
        from deepopt.compiler.trace.metadata_utils import contiguous_stride

        shape = (1, 2, 3, 6)
        actual_gradient = torch.empty(1, 2, 3, 8)[..., :6]
        metadata = TensorMeta(shape, torch.float32, "cuda:0", contiguous_stride(shape))
        self.assertNotEqual(actual_gradient.stride(), metadata.stride)
        gradient = TensorValue("attention_gradient", metadata)
        negated = TensorValue("negated", metadata)
        graph = IRGraph()
        graph.add(CallNode("attention.sdpa_vjp", lambda: None, (), {}, (gradient,)))
        consumer = PointwiseNode("neg", (gradient,), {}, negated, torch.neg)
        graph.add(consumer)
        self.assertTrue(valid_same_shape_region([consumer]))
        # JointProgram runs this pass after VJP/fusion rewrites and before
        # scheduling. Native consumers keep runtime strides; the flat CUDA
        # pointwise ABI cannot rely on the copied primal layout.
        backward_layouts(graph)
        self.assertIsNone(gradient.meta.stride)
        self.assertIsNone(negated.meta.stride)
        self.assertFalse(valid_same_shape_region([consumer]))

    def _check_cuda_flash_padding(self, dropout_cases):
        from deepopt.compiler.autodiff.attention import SDPA
        from deepopt.opt.quantization.attention import AttentionCompression

        def attention(q, k, v, scale, dropout):
            probability = (q @ k.transpose(-2, -1) * scale).softmax(-1)
            if dropout:
                # A fixed mask makes dropout-state forwarding testable on CPU;
                # the real CUDA test below exercises the actual RNG kernel.
                mask = torch.arange(probability.numel()).reshape(probability.shape) % 3 != 0
                probability = probability * mask / (1 - dropout)
            return probability @ v

        def flash_backward(g, q, k, v, output, lse, scale, dropout):
            # Private flash backward keeps the exact forward normalization and
            # output even when QKV saved hooks reconstruct approximate inputs.
            probability = (q @ k.transpose(-2, -1) * scale - lse.unsqueeze(-1)).exp()
            dp = g @ v.transpose(-2, -1)
            dropped = probability
            if dropout:
                mask = torch.arange(probability.numel()).reshape(probability.shape) % 3 != 0
                dp = dp * mask / (1 - dropout)
                dropped = probability * mask / (1 - dropout)
            delta = (g * output).sum(-1, keepdim=True)
            ds = probability * (dp - delta)
            return ds @ k * scale, ds.transpose(-2, -1) @ q * scale, dropped.transpose(-2, -1) @ g

        class PublicPaddedFlash(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v, scale, dropout):
                output = attention(q, k, v, scale, dropout)
                lse = (q @ k.transpose(-2, -1) * scale).logsumexp(-1)
                ctx.save_for_backward(q, k, v, output, lse)
                ctx.scale, ctx.dropout = scale, dropout
                return output

            @staticmethod
            def backward(ctx, gradient):
                return (*flash_backward(gradient, *ctx.saved_tensors, ctx.scale, ctx.dropout), None, None)

        for dim in (2, 6, 10):
            for compressed in (False, True):
                for scale in (None, .31):
                    for dropout in dropout_cases:
                        with self.subTest(dim=dim, compressed=compressed, scale=scale, dropout=dropout):
                            torch.manual_seed(1828)
                            # Even D=2 has a full group1024 logical payload.
                            operands = tuple(torch.randn(86, 2, 3, dim) for _ in range(3))
                            metadata = TensorMeta(operands[0].shape, torch.float32, "cpu", requires_grad=True)
                            options = {"compression": {"bits": 4, "group_size": 1024} if compressed else None,
                                       "scale": scale, "dropout_p": dropout, "is_causal": False,
                                       "enable_gqa": False, "backend": "native"}
                            node = SimpleNamespace(options=options,
                                args=(*(TensorValue(str(index), metadata) for index in range(3)), None),
                                value=TensorValue("output", metadata))
                            spec = SDPA(node, None)
                            expected_scale = dim ** -.5 if scale is None else scale
                            padded_dim = ((dim + 7) // 8) * 8
                            seed, offset = torch.tensor([1828]), torch.tensor([4])
                            empty = torch.empty(0)
                            seen = []

                            def forward(q, k, v, p, causal, debug, *, scale):
                                seen.append("forward")
                                self.assertEqual(scale, expected_scale)
                                self.assertEqual(p, dropout)
                                self.assertFalse(causal or debug)
                                for actual, original in zip((q, k, v), operands):
                                    self.assertEqual(actual.shape[-1], padded_dim)
                                    torch.testing.assert_close(actual[..., :dim], original)
                                    self.assertEqual(torch.count_nonzero(actual[..., dim:]).item(), 0)
                                output = attention(q, k, v, scale, p)
                                lse = (q @ k.transpose(-2, -1) * scale).logsumexp(-1)
                                return output, lse, None, None, 3, 3, seed, offset, empty

                            logical_output = original_lse = None

                            def backward(g, q, k, v, output, lse, cu_q, cu_k, max_q, max_k,
                                         p, causal, saved_seed, saved_offset, *, scale):
                                seen.append("backward")
                                self.assertEqual(scale, expected_scale)
                                self.assertIs(saved_seed, seed)
                                self.assertIs(saved_offset, offset)
                                self.assertEqual((max_q, max_k, p, causal), (3, 3, dropout, False))
                                self.assertIsNone(cu_q)
                                self.assertIsNone(cu_k)
                                self.assertEqual(g.shape[-1], padded_dim)
                                self.assertEqual(torch.count_nonzero(g[..., dim:]).item(), 0)
                                self.assertIs(lse, original_lse)
                                torch.testing.assert_close(output[..., :dim], logical_output)
                                self.assertEqual(torch.count_nonzero(output[..., dim:]).item(), 0)
                                return flash_backward(g, q, k, v, output, lse, scale, p)

                            with patch("torch.ops.aten._scaled_dot_product_flash_attention.default", side_effect=forward), \
                                    patch("torch.ops.aten._scaled_dot_product_flash_attention_backward.default", side_effect=backward):
                                logical_output, lse, cu_q, cu_k, _, _, _, _, _ = spec._flash_forward(*operands, dropout, False)
                                original_lse = lse
                                self.assertEqual(logical_output.shape, metadata.shape)
                                torch.testing.assert_close(logical_output, attention(*operands, expected_scale, dropout))
                                packed = tuple(part for value in operands for part in spec._pack(value, empty))
                                restored = tuple(spec._restore(packed[index:index + 3], metadata.shape, metadata.dtype)
                                                 for index in (0, 3, 6))
                                self.assertTrue(all(value.shape == metadata.shape for value in restored))
                                saved = (*packed, lse, empty, empty, empty, seed, offset, cu_q, cu_k,
                                         torch.tensor(int(SDPBackend.FLASH_ATTENTION)))
                                upstream = torch.randn_like(logical_output)
                                actual = spec.gradients(upstream, logical_output, None, *saved)
                            reference = tuple(value.detach().requires_grad_() for value in restored)
                            # Independent autograd surrogate: saved LSE and
                            # delta are constants, rather than recomputing a
                            # normalized softmax from the decoded Q/K values.
                            probability = (reference[0] @ reference[1].transpose(-2, -1)
                                           * expected_scale - original_lse.detach().unsqueeze(-1)).exp()
                            dp = upstream @ reference[2].transpose(-2, -1)
                            if dropout:
                                mask = torch.arange(probability.numel()).reshape(probability.shape) % 3 != 0
                                dp = dp * mask / (1 - dropout)
                            delta = (upstream * logical_output).sum(-1, keepdim=True).detach()
                            expected = torch.autograd.grad((probability * (dp - delta)).sum(), reference)
                            for actual_gradient, expected_gradient in zip(actual, expected):
                                self.assertEqual(actual_gradient.shape, metadata.shape)
                                torch.testing.assert_close(actual_gradient, expected_gradient)
                            if compressed and not dropout:
                                recomputed = torch.autograd.grad(attention(*reference, expected_scale, 0.), reference, upstream)
                                self.assertGreater(max((left - right).abs().max().item()
                                                       for left, right in zip(expected, recomputed)), 1e-4,
                                                   "Fixture must distinguish exact saved flash state from recomputation")

                            # Simulate the public CUDA wrapper: pad before the
                            # private autograd operation, then narrow its output.
                            # Eager hooks and the native codec must compress the
                            # same logical QKV rather than different padded grids.
                            eager_inputs = tuple(value.detach().clone().requires_grad_() for value in operands)
                            with AttentionCompression(*eager_inputs, group_size=1024, bits=4) if compressed else ExitStack():
                                padded = tuple(nn.functional.pad(value, (0, padded_dim - dim)) for value in eager_inputs)
                                eager_output = PublicPaddedFlash.apply(*padded, expected_scale, dropout)[..., :dim]
                            eager_output.backward(upstream)
                            torch.testing.assert_close(logical_output, eager_output)
                            for actual_gradient, eager_input in zip(actual, eager_inputs):
                                torch.testing.assert_close(actual_gradient, eager_input.grad)
                            self.assertEqual(seen, ["forward", "backward"])

    def _identity(self, metadata):
        graph = IRGraph()
        builder = GraphBuilder(graph)
        value = builder.input(0, metadata)
        builder.output((value,))
        return WrapperCodegen(nn.Module(), Scheduler(graph).run(), enable_autodiff=False,
                              consume_inputs=True).run().fn

    def test_optional_auxiliary_accepts_absence_and_empty_present_tensors(self):
        run = self._identity(AuxiliaryTensorMeta(dtype=torch.int32, device="cpu", presence="optional"))
        for value in (None, torch.empty(0, dtype=torch.int32), torch.tensor([0, 3], dtype=torch.int32)):
            with self.subTest(value=value):
                owned = [value]
                result, = run(owned)
                self.assertIs(result, value)
                self.assertEqual(owned, [None])
        for value, error in (([], TypeError), ("undefined", TypeError),
                             (torch.empty(0), ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                run([value])

    def test_optional_contract_does_not_weaken_required_or_dense_guards(self):
        for metadata in (AuxiliaryTensorMeta(), TensorMeta((0,), torch.int32, "cpu")):
            run = self._identity(metadata)
            with self.subTest(metadata=metadata), self.assertRaises(TypeError):
                run([None])
        run = self._identity(AuxiliaryTensorMeta((2,), torch.int32, "cpu", (1,), presence="optional"))
        with self.assertRaises(ValueError):
            run([torch.empty(0, dtype=torch.int32)])
        with self.assertRaises(ValueError):
            AuxiliaryTensorMeta(presence="unknown")

    def _compiled(self, backend="native", dropout=0.):
        class Model(nn.Module):
            def forward(self, q, k, v):
                return nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout)

        values = [torch.randn(1, 2, 3, 4, requires_grad=True) for _ in range(3)]
        graph = torch.fx.symbolic_trace(Model())
        ShapeProp(graph).propagate(*values)
        if backend != "native":
            from deepopt.opt.nn.functional import do_scaled_dot_product_attention

            for node in graph.graph.nodes:
                if node.op == "call_function":
                    node.target = do_scaled_dot_product_attention
                    node.kwargs = {**node.kwargs, "attention_backend": backend}
            graph.recompile()
        report = CompilationReport()
        run = compile_fx(graph, values, config=BackendConfig(require_full_aot=True), report=report)
        return run, values, report

    def _adapter(self, modes, observations, *, malformed=None, payload_refs=None):
        calls = iter(modes)

        def forward(q, k, v, *args, **kwargs):
            mode = next(calls)
            if mode == "absent":
                cu_q = cu_k = None
            elif mode == "empty":
                cu_q = torch.empty(0, dtype=torch.int32)
                cu_k = torch.empty(0, dtype=torch.int32)
            else:
                cu_q = torch.tensor([0, 3], dtype=torch.int32)
                cu_k = torch.tensor([0, 3], dtype=torch.int32)
            if payload_refs is not None and cu_q is not None:
                payload_refs.extend((weakref.ref(cu_q), weakref.ref(cu_k)))
            lse = torch.zeros(q.shape[:-1])
            if malformed == "required":
                lse = None
            elif malformed == "optional":
                cu_q = "not a tensor"
            elif malformed == "dtype":
                cu_q = torch.zeros(0)
            elif malformed == "rank":
                cu_q = torch.zeros(2, 2, dtype=torch.int32)
            return q * k + v, lse, cu_q, cu_k, q.shape[-2], k.shape[-2], None, None, torch.empty(0)

        def backward(g, q, k, v, output, lse, seed, offset, mask, cu_q, cu_k, *args, **kwargs):
            self.assertIsNone(seed)
            self.assertIsNone(offset)
            self.assertIsNone(mask)
            self.assertEqual(cu_q is None, cu_k is None)
            observations.append(None if cu_q is None else tuple(cu_q.tolist()))
            torch.testing.assert_close(output, q * k + v)
            return g * k, g * q, g

        stack = ExitStack()
        stack.enter_context(patch("torch.ops.aten._fused_sdp_choice.default", return_value=int(SDPBackend.CUDNN_ATTENTION)))
        stack.enter_context(patch("torch.ops.aten._scaled_dot_product_cudnn_attention.default", side_effect=forward))
        stack.enter_context(patch("torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default", side_effect=backward))
        return stack

    def test_backend_absence_empty_and_present_slots_survive_native_backward(self):
        run, values, report = self._compiled()
        observed = []
        with self._adapter(("absent", "empty", "present"), observed):
            for _ in range(3):
                for value in values:
                    value.grad = None
                actual = run(*values)
                actual.backward(torch.ones_like(actual))
                for value, expected in zip(values, (values[1], values[0], torch.ones_like(values[2]))):
                    torch.testing.assert_close(value.grad, expected)
        self.assertEqual(observed, [None, (), (0, 3)])
        self.assertFalse(report.graphs[0].legacy_backward_operations)
        self.assertFalse(report.graphs[0].eager_autograd_operations)
        self.assertGreater(report.graphs[0].native_autodiff_operations, 0)

    def test_interleaved_and_retained_graphs_keep_invocation_specific_state(self):
        run, first, _ = self._compiled()
        second = [torch.randn_like(value).requires_grad_() for value in first]
        observed = []
        with self._adapter(("absent", "present"), observed):
            # Backward deliberately executes outside the original precision scope.
            with torch.autocast("cpu", enabled=False):
                a, b = run(*first), run(*second)
            b.backward(torch.ones_like(b), retain_graph=True)
            a.backward(torch.ones_like(a), retain_graph=True)
            b.backward(torch.ones_like(b))
        self.assertEqual(observed, [(0, 3), None, (0, 3)])
        for values, multiplier in ((first, 1), (second, 2)):
            for value, expected in zip(values, (values[1], values[0], torch.ones_like(values[2]))):
                torch.testing.assert_close(value.grad, expected * multiplier)

    def test_malformed_backend_state_is_rejected_before_saving(self):
        run, values, _ = self._compiled()
        for malformed, name, error in (("required", "lse", TypeError),
                                       ("optional", "cu_q", TypeError), ("dtype", "cu_q", ValueError),
                                       ("rank", "cu_q", ValueError)):
            with self.subTest(malformed=malformed), self._adapter(("absent",), [], malformed=malformed):
                with self.assertRaisesRegex(error, name):
                    run(*values)

    def test_dropout_requires_fused_backend_rng_state(self):
        run, values, _ = self._compiled(dropout=.2)
        with self._adapter(("absent",), []), self.assertRaisesRegex(TypeError, "state 'seed'"):
            run(*values)

    def test_optional_payload_release_respects_retained_graphs_and_saved_hooks(self):
        run, values, _ = self._compiled()
        refs, observed = [], []
        packed_types = []

        def pack(value):
            packed_types.append(value.dtype)
            return value

        with self._adapter(("present",), observed, payload_refs=refs), \
                torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            output = run(*values)
        self.assertTrue(all(ref() is not None for ref in refs))
        with self._adapter((), observed):
            output.backward(torch.ones_like(output), retain_graph=True)
            self.assertTrue(all(ref() is not None for ref in refs))
            output.backward(torch.ones_like(output))
        # unittest.mock records native call arguments in reference cycles.
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(packed_types.count(torch.int32), 2)
        self.assertEqual(observed, [(0, 3), (0, 3)])

    def test_external_flash_adapter_keeps_its_separate_rng_state_contract(self):
        run, values, _ = self._compiled("flash_attn_2")
        seeds = iter((None, torch.tensor([173], dtype=torch.int64)))
        observed = []

        def forward(q, k, v, *args):
            return q * k + v, torch.zeros(q.shape[:-1]), None, next(seeds)

        def backward(g, q, k, v, output, lse, dq, dk, dv, *args, rng_state=None):
            observed.append(None if rng_state is None else tuple(rng_state.tolist()))
            dq.copy_(g * k)
            dk.copy_(g * q)
            dv.copy_(g)

        with patch("deepopt.opt.nn.attention_backend._validate_and_cast", side_effect=lambda q, k, v, *args: (q, k, v)), \
                patch("deepopt.opt.nn.attention_backend._kernels", return_value=(forward, backward)):
            first, second = run(*values), run(*values)
            second.backward(torch.ones_like(second))
            first.backward(torch.ones_like(first))
        self.assertEqual(observed, [(173,), None])
        for value, expected in zip(values, (values[1], values[0], torch.ones_like(values[2]))):
            torch.testing.assert_close(value.grad, expected * 2)


if __name__ == "__main__":
    unittest.main()
