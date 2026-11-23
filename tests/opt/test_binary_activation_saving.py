"""Multiplication forward semantics and the materialized saved-coefficient VJP."""

import gc
import operator
import unittest
import weakref
from unittest.mock import patch

import torch

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.nn.binary import multiplication_support
from deepopt._activation_memory import activation_memory_scope
from deepopt.kernel_policy import kernel_policy_scope
from deepopt.opt.quantization.activation import dequantize_unpack_activation, quantize_pack_activation


POLICY = {"enabled": True, "group_size": 1024, "bits": 2}
PLAN = {"fp8": False, "activation_compression": POLICY}
STREAMING_PLAN = {**PLAN, "activation_memory": {"multiply_backward": "packed_streaming"}}


def reconstructed(value):
    matrix = value.reshape((1, value.numel()) if value.ndim < 2 else (-1, value.shape[-1]))
    return dequantize_unpack_activation(quantize_pack_activation(matrix, 1024, bits=2),
                                       dtype=value.dtype, row_alignment=1).reshape(value.shape)


class BinaryActivationSavingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(928)

    def test_all_gradient_masks_save_only_consumed_coefficients(self):
        for left, right in ((True, False), (False, True), (True, True), (False, False)):
            with self.subTest(mask=(left, right)):
                x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=left)
                y = torch.randn_like(x, requires_grad=right)
                info, saved = {}, []
                with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value,
                                                            lambda value: value):
                    result = doF.do_mul(x, y, plan=PLAN, _execution_info=info)
                torch.testing.assert_close(result, torch.mul(x, y), rtol=0, atol=0)
                self.assertEqual(info["gradient_mask"], (left, right))
                self.assertEqual(info["coefficient_states"], tuple("packed" if need else "unused"
                                                                 for need in (left, right)))
                self.assertEqual(len(saved), 3 * (left + right))
                self.assertFalse(any(value is x or value is y for value in saved))
                if left or right:
                    gradient = torch.randn_like(result)
                    result.backward(gradient)
                    for source, coefficient in ((x, y), (y, x)):
                        if source.requires_grad:
                            torch.testing.assert_close(source.grad, gradient * reconstructed(coefficient), rtol=0, atol=0)

    def test_se_swiglu_and_residual_scale_follow_reconstructed_oracle(self):
        cases = (((2, 4, 16, 8), (2, 4, 1, 1)), ((2, 512), (2, 512)), ((2, 512), ()))
        for shape, coefficient_shape in cases:
            with self.subTest(shape=shape, coefficient_shape=coefficient_shape):
                x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
                y = torch.randn(coefficient_shape, dtype=torch.bfloat16, requires_grad=True)
                info = {}
                result = doF.do_mul(x, y, plan=PLAN, _execution_info=info)
                gradient = torch.randn_like(result)
                result.backward(gradient)
                restore_y = reconstructed(y) if y.numel() >= 1024 else y.detach()
                expected_x = (gradient * restore_y).sum_to_size(shape)
                expected_y = (gradient * reconstructed(x)).sum_to_size(coefficient_shape)
                torch.testing.assert_close(x.grad, expected_x, rtol=0, atol=0)
                torch.testing.assert_close(y.grad, expected_y, rtol=0, atol=0)
                self.assertEqual(info["rounding_domain"], "operand_dtype")

    def test_x_squared_shares_one_invocation_pack_and_retained_graph(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        info, saved = {}, []
        with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
            result = doF.do_mul(x, x, plan=PLAN, _execution_info=info)
        self.assertEqual(len(saved), 3)
        coefficient = reconstructed(x)
        gradient = torch.randn_like(x)
        expected = gradient * coefficient + gradient * coefficient
        result.backward(gradient, retain_graph=True)
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
        x.grad = None
        result.backward(gradient)
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)

    def test_scalar_promotion_and_frozen_scalar_do_not_save_large_input(self):
        for left_scalar in (False, True):
            for scalar in (1.25, torch.tensor(1.25), torch.tensor(1.25, requires_grad=True)):
                with self.subTest(left_scalar=left_scalar, tensor=isinstance(scalar, torch.Tensor)):
                    x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
                    operands = (scalar, x) if left_scalar else (x, scalar)
                    refs = tuple(value.detach().clone().requires_grad_(value.requires_grad)
                                 if isinstance(value, torch.Tensor) else value for value in operands)
                    info = {}
                    result = doF.do_mul(*operands, plan=PLAN, _execution_info=info)
                    expected = torch.mul(*refs)
                    self.assertEqual(result.dtype, expected.dtype)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    gradient = torch.randn_like(result)
                    result.backward(gradient)
                    if isinstance(scalar, torch.Tensor) and scalar.requires_grad:
                        torch.testing.assert_close(scalar.grad, (gradient * reconstructed(x)).sum().float(), rtol=0, atol=0)
                        scalar.grad = None
                    else:
                        self.assertNotIn("packed", info["coefficient_states"])
                        self.assertLessEqual(info["saved_state_bytes"], 4)
                    expected.backward(gradient)
                    torch.testing.assert_close(x.grad, refs[1 if left_scalar else 0].grad, rtol=0, atol=0)

    def test_zero_stride_transposed_and_shared_views(self):
        base = torch.randn(128, 16, dtype=torch.bfloat16)
        for x, y in ((base.T, torch.randn(128, dtype=torch.bfloat16)),
                     (base, torch.randn(1, 16, dtype=torch.bfloat16).expand_as(base)),
                     (base[:, :8], base[:, 8:])):
            with self.subTest(shape=tuple(x.shape), stride=x.stride()):
                x, y = x.detach().requires_grad_(), y.detach().requires_grad_()
                result = doF.do_mul(x, y, plan=PLAN)
                gradient = torch.randn_like(result)
                result.backward(gradient)
                cx = reconstructed(x) if x.numel() >= 1024 else x.detach()
                cy = reconstructed(y) if y.numel() >= 1024 and all(y.stride()) else y.detach()
                torch.testing.assert_close(x.grad, (gradient * cy).sum_to_size(x.shape), rtol=0, atol=0)
                torch.testing.assert_close(y.grad, (gradient * cx).sum_to_size(y.shape), rtol=0, atol=0)

    def test_no_grad_and_inference_allocate_no_backward_payload(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        with patch("deepopt.opt.nn.binary.quantize_pack_activation", side_effect=AssertionError("unexpected pack")):
            for scope in (torch.no_grad, torch.inference_mode):
                with scope():
                    info = {}
                    result = doF.do_mul(x, x, plan=PLAN, _execution_info=info)
                    self.assertIsNone(result.grad_fn)
                    self.assertEqual(info["saved_state_bytes"], 0)

    def test_empty_broadcast_gradients_preserve_shape_and_zero_reduction(self):
        for plan in (PLAN, STREAMING_PLAN):
            with self.subTest(memory=plan.get("activation_memory")):
                x = torch.empty(0, 1024, dtype=torch.bfloat16, requires_grad=True)
                gate = torch.randn(1, 1024, dtype=torch.bfloat16, requires_grad=True)
                result = doF.do_mul(x, gate, plan=plan)
                result.sum().backward()
                self.assertEqual(tuple(x.grad.shape), (0, 1024))
                torch.testing.assert_close(gate.grad, torch.zeros_like(gate), rtol=0, atol=0)

    def test_public_identities_intercept_only_out_of_place_calls(self):
        operations = (operator.mul, torch.mul, torch.multiply, torch.Tensor.mul, torch.Tensor.multiply,
                      torch.Tensor.__mul__, torch.Tensor.__rmul__)
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        y = torch.randn_like(x, requires_grad=True)
        original = doF.do_mul
        with deepopt.autocast("cpu", dtype=torch.bfloat16, activation_compression=POLICY):
            with patch.object(doF, "do_mul", wraps=original) as intercepted:
                for operation in operations:
                    result = operation(x, y)
                    self.assertEqual(type(result.grad_fn).__name__, "MultiplicationFunctionBackward")
                self.assertEqual(intercepted.call_count, len(operations))
                with torch.no_grad():
                    target = torch.empty_like(x)
                    torch.mul(x, y, out=target)
                    x.detach().mul_(y)
                self.assertEqual(intercepted.call_count, len(operations))

    def test_nested_disabled_scope_keeps_exact_native_multiplication(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        with deepopt.autocast("cpu", dtype=torch.bfloat16, activation_compression=POLICY):
            with deepopt.autocast("cpu", dtype=torch.bfloat16, activation_compression=False):
                result = x * x
                self.assertNotEqual(type(result.grad_fn).__name__, "MultiplicationFunctionBackward")
            self.assertEqual(type((x * x).grad_fn).__name__, "MultiplicationFunctionBackward")

    def test_packed_alias_mutation_uses_snapshot_and_fresh_forward_repacks(self):
        source = torch.randn(2, 512, dtype=torch.bfloat16)
        x = torch.randn_like(source, requires_grad=True)
        coefficient = reconstructed(source)
        first = doF.do_mul(x, source, plan=PLAN)
        with torch.no_grad():
            source.view(-1).add_(3)
        second = doF.do_mul(x, source, plan=PLAN)
        second_coefficient = reconstructed(source)
        first.sum().backward()
        torch.testing.assert_close(x.grad, coefficient, rtol=0, atol=0)
        x.grad = None
        second.sum().backward()
        torch.testing.assert_close(x.grad, second_coefficient, rtol=0, atol=0)

    def test_packed_frozen_coefficient_does_not_retain_original_for_version_check(self):
        source = torch.randn(2, 512, dtype=torch.bfloat16)
        source_ref = weakref.ref(source)
        x = torch.randn_like(source, requires_grad=True)
        coefficient = reconstructed(source)
        result = doF.do_mul(x, source, plan=PLAN)
        del source
        gc.collect()
        self.assertIsNone(source_ref())
        result.sum().backward()
        torch.testing.assert_close(x.grad, coefficient, rtol=0, atol=0)

    def test_backward_keeps_the_forward_recipe_after_scope_changes(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        coefficient = reconstructed(x)
        with deepopt.autocast("cpu", dtype=torch.bfloat16, activation_compression=POLICY):
            result = x * x
        with deepopt.autocast("cpu", dtype=torch.bfloat16, activation_compression={"enabled": True, "bits": 8, "group_size": 1024}):
            result.sum().backward()
        torch.testing.assert_close(x.grad, coefficient + coefficient, rtol=0, atol=0)

    def test_explicit_streaming_cpu_fallback_reports_the_actual_route(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        scalar = torch.tensor(1.25, requires_grad=True)
        coefficient, info = reconstructed(x), {}
        result = doF.do_mul(x, scalar, plan=STREAMING_PLAN, _execution_info=info)
        seed = torch.randn_like(result)
        result.backward(seed)
        torch.testing.assert_close(scalar.grad, (seed * coefficient).sum().float(), rtol=0, atol=0)
        route = info["multiply_backward"][1]
        self.assertEqual((route["route"], route["rejection_reason"]), ("materialized", "device"))
        self.assertEqual(route["scratch_bytes"], 0)
        self.assertGreater(route["decoded_full_temporary_bytes"], 0)

    def test_streaming_selection_scratch_and_kernel_policy_are_forward_owned(self):
        from deepopt.opt.nn.binary import packed_multiply

        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        info = {}
        with activation_memory_scope({"multiply_backward": "packed_streaming", "reduction_scratch_bytes": 1024}, "cpu"):
            result = doF.do_mul(x, x, plan=PLAN, _execution_info=info)
        with activation_memory_scope({"multiply_backward": "materialized"}, "cpu"), kernel_policy_scope("tilelang"):
            with patch("deepopt.opt.nn.binary.packed_multiply", wraps=packed_multiply) as executor:
                result.sum().backward(retain_graph=True)
        self.assertEqual(executor.call_count, 2)
        for call in executor.call_args_list:
            self.assertEqual(call.kwargs["scratch_bytes"], 1024)
            self.assertEqual(call.kwargs["kernel_policy"].backend, "triton")
        first = x.grad.clone()
        x.grad = None
        result.sum().backward()
        torch.testing.assert_close(x.grad, first, rtol=0, atol=0)

    def test_explicit_materialized_remains_the_saved_coefficient_oracle(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        plan = {**PLAN, "activation_memory": {"multiply_backward": "materialized"}}
        info, coefficient = {}, reconstructed(x)
        result = doF.do_mul(x, x, plan=plan, _execution_info=info)
        with patch("deepopt.opt.nn.binary.packed_multiply", side_effect=AssertionError("unexpected packed dispatch")):
            result.sum().backward()
        torch.testing.assert_close(x.grad, coefficient + coefficient, rtol=0, atol=0)
        self.assertTrue(all(route["rejection_reason"] == "memory_policy_materialized"
                            for route in info["multiply_backward"]))

    def test_operation_plan_rejects_unimplemented_or_pinned_conflicts_before_packing(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        for memory, policy, error in (({"pack_reuse": "forward"}, "triton", NotImplementedError),
                                      ({"recompute_modules": ["block"]}, "triton", NotImplementedError),
                                      ({"multiply_backward": "packed_streaming"}, "tilelang", ValueError)):
            plan = {**PLAN, "activation_memory": memory, "kernel_policy": {"backend": policy}}
            with self.subTest(memory=memory, policy=policy):
                rng = torch.get_rng_state()
                with patch("deepopt.opt.nn.binary.quantize_pack_activation") as pack:
                    with self.assertRaises(error):
                        doF.do_mul(x, x, plan=plan)
                    pack.assert_not_called()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a GPU")
    def test_cuda_streaming_matches_the_oracle_without_dense_eager_restore(self):
        for shape, target in (((2, 4, 33, 17), (2, 4, 1, 1)), ((2, 1024), (2, 1024)), ((2, 1024), ())):
            with self.subTest(shape=shape, target=target):
                source = torch.randn(shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
                gate = torch.randn(target, dtype=torch.float32 if target == () else torch.bfloat16,
                                   device="cuda", requires_grad=True)
                info = {}
                expected_coefficient = reconstructed(source)
                result = doF.do_mul(source, gate, plan=STREAMING_PLAN, _execution_info=info)
                gradient = torch.randn_like(result)
                original = gradient.clone()
                expected = (gradient * expected_coefficient).sum_to_size(target).to(gate.dtype)
                with patch("deepopt.opt.nn.binary.dequantize_unpack_activation",
                           side_effect=AssertionError("packed route decoded a dense eager coefficient")):
                    result.backward(gradient)
                torch.testing.assert_close(gate.grad, expected, rtol=.015, atol=.015)
                torch.testing.assert_close(gradient, original, rtol=0, atol=0)
                route = info["multiply_backward"][1]
                self.assertEqual(route["route"], "packed_streaming")
                self.assertEqual((route["decoded_full_temporary_bytes"], route["product_full_temporary_bytes"]), (0, 0))
                self.assertLessEqual(route["scratch_bytes"], 8388608)

    def test_exact_coefficients_and_parameters_keep_version_errors(self):
        for coefficient in (torch.ones(1, dtype=torch.bfloat16),
                            torch.nn.Parameter(torch.randn(2, 512, dtype=torch.bfloat16), requires_grad=False)):
            with self.subTest(parameter=isinstance(coefficient, torch.nn.Parameter)):
                x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
                result = doF.do_mul(x, coefficient, plan=PLAN)
                with torch.no_grad():
                    coefficient.add_(1)
                with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                    result.sum().backward()

    def test_saved_tensor_hooks_release_after_backward_and_interleaved_calls(self):
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        refs = []
        with torch.autograd.graph.saved_tensors_hooks(lambda value: refs.append(weakref.ref(value)) or value,
                                                    lambda value: value):
            a = doF.do_mul(x, x, plan=PLAN)
            b = doF.do_mul(x + 1, x + 1, plan=PLAN)
        self.assertEqual(len(refs), 9)
        a.sum().backward(retain_graph=True)
        self.assertTrue(all(ref() is not None for ref in refs))
        a.sum().backward()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs[:3]))
        self.assertTrue(all(ref() is not None for ref in refs[3:]))
        b.sum().backward()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_unsupported_values_have_deliberate_native_path(self):
        class Subclass(torch.Tensor):
            pass

        cases = ((torch.ones(3, dtype=torch.int64), 3), (torch.ones(3, dtype=torch.complex64), 1j),
                 (torch.ones(3).as_subclass(Subclass), 2),
                 (torch.eye(3).to_sparse(), 2))
        for x, other in cases:
            with self.subTest(dtype=x.dtype, layout=x.layout):
                self.assertIsNotNone(multiplication_support(x, other))
                info = {}
                result = doF.do_mul(x, other, plan=PLAN, _execution_info=info)
                expected = torch.mul(x, other)
                if result.layout == torch.strided:
                    torch.testing.assert_close(result, expected)
                else:
                    torch.testing.assert_close(result.to_dense(), expected.to_dense())
                self.assertIn("native_fallback", info)


if __name__ == "__main__":
    unittest.main()
