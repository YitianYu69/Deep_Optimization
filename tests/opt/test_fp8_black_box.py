"""Numerical FP8 training and adapter contracts through executable APIs."""

from copy import deepcopy
import io
import itertools
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from deepopt.opt.do_nn.functional import do_linear
from deepopt.opt.do_nn.linear_module import DOLinear
from deepopt.opt.do_nn.mha_module import DOMultiheadAttention
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.opt.fp8_support import round_fp8


def _plan(blocks=None, *, out_dtype=None):
    return {
        "fp8_training": {
            "fp8_dtypes": {
                "activation": torch.float8_e4m3fn,
                "weight": torch.float8_e4m3fn,
                "grad": torch.float8_e5m2,
            },
            "backend": "reference",
            "out_dtype": out_dtype,
        },
        "block_shapes": dict.fromkeys(("activation", "weight", "grad"), blocks),
    }


def _operand(matrix, plan, role):
    return round_fp8(
        matrix, plan["fp8_training"]["fp8_dtypes"][role],
        plan["block_shapes"][role],
    )


class FP8LinearBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(804)

    def test_forward_backward_match_independent_quantized_math(self):
        for blocks, shape, bias_enabled in itertools.product(
            (None, (2, 3), (1, 3), (1, 32)), ((7,), (5, 7), (2, 3, 7)), (False, True)
        ):
            with self.subTest(blocks=blocks, shape=shape, bias=bias_enabled):
                plan = _plan(blocks)
                x = torch.randn(shape, requires_grad=True)
                weight = torch.randn(4, 7, requires_grad=True)
                bias = torch.randn(4, requires_grad=True) if bias_enabled else None
                output = do_linear(x, weight, bias, plan=plan)
                grad = torch.randn_like(output)
                output.backward(grad)

                qx = _operand(x.reshape(-1, 7), plan, "activation")
                qw = _operand(weight, plan, "weight")
                qg = _operand(grad.reshape(-1, 4), plan, "grad")
                expected = qx @ qw.T
                if bias is not None:
                    expected += bias.detach()
                torch.testing.assert_close(output, expected.reshape(output.shape))
                torch.testing.assert_close(x.grad, (qg @ qw).reshape(x.shape))
                if blocks in ((1, 3), (1, 32)):
                    # Rowwise training quantizes each GEMM orientation
                    # independently from the original higher-precision values.
                    qxt = _operand(x.reshape(-1, 7).T, plan, "activation")
                    qgt = _operand(grad.reshape(-1, 4).T, plan, "grad")
                    expected_weight_grad = qgt @ qxt.T
                else:
                    expected_weight_grad = qg.T @ qx
                torch.testing.assert_close(weight.grad, expected_weight_grad)
                if bias is not None:
                    torch.testing.assert_close(bias.grad, grad.reshape(-1, 4).sum(0))

    def test_rowwise_weight_gradient_quantizes_original_transpose_independently(self):
        plan = _plan()
        plan["block_shapes"]["activation"] = (1, 2)
        x = torch.tensor([[1., 100.], [2., 2.]])
        weight = torch.ones(1, 2, requires_grad=True)
        gradient = torch.tensor([[1.], [0.]])
        do_linear(x, weight, plan=plan).backward(gradient)
        expected = _operand(gradient, plan, "grad").T @ _operand(x.T, plan, "activation").T
        stored_forward_math = _operand(gradient, plan, "grad").T @ _operand(x, plan, "activation")
        self.assertNotEqual(expected[0, 0].item(), stored_forward_math[0, 0].item())
        torch.testing.assert_close(weight.grad, expected, rtol=0, atol=0)

    def test_each_operand_can_use_a_distinct_scaling_grid(self):
        plan = _plan()
        plan["block_shapes"] = {"activation": (2, 3), "weight": (3, 4), "grad": (4, 2)}
        x = torch.randn(5, 7, requires_grad=True)
        weight = torch.randn(6, 7, requires_grad=True)
        output = do_linear(x, weight, plan=plan)
        grad = torch.randn_like(output)
        output.backward(grad)
        qx, qw, qg = (_operand(value, plan, role) for value, role in (
            (x, "activation"), (weight, "weight"), (grad, "grad")
        ))
        torch.testing.assert_close(output, qx @ qw.T)
        torch.testing.assert_close(x.grad, qg @ qw)
        torch.testing.assert_close(weight.grad, qg.T @ qx)

    def test_noncontiguous_inputs_weights_and_gradients(self):
        plan = _plan((2, 3))
        x = torch.randn(7, 5).T.requires_grad_()
        weight = torch.randn(7, 4).T.requires_grad_()
        grad = torch.randn(4, 5).T
        self.assertFalse(x.is_contiguous())
        self.assertFalse(weight.is_contiguous())
        output = do_linear(x, weight, plan=plan)
        output.backward(grad)
        qx, qw, qg = (_operand(value, plan, role) for value, role in (
            (x, "activation"), (weight, "weight"), (grad, "grad")
        ))
        torch.testing.assert_close(output, qx @ qw.T)
        torch.testing.assert_close(x.grad, qg @ qw)
        torch.testing.assert_close(weight.grad, qg.T @ qx)

    def test_requires_grad_combinations_and_no_grad_inference(self):
        for flags in itertools.product((False, True), repeat=3):
            with self.subTest(flags=flags):
                args = [torch.randn(shape, requires_grad=flag) for shape, flag in zip(
                    ((5, 7), (4, 7), (4,)), flags
                )]
                output = do_linear(*args, plan=_plan())
                self.assertEqual(output.requires_grad, any(flags))
                if any(flags):
                    output.sum().backward()
                for value, flag in zip(args, flags):
                    self.assertEqual(value.grad is not None, flag)
                    if flag:
                        self.assertTrue(torch.isfinite(value.grad).all())
        with torch.no_grad():
            self.assertFalse(do_linear(*args, plan=_plan()).requires_grad)

    def test_output_dtype_and_original_gradient_dtypes(self):
        for input_dtype, weight_dtype, bias_dtype, output_dtype in (
            (torch.float32, torch.bfloat16, torch.float16, torch.float32),
            (torch.bfloat16, torch.float32, torch.float32, torch.bfloat16),
            (torch.float16, torch.float32, torch.bfloat16, torch.float16),
        ):
            with self.subTest(input=input_dtype, weight=weight_dtype, output=output_dtype):
                plan = _plan((2, 3), out_dtype=output_dtype)
                x = torch.randn(5, 7, dtype=input_dtype, requires_grad=True)
                weight = torch.randn(4, 7, dtype=weight_dtype, requires_grad=True)
                bias = torch.randn(4, dtype=bias_dtype, requires_grad=True)
                output = do_linear(x, weight, bias, plan=plan)
                grad = torch.randn_like(output)
                output.backward(grad)
                qx = _operand(x, plan, "activation")
                qw = _operand(weight, plan, "weight")
                qg = _operand(grad, plan, "grad")
                self.assertEqual(output.dtype, output_dtype)
                torch.testing.assert_close(output, (qx @ qw.T + bias.float()).to(output_dtype))
                torch.testing.assert_close(x.grad, (qg @ qw).to(input_dtype))
                torch.testing.assert_close(weight.grad, (qg.T @ qx).to(weight_dtype))
                torch.testing.assert_close(bias.grad, grad.float().sum(0).to(bias_dtype))

    def test_autocast_controls_output_but_keeps_float32_accumulation(self):
        x, weight = torch.randn(5, 7), torch.randn(4, 7)
        expected = do_linear(x, weight, plan=_plan())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = do_linear(x, weight, plan=_plan())
            override = do_linear(x, weight, plan=_plan(out_dtype=torch.float32))
        torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(override, expected, rtol=0, atol=0)

    def test_numerical_error_against_dense_training_is_bounded(self):
        x = torch.randn(13, 17, requires_grad=True)
        weight = torch.randn(11, 17, requires_grad=True)
        grad = torch.randn(13, 11)
        output = do_linear(x, weight, plan=_plan((4, 8)))
        output.backward(grad)
        for actual, expected in (
            (output, F.linear(x, weight)),
            (x.grad, grad @ weight),
            (weight.grad, grad.T @ x),
        ):
            relative_error = (actual - expected).norm() / expected.norm()
            self.assertLess(relative_error.item(), .12)

    def test_plan_mutation_after_forward_does_not_change_backward(self):
        plan = _plan((2, 3))
        original_plan = deepcopy(plan)
        x = torch.randn(5, 7, requires_grad=True)
        weight = torch.randn(4, 7, requires_grad=True)
        result = do_linear(x, weight, plan=plan)
        grad = torch.randn_like(result)
        plan["block_shapes"]["activation"] = (1, 1)
        plan["fp8_training"]["fp8_dtypes"]["grad"] = torch.float8_e4m3fn
        result.backward(grad)
        qx = _operand(x, original_plan, "activation")
        qg = _operand(grad, original_plan, "grad")
        torch.testing.assert_close(weight.grad, qg.T @ qx)


class FP8ModuleBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(815)

    def test_shared_parameters_receive_accumulated_gradients_and_optimizer_updates(self):
        original = nn.Linear(7, 4)
        first = DOLinear.from_linear(original, plan=_plan((2, 3)))
        second = DOLinear.from_linear(original, plan=_plan((2, 3)))
        self.assertIs(first.weight, second.weight)
        self.assertIs(first.bias, original.bias)
        model = nn.ModuleList([first, second])
        self.assertEqual(len(list(model.parameters())), 2)
        before = original.weight.detach().clone()
        x = torch.randn(5, 7)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        first(x).sum().backward()
        single_grad = first.weight.grad.clone()
        second(x).sum().backward()
        torch.testing.assert_close(first.weight.grad, 2 * single_grad)
        optimizer.step()
        torch.testing.assert_close(original.weight, before - .2 * single_grad)

    def test_state_dict_loads_native_weights_and_preserves_biasless_keys(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                original = nn.Linear(7, 4, bias=bias).eval()
                transformed = DOLinear.from_linear(original, plan=_plan())
                self.assertFalse(transformed.training)
                self.assertEqual(tuple(transformed.state_dict()), tuple(original.state_dict()))
                fresh = nn.Linear(7, 4, bias=bias)
                transformed.load_state_dict(fresh.state_dict(), strict=True)
                torch.testing.assert_close(original.weight, fresh.weight)
                self.assertIs(transformed.weight, original.weight)

    def test_checkpoint_resume_matches_uninterrupted_adam_training(self):
        plan = _plan((2, 3))
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=plan)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        batches = [(torch.randn(5, 7), torch.randn(5, 4)) for _ in range(4)]

        def update(layer, opt, batch):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(layer(batch[0]), batch[1])
            loss.backward()
            opt.step()
            return loss.detach()

        for batch in batches[:2]:
            update(model, optimizer, batch)
        payload = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, payload)
        expected_losses = [update(model, optimizer, batch) for batch in batches[2:]]
        payload.seek(0)
        checkpoint = torch.load(payload, weights_only=True)
        restored = DOLinear.from_linear(nn.Linear(7, 4), plan=plan)
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.01)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        actual_losses = [update(restored, restored_optimizer, batch) for batch in batches[2:]]
        torch.testing.assert_close(torch.stack(actual_losses), torch.stack(expected_losses), rtol=0, atol=0)
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mha_forward_backward_stays_close_to_dense_with_masks_and_cross_attention(self):
        cases = (
            dict(batch_first=True, unbatched=False, kdim=8, vdim=8, need_weights=True),
            dict(batch_first=False, unbatched=False, kdim=6, vdim=10, need_weights=False),
            dict(batch_first=True, unbatched=True, kdim=8, vdim=8, need_weights=True),
        )
        for case in cases:
            with self.subTest(case=case):
                original = nn.MultiheadAttention(8, 2, batch_first=case["batch_first"],
                                                 kdim=case["kdim"], vdim=case["vdim"])
                dense = deepcopy(original)
                transformed = DOMultiheadAttention.from_mha(original, plan=_plan())
                inputs = [torch.randn(2, length, width) * .4 for length, width in (
                    (3, 8), (4, case["kdim"]), (4, case["vdim"])
                )]
                if case["unbatched"]:
                    inputs = [x[0] for x in inputs]
                elif not case["batch_first"]:
                    inputs = [x.transpose(0, 1) for x in inputs]
                actual_inputs = [x.detach().requires_grad_() for x in inputs]
                dense_inputs = [x.detach().clone().requires_grad_() for x in inputs]
                mask = torch.zeros(3, 4, dtype=torch.bool)
                mask[:, -1] = True
                kwargs = dict(attn_mask=mask, need_weights=case["need_weights"], average_attn_weights=False)
                actual, weights = transformed(*actual_inputs, **kwargs)
                expected, dense_weights = dense(*dense_inputs, **kwargs)
                self.assertLess(((actual - expected).norm() / expected.norm()).item(), .12)
                if weights is not None:
                    torch.testing.assert_close(weights, dense_weights, atol=.012, rtol=.08)
                    self.assertTrue((weights[..., -1] == 0).all())
                grad = torch.randn_like(actual)
                actual.backward(grad)
                expected.backward(grad)
                for actual_input, dense_input in zip(actual_inputs, dense_inputs):
                    relative_error = (actual_input.grad - dense_input.grad).norm() / dense_input.grad.norm()
                    self.assertLess(relative_error.item(), .2)
                parameter_names = tuple(dict(dense.named_parameters()))
                actual_grads = torch.cat([
                    transformed.get_parameter(name).grad.flatten() for name in parameter_names
                ])
                dense_grads = torch.cat([
                    dense.get_parameter(name).grad.flatten() for name in parameter_names
                ])
                self.assertLess(((actual_grads - dense_grads).norm() / dense_grads.norm()).item(), .15)

    def test_separate_qkv_attention_restores_native_optimizer_checkpoint(self):
        original = nn.MultiheadAttention(8, 2, kdim=6, vdim=10, batch_first=True)
        optimizer = torch.optim.Adam(original.parameters(), lr=.01)
        query, key, value = (torch.randn(2, length, width) for length, width in ((3, 8), (4, 6), (4, 10)))
        original(query, key, value)[0].square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = deepcopy(optimizer.state_dict())
        restored = DOMultiheadAttention.from_mha(deepcopy(original), plan=_plan())
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.01)
        restored_optimizer.load_state_dict(state)
        expected = DOMultiheadAttention.from_mha(original, plan=_plan())
        for module, opt in ((expected, optimizer), (restored, restored_optimizer)):
            opt.zero_grad(set_to_none=True)
            module(query, key, value)[0].square().mean().backward()
            opt.step()
        for name, parameter in expected.named_parameters():
            torch.testing.assert_close(restored.get_parameter(name), parameter, rtol=0, atol=0)


class FP8QuantizationBlackBoxTests(unittest.TestCase):
    def test_zero_and_edge_blocks_match_independent_oracle(self):
        matrix = torch.tensor([[0., 0., 4., -7., .125], [0., 0., -2., 1., .5],
                               [100., -31., .01, .02, .03]])
        for dtype, block in itertools.product(
            (torch.float8_e4m3fn, torch.float8_e5m2), (None, (2, 2), (4, 8))
        ):
            with self.subTest(dtype=dtype, block=block):
                actual = quantize_fp8(matrix, fp8_dtype=dtype, block_shape=block)
                torch.testing.assert_close(actual.dequantize(), round_fp8(matrix, dtype, block), rtol=0, atol=0)
                transposed = actual.transpose()
                torch.testing.assert_close(transposed.dequantize(), actual.dequantize().T, rtol=0, atol=0)
                torch.testing.assert_close(transposed.transpose().dequantize(), actual.dequantize(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
