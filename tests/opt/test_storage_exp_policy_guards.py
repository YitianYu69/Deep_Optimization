"""Pointer-kernel view guards, verified without claiming CUDA execution."""

import ast
from importlib.resources import files
from types import SimpleNamespace
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


class StorageExpPolicyGuardTests(unittest.TestCase):
    def test_lazy_negative_source_and_cotangent_are_rejected_before_launch(self):
        path = files("deepopt.opt.backends.triton").joinpath("activation/storage_exp.py")
        tree = ast.parse(path.read_text())
        function, = (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_validate")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        with FakeTensorMode():
            original = torch.empty(1031, device="cuda", dtype=torch.bfloat16)
            namespace["_validate"](original, bits=2, group_size=1024)
            with self.assertRaisesRegex(ValueError, "resolved negative"):
                namespace["_validate"](original._neg_view(), bits=2, group_size=1024)

    def test_lazy_negative_residual_is_rejected_before_device_scope_or_launch(self):
        path = files("deepopt.opt.backends.triton").joinpath("activation/storage_exp.py")
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_validate", "backward")]
        for node in functions:
            node.decorator_list = []
        namespace = {"torch": torch, "prod": __import__("math").prod,
                     "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        with FakeTensorMode():
            gradient = torch.empty(1031, device="cuda", dtype=torch.bfloat16)
            saved = torch.empty_like(gradient)._neg_view()
            with self.assertRaisesRegex(ValueError, "residuals require resolved"):
                namespace["backward"](gradient, saved, shape=gradient.shape, group_size=1024)


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA execution")
class StorageExpIndependentOracleCUDA(unittest.TestCase):
    def test_pre_rounding_codec_and_immutable_exact_input_oracle(self):
        from deepopt.opt.backends.triton.activation import storage_exp
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        torch.manual_seed(5403)
        ordinary = torch.randn(2, 3, 517, device="cuda", dtype=torch.bfloat16)
        sources = (ordinary, ordinary.transpose(0, 1), ordinary[:1].expand(2, 3, 517),
                   torch.tensor(0.5, device="cuda", dtype=torch.bfloat16))
        for source in sources:
            coefficient = source.float().exp()
            gradient = torch.randn_like(source)
            for packed in (False, True):
                for bits in (1, 2, 4, 8):
                    with self.subTest(shape=source.shape, strides=source.stride(), packed=packed, bits=bits):
                        # Independent pre-output-rounding FP32 coefficient and
                        # public codec oracle, never decode the helper's own
                        # payload as the expected compression recipe.
                        expected_q = quantize_pack_activation(coefficient.reshape(1, -1), 1024, bits=bits)
                        output, *saved = storage_exp.forward(source, packed=packed, bits=bits, group_size=1024)
                        torch.testing.assert_close(output, coefficient.bfloat16())
                        if packed:
                            torch.testing.assert_close(saved[0], expected_q.data, rtol=0, atol=0)
                            torch.testing.assert_close(saved[1], expected_q.step)
                            torch.testing.assert_close(saved[2], expected_q.minimum)
                            expected_coefficient = dequantize_unpack_activation(
                                expected_q, dtype=torch.float32).reshape_as(source)
                        else:
                            torch.testing.assert_close(saved[0], source, rtol=0, atol=0)
                            expected_coefficient = coefficient
                        dx = storage_exp.backward(gradient, *saved, shape=tuple(source.shape),
                                                  packed=packed, bits=bits, group_size=1024)
                        torch.testing.assert_close(dx, (gradient.float() * expected_coefficient).bfloat16(),
                                                   rtol=2**-7, atol=1e-4)
        mutable = ordinary.clone()
        _, snapshot = storage_exp.forward(mutable, packed=False, group_size=1024)
        expected = mutable.float().exp()
        mutable.add_(1)
        dx = storage_exp.backward(torch.ones_like(mutable), snapshot, shape=mutable.shape,
                                  packed=False, group_size=1024)
        torch.testing.assert_close(dx, expected.bfloat16())


if __name__ == "__main__":
    unittest.main()
