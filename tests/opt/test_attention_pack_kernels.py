"""Interpret installed joint QKV JIT indexing/packing; GPU compilation is separate."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.backends._shared.activation_codec import activation_codec_scope, attention_packing
from deepopt.opt.quantization.activation import _quantize_pack_strided_activation, _quantize_pack_qkv_activation
from tests.opt.test_activation_kernels import Language
from tests.opt.test_relu_kernels import TypedPointer, CPUCast
from tests.opt.convolution_kernel_support import Pointer


class Choice:
    def __init__(self, predicate, a, b):
        self.predicate, self.a, self.b = predicate, a, b

    def __add__(self, offset):
        return Choice(self.predicate, self.a + offset, self.b + offset)


class JointLanguage(Language):
    @staticmethod
    def where(predicate, a, b):
        if isinstance(a, (Pointer, Choice)):
            return Choice(predicate, a, b)
        return Language.where(predicate, a, b)

    @staticmethod
    def load(pointer, mask=True, other=0):
        if isinstance(pointer, Choice):
            a = JointLanguage.load(pointer.a, mask & pointer.predicate, other)
            b = JointLanguage.load(pointer.b, mask & ~pointer.predicate, other)
            return torch.where(pointer.predicate, a, b)
        return Language.load(pointer, mask, other)


class AttentionPackKernelTests(unittest.TestCase):
    def test_joint_jit_matches_generic_payload_and_metadata_for_real_strides(self):
        tl = JointLanguage()
        namespace = {"tl": tl, "triton": SimpleNamespace(cdiv=tl.cdiv)}
        tree = ast.parse(files("deepopt.opt.backends.triton").joinpath("quantization/activation/affine.py").read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
            isinstance(d, ast.Attribute) and d.attr == "jit" for d in n.decorator_list)]
        for function in functions:
            function.decorator_list = []
        module = CPUCast().visit(ast.Module(body=functions, type_ignores=[]))
        exec(compile(ast.fix_missing_locations(module), "affine.py", "exec"), namespace)
        torch.manual_seed(141)
        source = torch.randn(2, 5, 3, 3, 8, dtype=torch.bfloat16)
        interleaved = source.unbind(2)
        for layout in (interleaved, tuple(x.contiguous() for x in interleaved),
                       tuple(x.transpose(1, 2).contiguous().transpose(1, 2) for x in interleaved)):
            shape, strides = tuple(layout[0].shape), tuple(layout[0].stride())
            rows, cols = prod(shape[:-1]), shape[-1]
            for bits in (1, 2, 4, 8):
                for group in (17, 64, 256, 1024):
                    for narrow in (False, True):
                        with self.subTest(strides=strides, bits=bits, group=group, narrow=narrow):
                            groups = (rows * cols + group - 1) // group
                            words = (group + 32 // bits - 1) // (32 // bits)
                            block = max(32 // bits, 1 << (group - 1).bit_length())
                            data = torch.empty(3, groups, words, dtype=torch.int32)
                            steps, minima = torch.empty(3, groups), torch.empty(3, groups)
                            pointers = tuple(TypedPointer(x) for x in (*layout, data, steps, minima))
                            for index in range(groups):
                                tl.programs = (index, 0, 0)
                                namespace["_quantize_pack_qkv_groups"](*pointers, rows, cols, group, groups,
                                    words, bits, block, shape, strides, narrow)
                            for index, value in enumerate(layout):
                                expected = _quantize_pack_strided_activation(value, group, bits=bits)
                                for actual, wanted in zip((data[index], steps[index], minima[index]),
                                                          (expected.data, expected.step, expected.minimum)):
                                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_cpu_fallback_is_exact_and_scope_restores(self):
        values = tuple(torch.randn(2, 5, 3, 8, dtype=torch.bfloat16) for _ in range(3))
        actual = _quantize_pack_qkv_activation(values, 17, bits=4)
        expected = tuple(_quantize_pack_strided_activation(x, 17, bits=4) for x in values)
        for a, b in zip(actual, expected):
            for x, y in zip((a.data, a.step, a.minimum), (b.data, b.step, b.minimum)):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(attention_packing(), "auto")
        with activation_codec_scope(attention_pack="joint"):
            self.assertEqual(attention_packing(), "joint")
        self.assertEqual(attention_packing(), "auto")
        with self.assertRaises(ValueError):
            with activation_codec_scope(attention_pack="joint", pointwise_tile=3):
                pass
        self.assertEqual(attention_packing(), "auto")
