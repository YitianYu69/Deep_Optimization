"""Bit-exact boolean storage contracts and CPU execution of installed JIT bodies."""

import ast
from importlib.resources import files
import subprocess
import sys
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt.opt.quantization.boolean import PackedBooleanMask, pack_boolean_mask, unpack_boolean_mask
from tests.opt.convolution_kernel_support import KernelLanguage, Pointer


def expected_words(mask):
    values = mask.reshape(-1).tolist()
    words = []
    for start in range(0, len(values), 32):
        word = sum(int(value) << bit for bit, value in enumerate(values[start:start + 32]))
        words.append(word if word < 1 << 31 else word - (1 << 32))
    return torch.tensor(words, dtype=torch.int32)


def masks():
    torch.manual_seed(193)
    for count in (0, 1, 2, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 4097):
        for kind in ("random", "false", "true"):
            yield (torch.randint(2, (count,), dtype=torch.bool) if kind == "random" else
                   torch.full((count,), kind == "true", dtype=torch.bool))
    yield torch.tensor(True)
    yield torch.tensor(False)
    yield torch.empty((2, 0, 5), dtype=torch.bool)
    source = torch.randint(2, (3, 7, 13), dtype=torch.bool)
    yield source.transpose(0, 2)
    yield source[1:, 1::2, 2::3]
    yield source[:, :1, :1].expand(3, 9, 17)
    yield source[0, 0, 0].expand(2, 3, 5, 7)
    yield source.unsqueeze(0).to(memory_format=torch.channels_last)


class BooleanMaskTests(unittest.TestCase):
    def test_roundtrip_matches_independent_words_for_shapes_strides_and_tails(self):
        for mask in masks():
            with self.subTest(shape=mask.shape, stride=mask.stride()):
                snapshot = mask.clone()
                packed = pack_boolean_mask(mask)
                restored = unpack_boolean_mask(packed)
                self.assertEqual(packed.shape, tuple(mask.shape))
                self.assertEqual(packed.data.dtype, torch.int32)
                self.assertEqual(packed.data.numel(), (mask.numel() + 31) // 32)
                self.assertTrue(packed.data.is_contiguous())
                self.assertTrue(restored.is_contiguous())
                self.assertEqual(restored.dtype, torch.bool)
                torch.testing.assert_close(packed.data, expected_words(mask), rtol=0, atol=0)
                torch.testing.assert_close(restored, mask, rtol=0, atol=0)
                torch.testing.assert_close(mask, snapshot, rtol=0, atol=0)

    def test_known_words_cover_signed_high_bit_and_ignore_unused_tail_bits(self):
        packed = PackedBooleanMask(torch.tensor([-2147483648, -1, -1], dtype=torch.int32), (65,))
        expected = torch.zeros(65, dtype=torch.bool)
        expected[31:] = True
        torch.testing.assert_close(unpack_boolean_mask(packed), expected, rtol=0, atol=0)
        source = torch.zeros(33, dtype=torch.bool)
        source[31:] = True
        self.assertEqual(pack_boolean_mask(source).data.tolist(), [-2147483648, 1])

    def test_payload_owns_storage_and_reconstruction_does_not_alias_it(self):
        source = torch.ones(64, dtype=torch.bool)
        packed = pack_boolean_mask(source)
        self.assertEqual(packed.data.numel() * packed.data.element_size(), source.numel() // 8)
        source.zero_()
        restored = unpack_boolean_mask(packed)
        self.assertTrue(restored.all().item())
        restored.zero_()
        self.assertTrue(unpack_boolean_mask(packed).all().item())

    def test_invalid_inputs_and_malformed_metadata_fail_explicitly(self):
        for value in (None, [True], torch.ones(3), torch.ones(3, dtype=torch.uint8)):
            with self.subTest(value=type(value)), self.assertRaises(TypeError):
                pack_boolean_mask(value)
        sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([True]), (3,), check_invariants=True)
        with self.assertRaises(ValueError):
            pack_boolean_mask(sparse)
        for shape in ([32], (-1,), (True,), (2.0,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                PackedBooleanMask(torch.empty(1, dtype=torch.int32), shape)
        for data, shape, error in (
            (torch.zeros(1), (32,), TypeError), (None, (32,), TypeError),
            (torch.empty(2, dtype=torch.int32), (32,), ValueError),
            (torch.empty(1, 1, dtype=torch.int32), (32,), ValueError),
            (torch.empty(4, dtype=torch.int32)[::2], (64,), ValueError),
        ):
            with self.subTest(shape=shape, error=error), self.assertRaises(error):
                PackedBooleanMask(data, shape)
        with self.assertRaises(TypeError):
            unpack_boolean_mask(torch.empty(1, dtype=torch.int32))

    def test_fake_and_meta_shapes_do_not_import_or_launch_cuda_kernels(self):
        # A fresh process also establishes that importing the CPU codec is lazy.
        code = """
import sys
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from deepopt.opt.quantization.boolean import pack_boolean_mask, unpack_boolean_mask
assert 'triton' not in sys.modules
for device in ('cpu', 'cuda:0'):
    with FakeTensorMode():
        x = torch.empty((5, 3, 7), dtype=torch.bool, device=device).transpose(0, 2)
        p = pack_boolean_mask(x)
        y = unpack_boolean_mask(p)
        assert p.data.shape == (4,)
        assert y.shape == x.shape and y.dtype == torch.bool and y.device == x.device
assert 'deepopt.opt.backends.triton.quantization.activation.boolean' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        for shape in ((), (0,), (2, 0, 4), (3, 11)):
            with self.subTest(shape=shape):
                source = torch.empty(shape, dtype=torch.bool, device="meta")
                restored = unpack_boolean_mask(pack_boolean_mask(source))
                self.assertEqual(tuple(restored.shape), shape)
                self.assertEqual(restored.dtype, torch.bool)
        with FakeTensorMode():
            source = torch.empty(0, dtype=torch.bool)
            self.assertEqual(unpack_boolean_mask(pack_boolean_mask(source)).numel(), 0)


class BooleanKernelLanguage(KernelLanguage):
    static_range = staticmethod(range)

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis, dtype=torch.int32)


class BooleanMaskKernelTests(unittest.TestCase):
    def test_installed_kernel_bodies_match_format_without_hidden_input_copies(self):
        language = BooleanKernelLanguage()
        namespace = {"tl": language}
        source = files("deepopt.opt.backends.triton.quantization.activation").joinpath("boolean.py")
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.decorator_list]
        for function in functions:
            function.decorator_list = []
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        for mask in masks():
            if not mask.numel():
                continue
            with self.subTest(shape=mask.shape, stride=mask.stride()):
                words = (mask.numel() + 31) // 32
                packed = torch.full((words,), 123, dtype=torch.int32)
                for pid in range((words + 3) // 4):
                    language.programs = (pid, 0, 0)
                    namespace["pack_boolean_mask_kernel"](
                        Pointer(mask), Pointer(packed), mask.numel(), words,
                        tuple(mask.shape), tuple(mask.stride()), mask.is_contiguous(), 4)
                torch.testing.assert_close(packed, expected_words(mask), rtol=0, atol=0)
                output = torch.empty(mask.shape, dtype=torch.bool)
                for pid in range((mask.numel() + 127) // 128):
                    language.programs = (pid, 0, 0)
                    namespace["unpack_boolean_mask_kernel"](Pointer(packed), Pointer(output), mask.numel(), 128)
                torch.testing.assert_close(output, mask, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
