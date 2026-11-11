"""Execute codec JIT bodies and real launch selection on CPU, separately from CUDA compilation."""

import ast
from importlib.resources import files
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from tests.opt.test_relu_kernels import ReLULanguage, TypedPointer, CPUCast


class CodecLanguage(ReLULanguage):
    floor = staticmethod(torch.floor)
    trans = staticmethod(torch.t)

    @staticmethod
    def min(value, axis=0):
        return value.amin(dim=axis)

    @staticmethod
    def load(pointer, mask=True, other=0.0):
        if pointer.storage.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and type(other) is int:
            raise AssertionError("Triton FP8 loads require floating padding")
        return ReLULanguage.load(pointer, mask, other)


class Launch:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            self.owner.events.append(self.name)
            self.owner.launch(self.name, grid, *args)
        return call


class ActivationCodecKernelTests(unittest.TestCase):
    def setUp(self):
        self.tl = CodecLanguage()
        self.k = {"tl": self.tl, "triton": SimpleNamespace(cdiv=self.tl.cdiv)}
        tree = ast.parse(files("deepopt.opt.backends.triton").joinpath("quantization/activation/affine.py").read_text())
        jit = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.decorator_list]
        for node in jit:
            node.decorator_list = []
        module = CPUCast().visit(ast.Module(body=jit, type_ignores=[]))
        exec(compile(ast.fix_missing_locations(module), "affine.py", "exec"), self.k)
        self.events = []
        self.host = {"torch": torch, "triton": SimpleNamespace(cdiv=self.tl.cdiv,
                    next_power_of_2=lambda v: 1 << (v - 1).bit_length())}
        self.host.update({node.name: Launch(self, node.name) for node in jit})
        host = [node for node in tree.body if isinstance(node, ast.FunctionDef) and not node.decorator_list
                and node.name in ("_bits_consts", "_transpose_tiles", "quantize_pack_affine", "dequantize_unpack_affine")]
        exec(compile(ast.Module(body=host, type_ignores=[]), "affine_launch.py", "exec"), self.host)

    def launch(self, name, grid, *args):
        args = tuple(TypedPointer(value) if isinstance(value, torch.Tensor) else value for value in args)
        grid = tuple(grid) + (1,) * (2 - len(grid))
        for row in range(grid[0]):
            for col in range(grid[1]):
                self.tl.programs = (row, col, 0)
                self.k[name](*args)

    def test_tile_grouped_and_copy_match_reference_bits_scales_tails_and_output_alignment(self):
        torch.manual_seed(991)
        for shape, group, block in (((37, 65), 17, None), ((65, 131), 1024, None),
                                    ((33, 67), 17, (4, 7)), ((3, 5), 20003, None),
                                    ((1, 33), 512, (1, 32))):
            source = torch.randn(shape, dtype=torch.bfloat16)
            for bits in (1, 2, 4, 8):
                packed = quantize_pack_activation(source, group, block, bits=bits)
                for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    expected = dequantize_unpack_activation(packed, dtype=dtype, transpose=True)
                    for implementation in ("grouped", "tiled", "copy", "auto"):
                        with self.subTest(shape=shape, bits=bits, dtype=dtype, implementation=implementation):
                            actual = self.host["dequantize_unpack_affine"](
                                packed.data, packed.step, packed.minimum, packed.shape, group, block,
                                bits=bits, dtype=dtype, transpose=True, implementation=implementation)
                            torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
                            storage_dtype = torch.int16 if dtype == torch.bfloat16 else torch.uint8
                            self.assertTrue(torch.equal(actual.view(storage_dtype), expected.view(storage_dtype)))
                            self.assertEqual(actual.stride(1), 1)
                            self.assertEqual(actual.stride(0) % 16, 0)

    def test_tiled_decode_preserves_overflow_binary_steps_and_constant_groups(self):
        maximum = torch.finfo(torch.bfloat16).max
        for source in (torch.tensor([[-maximum, 0, maximum]], dtype=torch.bfloat16).expand(33, 3),
                       torch.full((33, 65), -2.5), torch.zeros(33, 65)):
            packed = quantize_pack_activation(source, 17, bits=1)
            for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                actual = self.host["dequantize_unpack_affine"](
                    packed.data, packed.step, packed.minimum, packed.shape, 17,
                    bits=1, dtype=dtype, transpose=True, implementation="tiled")
                expected = dequantize_unpack_activation(packed, dtype=dtype, transpose=True)
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)

    def test_prequantized_fp8_inputs_use_floating_padding_in_grouped_and_chunked_packers(self):
        torch.manual_seed(992)
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            source = torch.randn(37, 65).to(dtype).T
            for group, block in ((17, None), (20003, None), (512, (16, 32))):
                for bits in (1, 2, 4, 8):
                    actual = self.host["quantize_pack_affine"](source, group, block, bits=bits)
                    expected = quantize_pack_activation(source, group, block, bits=bits)
                    for value, reference in zip(actual, (expected.data, expected.step, expected.minimum)):
                        torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_launch_policy_uses_tiles_for_matrix_transposes_and_keeps_grouped_tiny_and_native_paths(self):
        for shape, transpose, expected in (((65, 131), True, "_dequantize_unpack_transpose"),
                                         ((1, 131), True, "_dequantize_unpack_groups"),
                                         ((65, 131), False, "_dequantize_unpack_groups")):
            packed = quantize_pack_activation(torch.randn(shape), 512)
            self.events.clear()
            self.host["dequantize_unpack_affine"](packed.data, packed.step, packed.minimum, shape, transpose=transpose)
            self.assertEqual(self.events, [expected])


if __name__ == "__main__":
    unittest.main()
