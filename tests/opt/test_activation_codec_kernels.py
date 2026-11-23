"""Execute codec JIT bodies and real launch selection on CPU, separately from CUDA compilation."""

import ast
from math import prod
from importlib.resources import files
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from tests.opt.test_relu_kernels import ReLULanguage, TypedPointer, CPUCast


class CodecLanguage(ReLULanguage):
    bfloat16,float16,float64=torch.bfloat16,torch.float16,torch.float64
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
            self.owner.launch(self.name, grid, *args,
                              **{k: v for k, v in kwargs.items() if k.startswith("SOURCE_")})
        return call


class ActivationCodecKernelTests(unittest.TestCase):
    def test_coalesced_spatial_launcher_rounds_before_minmax_and_preserves_groups(self):
        source=torch.randn(2,64,3,5)
        view=source.movedim(1,-1)
        for bits in (1,2,4,8):
            expected=quantize_pack_activation(view.bfloat16().reshape(-1,64),1024,bits=bits)
            actual=self.host["quantize_pack_affine"](view,1024,bits=bits,logical_shape=(30,64),source_dtype=torch.bfloat16)
            self.assertIn("_quantize_pack_spatial_coalesced",self.events)
            for a,b in zip(actual,(expected.data,expected.step,expected.minimum)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_nd_strided_reader_matches_materialized_payload_for_heads_broadcasts_and_tails(self):
        source = torch.randn(2, 7, 3, 5).permute(0, 2, 1, 3)
        for x in (source, source[:, :, ::2, :], source[:1].expand(3, -1, -1, -1)):
            shape = (x.numel() // x.shape[-1], x.shape[-1])
            for bits in (1, 2, 4, 8):
                for group in (1024, 1024, 1024):  # Preserve repeated cases under the qualification group policy.
                    expected = self.host["quantize_pack_affine"](x.reshape(shape), group, bits=bits)
                    actual = self.host["quantize_pack_affine"](x, group, bits=bits, logical_shape=shape)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_compact_decode_preserves_values_and_uses_requested_stride(self):
        source = torch.randn(67, 3)
        packed = quantize_pack_activation(source, 1024)
        for alignment in (1, 16):
            actual = self.host["dequantize_unpack_affine"](
                packed.data, packed.step, packed.minimum, packed.shape, 1024,
                row_alignment=alignment)
            torch.testing.assert_close(actual, dequantize_unpack_activation(packed), rtol=0, atol=0)
            self.assertEqual(actual.stride(0), 3 if alignment == 1 else 16)

    def test_batched_packing_preserves_payload_metadata_strides_and_group_tails(self):
        maximum = torch.finfo(torch.bfloat16).max
        sources = [torch.randn(33, 67).T, torch.full((3, 67), -2.5),
                   torch.tensor([[-maximum, 0, maximum]], dtype=torch.bfloat16).expand(3, 3)]
        for source in sources:
            for bits in (1, 2, 4, 8):
                for group, block in ((1024, None), (1024, None), (1024, None), (1024, (4, 256))):
                    expected = self.host["quantize_pack_affine"](source, group, block, bits=bits, implementation="grouped")
                    for variant in ("batched", "serial"):
                        for batch in (1, 2, 4, 8):
                            actual = self.host["quantize_pack_affine"](source, group, block, bits=bits,
                                implementation=variant, groups_per_program=batch)
                            for value, wanted in zip(actual, expected):
                                torch.testing.assert_close(value, wanted, rtol=0, atol=0)

    def test_measured_decode_tile_is_limited_to_the_measured_precision_and_hopper_shapes(self):
        select = self.host["_transpose_tiles"]
        options = dict(dtype=torch.float8_e4m3fn, bits=2, group_size=1024,
                       block_shape=None, device=torch.device("cuda:0"))
        with patch("deepopt.opt.backends._shared.cuda.device_metadata", return_value=(90, 132)):
            for width in (768, 3072):
                self.assertEqual(select(100864, width, **options), (64, 64))
            for key, value in (("dtype", torch.float8_e5m2), ("bits", 4), ("dtype", torch.bfloat16),
                               ("block_shape", (1, 1024)), ("device", torch.device("cpu"))):
                self.assertEqual(select(100864, 768, **{**options, key: value}), (32, 64))
            self.assertEqual(select(100865, 768, **options), (32, 64))
        with patch("deepopt.opt.backends._shared.cuda.device_metadata", return_value=(100, 148)):
            self.assertEqual(select(100864, 768, **options), (32, 64))

    def setUp(self):
        self.tl = CodecLanguage()
        self.k = {"tl": self.tl, "triton": SimpleNamespace(cdiv=self.tl.cdiv)}
        tree = ast.parse(files("deepopt.opt.backends.triton").joinpath("quantization/activation/affine.py").read_text())
        jit = [node for node in tree.body if isinstance(node, ast.FunctionDef) and any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "jit"
            for decorator in node.decorator_list)]
        for node in jit:
            node.decorator_list = []
        module = CPUCast().visit(ast.Module(body=jit, type_ignores=[]))
        exec(compile(ast.fix_missing_locations(module), "affine.py", "exec"), self.k)
        self.events = []
        self.host = {"torch": torch, "tl":self.tl, "prod":prod, "triton": SimpleNamespace(cdiv=self.tl.cdiv,
                    next_power_of_2=lambda v: 1 << (v - 1).bit_length())}
        self.host.update({node.name: Launch(self, node.name) for node in jit})
        host = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("_bits_consts", "_transpose_tiles", "quantize_pack_affine", "dequantize_unpack_affine")]
        for node in host:
            node.decorator_list = []  # CPU interpreter bypasses only the CUDA launch scope.
        exec(compile(ast.Module(body=host, type_ignores=[]), "affine_launch.py", "exec"), self.host)

    def launch(self, name, grid, *args, **kwargs):
        args = tuple(TypedPointer(value) if isinstance(value, torch.Tensor) else value for value in args)
        grid = tuple(grid) + (1,) * (2 - len(grid))
        for row in range(grid[0]):
            for col in range(grid[1]):
                self.tl.programs = (row, col, 0)
                self.k[name](*args, **kwargs)

    def test_tile_grouped_and_copy_match_reference_bits_scales_tails_and_output_alignment(self):
        torch.manual_seed(991)
        for shape, group, block in (((37, 65), 1024, None), ((65, 131), 1024, None),
                                    ((33, 67), 1024, (4, 256)), ((3, 5), 1024, None),
                                    ((1, 33), 1024, (1, 1024))):
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
            packed = quantize_pack_activation(source, 1024, bits=1)
            for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                actual = self.host["dequantize_unpack_affine"](
                    packed.data, packed.step, packed.minimum, packed.shape, 1024,
                    bits=1, dtype=dtype, transpose=True, implementation="tiled")
                expected = dequantize_unpack_activation(packed, dtype=dtype, transpose=True)
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)

    def test_prequantized_fp8_inputs_use_floating_padding_in_grouped_and_chunked_packers(self):
        torch.manual_seed(992)
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            source = torch.randn(37, 65).to(dtype).T
            for group, block in ((1024, None), (1024, None), (1024, (16, 64))):
                for bits in (1, 2, 4, 8):
                    actual = self.host["quantize_pack_affine"](source, group, block, bits=bits)
                    expected = quantize_pack_activation(source, group, block, bits=bits)
                    for value, reference in zip(actual, (expected.data, expected.step, expected.minimum)):
                        torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_launch_policy_uses_tiles_for_matrix_transposes_and_keeps_grouped_tiny_and_native_paths(self):
        for shape, transpose, expected in (((65, 131), True, "_dequantize_unpack_transpose"),
                                         ((1, 131), True, "_dequantize_unpack_groups"),
                                         ((65, 131), False, "_dequantize_unpack_groups")):
            packed = quantize_pack_activation(torch.randn(shape), 1024)
            self.events.clear()
            self.host["dequantize_unpack_affine"](packed.data, packed.step, packed.minimum, shape, 1024, transpose=transpose)
            self.assertEqual(self.events, [expected])


if __name__ == "__main__":
    unittest.main()
