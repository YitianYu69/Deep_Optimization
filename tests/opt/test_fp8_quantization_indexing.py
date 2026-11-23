"""Execute FP8 kernel address arithmetic without allocating multi-GB tensors.

These CPU checks interpret the installed JIT bodies. They verify addresses and
masks, but do not compile Triton or establish real CUDA correctness.
"""

import ast
from importlib.resources import files
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.opt.test_relu_kernels import CPUCast, ReLULanguage, TypedPointer


class VirtualPointer:
    def __init__(self, length, dtype=torch.float32, *, offset=0, accesses=None):
        self.length, self.offset = length, offset
        self.dtype = SimpleNamespace(element_ty=dtype)
        self.accesses = [] if accesses is None else accesses

    def __add__(self, offset):
        return VirtualPointer(self.length, self.dtype.element_ty,
                              offset=self.offset + offset, accesses=self.accesses)


class AddressLanguage(ReLULanguage):
    def program_id(self, axis):
        # Triton promotes scalar/vector integer arithmetic by dtype. A torch
        # rank-zero integer instead yields to the vector's dtype, so use one
        # lane to preserve Triton's promotion when a program ID becomes int64.
        return torch.tensor([self.programs[axis]], dtype=torch.int32)

    @staticmethod
    def cast(value, dtype):
        return torch.as_tensor(value).to(dtype)

    @staticmethod
    def checked(pointer, mask, operation):
        indices, valid = torch.broadcast_tensors(
            torch.as_tensor(pointer.offset).long(), torch.as_tensor(mask))
        selected = indices[valid]
        if ((selected < 0) | (selected >= pointer.length)).any():
            raise AssertionError(f"unmasked out-of-bounds FP8 {operation}")
        pointer.accesses.append((operation, selected))
        return indices, valid

    @staticmethod
    def load(pointer, mask=True, other=0.0):
        if not isinstance(pointer, VirtualPointer):
            return ReLULanguage.load(pointer, mask, other)
        indices, valid = AddressLanguage.checked(pointer, mask, "load")
        return torch.where(valid, torch.ones(indices.shape), other).to(pointer.dtype.element_ty)

    @staticmethod
    def store(pointer, values, mask=True):
        if not isinstance(pointer, VirtualPointer):
            return ReLULanguage.store(pointer, values, mask)
        AddressLanguage.checked(pointer, mask, "store")


class FP8QuantizationIndexingTests(unittest.TestCase):
    def setUp(self):
        self.tl = AddressLanguage()
        self.k = {"tl": self.tl}
        tree = ast.parse(files("deepopt.opt.backends.triton").joinpath("quantization/fp8.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)]
        for node in functions:
            node.decorator_list = []
        module = CPUCast().visit(ast.Module(body=functions, type_ignores=[]))
        exec(compile(ast.fix_missing_locations(module), "quantization/fp8.py", "exec"), self.k)

    def run_program(self, name, pid, *args):
        self.tl.programs = (pid, 0, 0)
        args = tuple(TypedPointer(value) if isinstance(value, torch.Tensor) and value.ndim else value
                     for value in args)
        self.k[name](*args)

    def assert_addresses(self, pointer, start, end):
        addresses = pointer.accesses[-1][1]
        self.assertGreater(addresses.numel(), 0)
        self.assertEqual(int(addresses.min()), start)
        self.assertEqual(int(addresses.max()), end)

    def test_vae_tensorwise_amax_on_both_sides_of_signed_int32_limit(self):
        # encoder.3.body.2 for the VAE at batch=64, channels=768, size=224.
        m, n, block = 64 * 224 * 224, 768, 4096
        count = (m * n + block - 1) // block
        for pid in (0, (2**31 // block) - 1, 2**31 // block, count - 1):
            with self.subTest(pid=pid):
                source, partials = VirtualPointer(m * n), VirtualPointer(count)
                self.run_program("_partial_amax", pid, source, partials, VirtualPointer(1),
                                 torch.tensor(m, dtype=torch.int32), torch.tensor(n, dtype=torch.int32),
                                 n, 1, m, n, 1, count, block, True, True, False)
                self.assert_addresses(source, pid * block, min((pid + 1) * block, m * n) - 1)
                self.assert_addresses(partials, pid, pid)

    def test_small_strided_input_can_have_a_storage_span_above_int32(self):
        for stride_m, stride_n in ((2**30, 2), (1, 2**30)):
            with self.subTest(strides=(stride_m, stride_n)):
                m, n = 3, 7
                length = (m - 1) * stride_m + (n - 1) * stride_n + 1
                source = VirtualPointer(length)
                output, inverse = VirtualPointer(m * n, torch.float8_e4m3fn), VirtualPointer(1)
                self.run_program("_fused_blocks", 0, source, output, inverse, VirtualPointer(1),
                                 m, n, stride_m, stride_n, 1, 1, m, n, 32, 1,
                                 448.0, False, True, False)
                self.assert_addresses(source, 0, length - 1)
                self.assert_addresses(output, 0, m * n - 1)

    def test_blockwise_fused_conversion_at_large_flat_offsets(self):
        m, n, br, bc = 64 * 224 * 224, 768, 1, 256
        blocks, groups = m * (n // bc), 4
        pid = blocks // groups - 1
        source, output = VirtualPointer(m * n), VirtualPointer(m * n, torch.float8_e4m3fn)
        self.run_program("_fused_blocks", pid, source, output, VirtualPointer(blocks), VirtualPointer(pid + 1),
                         m, n, n, 1, n // bc, blocks, br, bc, br * bc, groups,
                         448.0, True, True, True)
        self.assert_addresses(source, m * n - groups * br * bc, m * n - 1)
        self.assert_addresses(output, m * n - groups * br * bc, m * n - 1)

    def test_cast_and_dequantization_preserve_large_offsets_and_tail_masks(self):
        m, n, block = 64 * 224 * 224 + 1, 769, 1024
        count = (m * n + block - 1) // block
        for pid in (2**31 // block, count - 1):
            with self.subTest(pid=pid):
                source, output = VirtualPointer(m * n), VirtualPointer(m * n, torch.float8_e4m3fn)
                self.run_program("_cast_scaled", pid, source, output, VirtualPointer(1), VirtualPointer(count),
                                 n, m * n, n, 1, m, n, 1, True, block, 448.0, True, True)
                self.assert_addresses(source, pid * block, min((pid + 1) * block, m * n) - 1)
                self.assert_addresses(output, pid * block, min((pid + 1) * block, m * n) - 1)
                restored = VirtualPointer(m * n)
                self.run_program("_dequantize", pid, output, VirtualPointer(1), restored,
                                 torch.tensor(m, dtype=torch.int32), torch.tensor(n, dtype=torch.int32),
                                 n, 1, 1, 1, m, n, True, True, block)
                self.assert_addresses(restored, pid * block, min((pid + 1) * block, m * n) - 1)

    def test_paired_orientation_cast_uses_wide_source_and_destination_addresses(self):
        m, n, block = 64 * 224 * 224, 768, 32
        tiles_n = n // block
        pid = (m // block) * tiles_n - 1
        source, normal = VirtualPointer(m * n), VirtualPointer(m * n, torch.float8_e4m3fn)
        transpose = VirtualPointer(m * n, torch.float8_e4m3fn)
        self.run_program("_cast_tensorwise_pair", pid, source, normal, transpose, VirtualPointer(1),
                         m, n, n, 1, m, n, tiles_n, 448.0, block, True)
        self.assert_addresses(source, (m - block) * n + n - block, m * n - 1)
        self.assert_addresses(normal, (m - block) * n + n - block, m * n - 1)
        self.assert_addresses(transpose, (n - block) * m + m - block, m * n - 1)

    def test_pair_tile_candidates_write_exact_values_and_all_padding(self):
        # Execute the actual JIT body for every candidate tile width. CUDA
        # tests separately check warp layouts and compiled descriptor replay.
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for strided in (False, True):
                values = torch.randn(65, 97)
                values[0, 0] = -0.0
                if strided:
                    values = values.T
                m, n = values.shape
                pm, pn = (m + 15) // 16 * 16, (n + 15) // 16 * 16
                maximum = torch.finfo(dtype).max
                quantized = (values * 2).clamp(-maximum, maximum).to(dtype)
                expected = torch.zeros(m, pn, dtype=dtype)
                expected[:, :n].copy_(quantized)
                expected_t = torch.zeros(n, pm, dtype=dtype)
                expected_t[:, :m].copy_(quantized.T)
                for block in (32, 64, 128):
                    for transpose_only in (False, True):
                        with self.subTest(dtype=dtype, strided=strided, block=block, transpose_only=transpose_only):
                            normal, transpose = torch.empty_like(expected), torch.empty_like(expected_t)
                            tiles_n = (n + block - 1) // block
                            for pid in range(((m + block - 1) // block) * tiles_n):
                                self.run_program("_cast_tensorwise_pair", pid, values, normal, transpose,
                                                 torch.tensor([2.0]), m, n, *values.stride(), pm, pn,
                                                 tiles_n, maximum, block, not transpose_only)
                            if not transpose_only:
                                self.assertTrue(torch.equal(normal.view(torch.uint8), expected.view(torch.uint8)))
                            self.assertTrue(torch.equal(transpose.view(torch.uint8), expected_t.view(torch.uint8)))

    def test_scale_reductions_use_wide_partial_buffer_offsets(self):
        pid, block = 2**21, 1024
        partials, reduced = VirtualPointer((pid + 1) * block), VirtualPointer(pid + 1)
        self.run_program("_reduce_amax", pid, partials, reduced, block, 1, block)
        self.assert_addresses(partials, pid * block, (pid + 1) * block - 1)
        scale, inverse = VirtualPointer(pid + 1), VirtualPointer(pid + 1)
        self.run_program("_block_scales", pid, partials, scale, inverse, block, block, 448.0)
        self.assert_addresses(partials, pid * block, (pid + 1) * block - 1)

    def test_small_strided_quantize_and_dequantize_numerics(self):
        x = torch.linspace(-224, 224, 17 * 9).reshape(17, 9).T
        m, n, block = *x.shape, 32
        count = (m * n + block - 1) // block
        partials, scale, inverse = torch.empty(count), torch.empty(1), torch.empty(1)
        for pid in range(count):
            self.run_program("_partial_amax", pid, x, partials, inverse, m, n, *x.stride(),
                             m, n, 1, count, block, False, True, False)
        self.run_program("_block_scales", 0, partials, scale, inverse, count,
                         1 << (count - 1).bit_length(), 448.0)
        output = torch.empty(m, n, dtype=torch.float8_e4m3fn)
        restored = torch.empty(m, n)
        for pid in range(count):
            self.run_program("_cast_scaled", pid, x, output, scale, inverse, n, m * n, *x.stride(),
                             m, n, 1, True, block, 448.0, False, False)
            self.run_program("_dequantize", pid, output, inverse, restored, m, n, n, 1, 1, 1,
                             m, n, True, True, block)
        expected = quantize_fp8(x)
        torch.testing.assert_close(output.float(), expected.data.float(), rtol=0, atol=0)
        torch.testing.assert_close(inverse, expected.scale_inv.reshape(1), rtol=2e-7, atol=0)
        torch.testing.assert_close(restored, expected.dequantize(), rtol=2e-7, atol=0)


if __name__ == "__main__":
    unittest.main()
