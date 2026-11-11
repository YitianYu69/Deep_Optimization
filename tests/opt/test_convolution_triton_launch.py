"""CPU contracts for pointer-free implicit-convolution launch metadata."""

import unittest
from dataclasses import replace

import torch

from deepopt.opt.backends.triton.convolution import execution_mode
from deepopt.opt.backends.triton.convolution.launch import (
    geometry_arguments, launch_options, operand_arguments, weight_splits, needs_int64,
)
from deepopt.opt.nn.convolution import ConvGeometry
from deepopt.opt.quantization.tensor import QuantizedTensor


class ConvolutionLaunchTests(unittest.TestCase):
    def test_rank_one_padding_does_not_create_spurious_spatial_axes(self):
        geometry = ConvGeometry.from_tensors(
            torch.empty(2, 6, 13), torch.empty(8, 3, 3), stride=2, padding=2, dilation=2, groups=2)
        arguments = geometry_arguments(geometry)
        self.assertEqual(tuple(arguments[k] for k in ("ID", "IH", "IW")), (1, 1, 13))
        self.assertEqual(tuple(arguments[k] for k in ("OD", "OH", "OW")), (1, 1, 7))
        self.assertEqual(tuple(arguments[k] for k in ("KD", "KH", "KW")), (1, 1, 3))
        self.assertEqual(tuple(arguments[k] for k in ("PD", "PH", "PW")), (0, 0, 2))
        self.assertEqual(tuple(arguments[k] for k in ("DD", "DH", "DW")), (1, 1, 2))
        self.assertEqual((arguments["CI"], arguments["CO"], arguments["GROUPS"]), (3, 4, 2))

    def test_rank_three_keeps_distinct_axis_geometry(self):
        geometry = ConvGeometry.from_tensors(
            torch.empty(2, 4, 5, 7, 9), torch.empty(6, 2, 2, 3, 4),
            stride=(1, 2, 3), padding=(2, 1, 0), dilation=(3, 2, 1), groups=2)
        arguments = geometry_arguments(geometry)
        for names, expected in (
            (("ID", "IH", "IW"), (5, 7, 9)),
            (("KD", "KH", "KW"), (2, 3, 4)),
            (("SD", "SH", "SW"), (1, 2, 3)),
            (("PD", "PH", "PW"), (2, 1, 0)),
            (("DD", "DH", "DW"), (3, 2, 1)),
        ):
            self.assertEqual(tuple(arguments[k] for k in names), expected)

    def test_structural_padding_bounds_differ_from_physical_source_strides(self):
        geometry = replace(ConvGeometry.from_tensors(
            torch.empty(2, 4, 6, 8), torch.empty(6, 2, 2, 2), padding=1, groups=2),
            synthetic_padding_right=(1, 1))
        arguments = geometry_arguments(geometry)
        self.assertEqual(tuple(arguments[k] for k in ("ID", "IH", "IW")), (1, 6, 8))
        self.assertEqual(tuple(arguments[k] for k in ("VALID_I0", "VALID_I1", "VALID_I2")), (1, 5, 7))
        self.assertEqual(geometry.input_volume, 48)

    def test_gather_metadata_preserves_both_data_and_scale_strides(self):
        data = torch.empty_strided((4, 6), (1, 4), dtype=torch.float8_e4m3fn)
        scales = torch.ones(2, 2).T
        operand = QuantizedTensor(data, scales, (2, 3))
        arguments = operand_arguments("A", operand)
        self.assertEqual((arguments["A_RS"], arguments["A_CS"]), (1, 4))
        self.assertEqual((arguments["A_SRS"], arguments["A_SCS"]), (1, 2))
        self.assertEqual((arguments["A_BR"], arguments["A_BC"]), (2, 3))

    def test_tensorwise_scales_use_one_source_block(self):
        operand = QuantizedTensor(torch.empty(3, 5, dtype=torch.float8_e5m2), torch.ones(1, 1))
        arguments = operand_arguments("B", operand)
        self.assertEqual((arguments["B_BR"], arguments["B_BC"]), (3, 5))

    def test_block_scales_keep_native_fp8_arithmetic(self):
        data = torch.empty(3, 5, dtype=torch.float8_e4m3fn)
        tensorwise = QuantizedTensor(data, torch.ones(1, 1))
        blocked = QuantizedTensor(data, torch.ones(2, 3), (2, 2))
        self.assertEqual(execution_mode(tensorwise, tensorwise), "native_fp8")
        self.assertEqual(execution_mode(tensorwise, blocked), "native_fp8")
        self.assertEqual(execution_mode(blocked, tensorwise), "native_fp8")
        self.assertEqual(execution_mode(blocked, blocked), "native_fp8")

    def test_pipeline_options_reject_nonintegral_and_out_of_range_values(self):
        for invalid in (True, False, 0, 6, 2.0, "3", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                launch_options(invalid)
        for stages in (1, 2, 3, 4, 5):
            self.assertEqual(launch_options(stages)["num_stages"], stages)
        self.assertGreater(launch_options()["BLOCK_K"], launch_options(tensorwise=False)["BLOCK_K"])

    def test_tensorwise_tiles_qualify_for_native_hopper_fp8_wgmma(self):
        for tensorwise in (True, False):
            options = launch_options(tensorwise=tensorwise)
            self.assertEqual(options["BLOCK_M"] % 64, 0)
            self.assertEqual(options["BLOCK_N"] % 8, 0)
            self.assertGreaterEqual(options["BLOCK_K"], 32)
            self.assertEqual(options["num_warps"] % 4, 0)

    def test_split_reduction_is_bounded_by_weight_storage(self):
        for rows, cols, groups in ((4, 27, 1), (16, 144, 4), (1024, 1024, 1)):
            splits = weight_splits(rows, cols, 100_000, groups)
            self.assertGreaterEqual(splits, 1)
            self.assertLessEqual(splits, 16)
            if splits > 1:
                self.assertLessEqual(splits * rows * cols * groups * 4, 32 * 1024 * 1024)
        self.assertGreater(weight_splits(4, 27, 100_000, 1), 1)
        self.assertEqual(weight_splits(4, 27, 100, 1), 1)
        self.assertEqual(weight_splits(1024, 1024, 100_000, 1), 1)

    def test_wide_indices_cover_strided_payload_metadata_and_bias(self):
        geometry = ConvGeometry.from_tensors(torch.empty(1, 2, 5), torch.empty(3, 2, 3))
        normal = QuantizedTensor(torch.empty(5, 2, dtype=torch.float8_e4m3fn, device="meta"),
                                 torch.ones(1, 1, device="meta"))
        strided = QuantizedTensor(torch.empty_strided((5, 2), (1 << 30, 1),
                                                     dtype=torch.float8_e4m3fn, device="meta"),
                                  torch.ones(1, 1, device="meta"))
        scale_strided = QuantizedTensor(normal.data,
                                        torch.empty_strided((3, 1), (1 << 30, 1), device="meta"), (2, 2))
        bias = torch.empty_strided((3,), (1 << 30,), device="meta")
        self.assertFalse(needs_int64(normal, normal, geometry))
        self.assertTrue(needs_int64(strided, normal, geometry))
        self.assertTrue(needs_int64(normal, scale_strided, geometry))
        self.assertTrue(needs_int64(normal, normal, geometry, extra_tensors=(bias,)))

    def test_wide_indices_cover_large_spatial_products_and_dilation(self):
        operand = QuantizedTensor(torch.empty(2, 2, dtype=torch.float8_e4m3fn, device="meta"),
                                  torch.ones(1, 1, device="meta"))
        large = ConvGeometry.from_tensors(torch.empty(1, 2, 32768, 32768, device="meta"),
                                         torch.empty(2, 2, 1, 1, device="meta"))
        self.assertTrue(needs_int64(operand, operand, large))
        dilated = ConvGeometry.from_tensors(torch.empty(1, 1, 3, device="meta"),
                                           torch.empty(1, 1, 2, device="meta"),
                                           padding=1 << 30, dilation=1 << 31)
        self.assertTrue(needs_int64(operand, operand, dilated))


if __name__ == "__main__":
    unittest.main()
