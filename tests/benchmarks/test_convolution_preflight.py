"""Strict convolution implementation checks respect native FP8 accumulation.

The exact-code fixture isolates addressing, scale application, and replay from
the reduced-precision accumulation used by Hopper FP8 Tensor Cores. General
random-data errors remain visible in a separate diagnostic.
"""

import math
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.nn.convolution import ConvGeometry
from tests.cuda import test_runtime_speed as preflight


class ConvolutionPreflightTests(unittest.TestCase):
    def geometry(self, *, groups=1):
        channels = 32 if groups == 1 else 5
        output_channels = 64 if groups == 1 else 5
        spatial=(9,11) if groups==1 else (23,29)
        return ConvGeometry.from_tensors(
            torch.empty(2, channels, *spatial),
            torch.empty(output_channels, channels // groups, 3, 3),
            padding=1, groups=groups,
        )

    def operands(self, geometry):
        torch.manual_seed(14)
        with patch("deepopt.opt.quantization.quantizer.quantize_fp8",
                   side_effect=AssertionError("Exact codes must not be requantized")):
            return preflight.exact_convolution_operands(geometry, "cpu")

    def test_exact_payloads_and_scales_cover_dense_and_depthwise_geometry(self):
        for groups in (1, 5):
            with self.subTest(groups=groups):
                geometry = self.geometry(groups=groups)
                values = self.operands(geometry)
                expected_shapes = (
                    (geometry.batch_size * geometry.input_volume, geometry.in_channels),
                    geometry.weight_matrix_shape,
                    (geometry.batch_size * geometry.output_volume, geometry.out_channels),
                )
                for value, shape, scale in zip(values, expected_shapes, (.5, 2., .25)):
                    self.assertEqual(tuple(value.data.shape), tuple(shape))
                    self.assertIn(value.data.dtype, (torch.float8_e4m3fn, torch.float8_e5m2))
                    self.assertIsNone(value.block_shape)
                    self.assertEqual(value.scale_inv.dtype, torch.float32)
                    self.assertEqual(tuple(value.scale_inv.shape), (1, 1))
                    self.assertEqual(value.scale_inv.item(), scale)
                    codes = value.data.float()
                    self.assertTrue(torch.isin(codes, torch.tensor([-1., 0., 1.])).all())
                    self.assertEqual(set(codes.unique().tolist()), {-1., 0., 1.})

    def test_all_direction_sums_are_exact_even_at_half_precision(self):
        geometry = self.geometry()
        values = self.operands(geometry)
        actual = preflight.convolution_cpu_reference(*values, geometry)
        bounds = (geometry.in_channels_per_group * geometry.kernel_volume,
                  geometry.out_channels_per_group * geometry.kernel_volume,
                  geometry.batch_size * geometry.output_volume)
        for role, result, scale, bound in zip(
                ("forward", "dx", "dw"), actual, (1., .5, .125), bounds):
            with self.subTest(role=role):
                self.assertLess(bound, 2048)
                self.assertTrue(torch.isfinite(result).all())
                unscaled = result / scale
                torch.testing.assert_close(unscaled, unscaled.round(), rtol=0, atol=0)
                self.assertLessEqual(unscaled.abs().max().item(), bound)
                torch.testing.assert_close(result.half().float(), result, rtol=0, atol=0)
        preflight.assert_convolution_close(
            actual, tuple(value.clone() for value in actual),
            variant="strict", policy="baseline", launches=[], exact=True,
        )

    def test_exact_fixture_rejects_geometry_outside_its_sum_bound(self):
        # Each case exceeds the exact fixture's bound in a different role.
        shapes = (
            ((2, 256, 9, 11), (64, 256, 3, 3)),
            ((2, 32, 9, 11), (256, 32, 3, 3)),
            ((32, 32, 9, 11), (64, 32, 3, 3)),
            ((1, 2048, 1, 1), (1, 2048, 1, 1)),
        )
        for input_shape, weight_shape in shapes:
            geometry = ConvGeometry.from_tensors(
                torch.empty(input_shape, device="meta"),
                torch.empty(weight_shape, device="meta"),
                padding=1 if weight_shape[-1] == 3 else 0,
            )
            with self.subTest(input_shape=input_shape, weight_shape=weight_shape), \
                    self.assertRaisesRegex(ValueError, "2048"):
                preflight.exact_convolution_operands(geometry, "cpu")
        below_bound = ConvGeometry.from_tensors(
            torch.empty(1, 2047, 1, 1, device="meta"),
            torch.empty(1, 2047, 1, 1, device="meta"),
        )
        self.assertEqual(self.operands(below_bound)[0].data.shape, (1, 2047))

    def test_operand_updates_keep_addresses_and_oracle_reads_current_values(self):
        geometry = self.geometry()
        values = self.operands(geometry)
        before_codes = [value.data.float().clone() for value in values]
        before_scales = [value.scale_inv.clone() for value in values]
        addresses = [(value.data.data_ptr(), value.scale_inv.data_ptr()) for value in values]
        before_reference = preflight.convolution_cpu_reference(*values, geometry)
        frozen_reference = [value.clone() for value in before_reference]

        preflight.advance_exact_operands(values)
        for value, codes, scale, pointers in zip(values, before_codes, before_scales, addresses):
            self.assertEqual((value.data.data_ptr(), value.scale_inv.data_ptr()), pointers)
            self.assertFalse(torch.equal(value.data.float(), codes))
            self.assertTrue(torch.isin(value.data.float(), torch.tensor([-1., 0., 1.])).all())
            torch.testing.assert_close(value.scale_inv, scale * .5, rtol=0, atol=0)
        after_reference = preflight.convolution_cpu_reference(*values, geometry)
        for before, frozen, after in zip(before_reference, frozen_reference, after_reference):
            torch.testing.assert_close(before, frozen, rtol=0, atol=0)
            self.assertFalse(torch.equal(before, after))
            torch.testing.assert_close(after.half().float(), after, rtol=0, atol=0)

    def test_exact_gate_rejects_small_errors_in_every_direction(self):
        expected = tuple(torch.zeros(2, 3) for _ in range(3))
        for index, role in enumerate(("forward", "dx", "dw")):
            actual = [value.clone() for value in expected]
            actual[index][0, 0] = 1e-5
            with self.subTest(role=role), self.assertRaisesRegex(AssertionError, role):
                preflight.assert_convolution_close(
                    actual, expected, variant="strict", policy="tuned",
                    launches=[{"tile": {"BLOCK_K": 64}}], exact=True,
                )

    def test_random_data_diagnostic_exposes_mismatches_without_claiming_a_pass(self):
        expected = tuple(torch.tensor([[1., -2.], [0., 4.]]) for _ in range(3))
        actual = [value.clone() for value in expected]
        actual[0][1, 0] = .0136503
        diagnostics = preflight.convolution_error_diagnostics(actual, expected)
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual([row["role"] for row in diagnostics], ["forward", "dx", "dw"])
        row = diagnostics[0]
        self.assertTrue(row["finite"])
        self.assertEqual(row["mismatched_elements"], 1)
        self.assertAlmostEqual(row["max_abs"], .0136503, places=7)
        self.assertAlmostEqual(row["relative_l2"], .0136503 / math.sqrt(21.), places=7)
        self.assertNotIn("passed", row)
        self.assertEqual(diagnostics[1]["mismatched_elements"], 0)
        self.assertEqual(diagnostics[2]["max_abs"], 0.)

    def test_native_scale_identity_distinguishes_all_operand_roles(self):
        unit = tuple(torch.tensor([[3., -9.], [21., -45.]]) for _ in range(3))
        scales = tuple(torch.tensor([[value]]) for value in (.07319, .14276, .37882))
        x, w, g = scales
        actual = ((unit[0] * x) * w, (unit[1] * g) * w, (unit[2] * g) * x)
        preflight.assert_convolution_scaling(
            actual, unit, scales, variant="strict", policy="tuned",
        )
        wrong = ((unit[0] * g) * w, (unit[1] * x) * w, (unit[2] * g) * w)
        for index, role in enumerate(("forward", "dx", "dw")):
            corrupted = list(actual)
            corrupted[index] = wrong[index]
            with self.subTest(role=role), self.assertRaisesRegex(AssertionError, role):
                preflight.assert_convolution_scaling(
                    corrupted, unit, scales, variant="strict", policy="tuned",
                )

    def test_nonfinite_values_remain_visible_in_diagnostic(self):
        expected = tuple(torch.ones(2, 3) for _ in range(3))
        actual = [value.clone() for value in expected]
        actual[1][0, 0] = float("nan")
        actual[2][0, 0] = float("inf")
        diagnostics = preflight.convolution_error_diagnostics(actual, expected)
        self.assertTrue(diagnostics[0]["finite"])
        for row in diagnostics[1:]:
            self.assertFalse(row["finite"])
            self.assertGreaterEqual(row["mismatched_elements"], 1)

    def test_saved_diagnostic_contains_current_codes_outputs_and_visible_errors(self):
        geometry=self.geometry()
        values=self.operands(geometry)
        expected=preflight.convolution_cpu_reference(*values,geometry)
        actual=tuple(value.clone() for value in expected)
        actual[0][0,0]+=.0136503
        with tempfile.TemporaryDirectory() as folder,\
             patch.object(preflight,'CONVOLUTION_DIAGNOSTICS',[]),\
             patch.object(preflight,'_CONVOLUTION_ARTIFACT_ROWS',{}),\
             patch.object(preflight,'CONVOLUTION_ARTIFACT_WRITTEN',False),\
             patch.object(preflight,'CONVOLUTION_ARTIFACT_PATH',Path(folder)/'operands.pt'),\
             redirect_stdout(StringIO()):
            row=preflight.record_native_convolution(actual,expected,values,geometry,
                variant='strict',policy='baseline',stage='random-0',launches=[])
            self.assertFalse(row['precision_gate'])
            self.assertTrue(preflight.CONVOLUTION_ARTIFACT_WRITTEN)
            saved=torch.load(Path(folder)/'operands.pt',weights_only=True)['strict-baseline-random-0']
        self.assertGreater(saved['diagnostics']['metrics'][0]['max_abs'],.01)
        for a,b in zip(saved['actual'],actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(saved['operands'],values):
            torch.testing.assert_close(a['data'].view(torch.uint8),b.data.view(torch.uint8),rtol=0,atol=0)
            torch.testing.assert_close(a['scale_inv'],b.scale_inv,rtol=0,atol=0)

    def test_focused_suite_loads_only_convolution_and_full_suite_covers_new_candidates(self):
        from benchmarks.analysis import benchmark_attention_preflight as runner
        root=Path(runner.__file__).parent
        focused,path,extra,module=runner.load_preflight_suite(root,convolution_only=True)
        self.assertEqual(focused.countTestCases(),3)
        self.assertEqual(path,extra)
        self.assertEqual(module.RUNTIME_SPEED_PREFLIGHT_REVISION,4)
        self.assertTrue(next(iter(focused)).id().endswith('test_dense_and_depthwise_tuning_and_changing_value_replay'))
        complete,_,_,_=runner.load_preflight_suite(root,runtime_speed=True)
        self.assertEqual(complete.countTestCases(),25)


if __name__ == "__main__":
    unittest.main()
