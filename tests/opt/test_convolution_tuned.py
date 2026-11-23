"""Tuned convolution indexing, launch ownership and bounded search contracts."""

from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends._shared import convolution as policy
from deepopt.opt.backends.reference import convolution as oracle
from deepopt.opt.backends.triton.convolution.launch import (
    geometry_arguments, operand_arguments, launch_options, weight_split_options,
)
from deepopt.opt.nn.convolution import ConvGeometry, source_matrix
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor
from tests.opt.convolution_kernel_support import Pointer, kernel_namespace


class TunedConvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        policy._choices.clear()
        torch.manual_seed(1721)

    def test_grouped_m_visits_every_tile_once_with_short_final_groups(self):
        language, kernels = kernel_namespace()
        for m, n, groups, group_m in product((1, 65, 319), (1, 33, 95), (1, 3), (1, 4, 8)):
            tm, tn = (m + 63) // 64, (n + 31) // 32
            observed = []
            for pid in range(tm * tn * groups):
                language.programs = (pid, 0, 0)
                observed.append(tuple(map(int, kernels['_tile_coordinates'](m, n, 64, 32, group_m))))
            self.assertEqual(len(set(observed)), len(observed))
            self.assertEqual(set(observed), set(product(range(groups), range(tm), range(tn))))

    def test_tap_major_and_grouped_m_real_kernel_bodies_match_decoded_oracle(self):
        language, kernels = kernel_namespace()
        cases = (
            ((2, 6, 19), (10, 3, 3), dict(stride=2, padding=2, dilation=2, groups=2)),
            ((2, 5, 9, 11), (7, 5, 3, 2), dict(stride=(2, 1), padding=1)),
            ((1, 4, 5, 7, 6), (6, 4, 2, 3, 2), dict(stride=(1, 2, 1), padding=1, dilation=(1, 1, 2))),
        )
        for index, (shape, weight_shape, options) in enumerate(cases):
            x, weight = torch.randn(shape), torch.randn(weight_shape)
            geometry = ConvGeometry.from_tensors(x, weight, **options)
            if index == 1:
                geometry = replace(geometry, synthetic_padding_right=(1, 1))
            gradient = torch.randn(geometry.output_shape)
            qx, qw, qg = [quantize_fp8(value) for value in
                          (source_matrix(x), weight.reshape(weight_shape[0], -1), source_matrix(gradient))]
            # Non-row-major source strides must remain valid in tap indexing.
            qx = QuantizedTensor(qx.data.T.contiguous().T, qx.scale_inv)
            for index64 in (False, True):
                def arguments(a, b):
                    return {**geometry_arguments(geometry), **operand_arguments('A', a),
                            **operand_arguments('B', b), 'TENSORWISE':True, 'INDEX64':index64,
                            'BLOCK_M':64, 'BLOCK_N':32, 'BLOCK_K':32, 'GROUP_M':4}
                for name, a, b, expected in (
                    ('_forward_kernel', qx, qw, oracle.forward(qx, qw, geometry)),
                    ('_input_gradient_kernel', qg, qw, oracle.input_gradient(qg, qw, geometry)),
                ):
                    with self.subTest(shape=shape, kernel=name, index64=index64):
                        # The launcher allocates a canonical contiguous matrix;
                        # an oracle view can be column-major when batch == 1.
                        output = torch.full(expected.shape, float('nan'), dtype=expected.dtype)
                        channels = geometry.out_channels_per_group if name == '_forward_kernel' else geometry.in_channels_per_group
                        programs = (output.shape[0] + 63) // 64 * ((channels + 31) // 32) * geometry.groups
                        tensors = [a.data, b.data, a.scale_inv, b.scale_inv, output]
                        if name == '_forward_kernel':
                            tensors.append(output)
                        for pid in range(programs):
                            language.programs = (pid, 0, 0)
                            kernels[name](*(Pointer(t) for t in tensors), **arguments(a,b),
                                          TAP_MAJOR=True, HAS_BIAS=False, BIAS_STRIDE=1)
                        torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-5)
                for splits in (1, 3):
                    output = torch.full((splits, *geometry.weight_matrix_shape), float('nan'))
                    programs = ((geometry.out_channels_per_group + 63) // 64
                                * ((geometry.weight_matrix_shape[1] + 31) // 32) * geometry.groups)
                    for pid, split in product(range(programs), range(splits)):
                        language.programs = (pid, split, 0)
                        kernels['_weight_gradient_kernel'](*(Pointer(t) for t in
                            (qg.data, qx.data, qg.scale_inv, qx.scale_inv, output)),
                            **arguments(qg,qx), CHANNEL_FAST=True, SPLIT_K=splits)
                    torch.testing.assert_close(output.sum(0), oracle.weight_gradient(qx,qg,geometry), rtol=3e-5, atol=3e-5)

    def test_expanded_search_is_opt_in_role_specific_and_respects_stages(self):
        geometry = ConvGeometry.from_tensors(torch.empty(2,64,17,19), torch.empty(128,64,3,3), padding=1)
        baseline = policy.launch_candidates({'TENSORWISE':True, **launch_options()}, geometry, 'forward')
        self.assertLessEqual(len(baseline), 6)
        self.assertTrue(all(not c['TAP_MAJOR'] and c['GROUP_M'] == 1 for c in baseline))
        with policy.convolution_kernel_scope('tuned'):
            role_shapes = []
            for role in ('forward', 'dx', 'dw'):
                choices = policy.launch_candidates({'TENSORWISE':True, **launch_options()}, geometry, role, num_stages='auto')
                self.assertLessEqual(len(choices), 18)
                self.assertTrue(any(c['GROUP_M'] > 1 for c in choices))
                self.assertTrue(any(c['num_stages'] > 2 for c in choices))
                self.assertTrue(all(c['BLOCK_M'] % 64 == 0 for c in choices))
                self.assertEqual(any(c.get('TAP_MAJOR') for c in choices), role != 'dw')
                role_shapes.append({(c['BLOCK_M'], c['BLOCK_N'], c['BLOCK_K']) for c in choices})
                explicit = policy.launch_candidates({'TENSORWISE':True, **launch_options(3)}, geometry, role, num_stages=3)
                self.assertTrue(all(c['num_stages'] == 3 for c in explicit))
            self.assertNotEqual(role_shapes[0], role_shapes[2])
            blocked = policy.launch_candidates({'TENSORWISE':False, **launch_options()}, geometry, 'forward', num_stages='auto')
            self.assertEqual(len(blocked), 1)

    def test_split_trials_remain_within_workspace_budget_and_reduction(self):
        for rows, cols, groups, reduction in ((3,27,2,100_000), (1024,1024,1,100_000), (64,64,1,17)):
            for splits in weight_split_options(rows, cols, reduction, groups, 1):
                self.assertLessEqual(splits, (reduction + 63) // 64)
                if splits > 1:
                    self.assertLessEqual(splits * rows * cols * groups * 4, 32 * 1024 * 1024)

    def test_candidate_metadata_invalidates_cache_and_returned_choices_are_owned(self):
        first = [dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, num_warps=4, num_stages=2, GROUP_M=1)]
        changed = [dict(first[0]), {**first[0], 'GROUP_M':4, 'num_stages':3}]
        launch = lambda c: 10 / c['GROUP_M']
        chosen = policy.select_convolution_launch(('forward','cache'), first, launch, capturing=False, benchmark=lambda call:call())
        chosen['GROUP_M'] = 99
        chosen = policy.select_convolution_launch(('forward','cache'), changed, launch, capturing=False, benchmark=lambda call:call())
        self.assertEqual(chosen['GROUP_M'], 4)
        chosen['GROUP_M'] = 99
        self.assertEqual(policy.select_convolution_launch(('forward','cache'), changed, launch, capturing=True)['GROUP_M'], 4)

    def test_only_resource_rejection_falls_back_and_is_recorded(self):
        candidates = [dict(BLOCK_M=64, BLOCK_N=n, BLOCK_K=64, num_warps=4) for n in (32,64)]
        def launch(config):
            if config['BLOCK_N'] == 64:
                raise MemoryError('compiler resource exhaustion')
            return 1.0
        with patch.object(policy, '_resource_exhausted', side_effect=lambda error:isinstance(error,MemoryError)):
            with policy.record_convolution_launches() as records:
                selected = policy.select_convolution_launch(('dx','resource'), candidates, launch,
                    capturing=False, benchmark=lambda call:call())
            self.assertEqual(selected['BLOCK_N'], 32)
            self.assertEqual(records[0]['rejected_candidates'], [{'candidate':1, 'reason':'compiler_resources'}])
        with self.assertRaises(MemoryError):
            policy.select_convolution_launch(('dx','failure'), candidates, launch, capturing=False, benchmark=lambda call:call())
        with self.assertRaisesRegex(ValueError, 'finite'):
            policy.select_convolution_launch(('dx','nan'), candidates, lambda c:None,
                                            capturing=False, benchmark=lambda call:float('nan'))


if __name__ == '__main__':
    unittest.main()
