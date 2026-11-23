"""Experimental allocation rewrites retain strict controls and honest reports."""

from dataclasses import replace
import unittest

import torch

from deepopt.compiler import BackendConfig
from deepopt.compiler.ir import FreshAllocationContract, TensorMeta
from tests.compiler.test_backward_add_donation import _compile, _Projections


class MemoryOptimizationContractTests(unittest.TestCase):
    def test_options_default_off_reject_nonbooleans_and_require_native_autodiff(self):
        default = BackendConfig.from_options({})
        for name in ('fuse_norm_cat_body', 'donate_fresh_backward_add'):
            self.assertIs(getattr(default, name), False)
            for value in (1, None, 'true'):
                with self.subTest(option=name, value=value), self.assertRaises(ValueError):
                    BackendConfig.from_options({name: value})
            for owner in ({'enable_aot_autodiff': False}, {'use_native_autodiff': False}):
                with self.subTest(option=name, owner=owner), self.assertRaisesRegex(ValueError, 'native AutoDiff'):
                    BackendConfig.from_options({name: True, **owner})
            self.assertIs(getattr(BackendConfig.from_options({name: True}), name), True)

    def test_fresh_contract_refuses_unknown_layout_saved_representation_and_higher_order(self):
        metadata = TensorMeta((2, 4, 3, 7), torch.float32, 'cpu', (84, 21, 7, 1), False)
        self.assertEqual(FreshAllocationContract(metadata, 'convolution.native_dx_db').meta, metadata)
        for changed in (replace(metadata, stride=None), replace(metadata, stride=(84, 1, 28, 4)),
                        replace(metadata, requires_grad=True), replace(metadata, representation='opaque'),
                        replace(metadata, storage_dtype=torch.uint8), replace(metadata, device=None)):
            with self.subTest(meta=changed), self.assertRaises(ValueError):
                FreshAllocationContract(changed, 'convolution.native_dx_db')
        with self.assertRaises(ValueError):
            FreshAllocationContract(metadata, 'convolution.native_dx_db', owner='saved_forward')

    def test_actual_donation_report_separates_reuse_from_kernel_fusion(self):
        x = torch.randn(2, 4, 3, 7, requires_grad=True)
        model, output, report, _ = _compile(_Projections(), x, True)
        torch.autograd.grad(output, (x, *model.parameters()), torch.randn_like(output))
        payload = report.as_dict()
        self.assertEqual(payload['memory_optimization_regions'], 1)
        self.assertEqual(payload['graphs'][0]['memory_optimization_regions'], 1)
        self.assertEqual(payload['graphs'][0]['fused_deepopt_regions'], 0)
        self.assertEqual(payload['runtime_fallbacks'], 0)


if __name__ == '__main__':
    unittest.main()
