"""Measured norm rollout retains the unmeasured precision/layout fallbacks."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends._shared.normalization import measured_layer_variant, measured_group_forward


class MeasuredNormTests(unittest.TestCase):
    def test_group_norm_uses_the_realized_domain_group_and_native_output_dtype(self):
        geometry = SimpleNamespace(kind="group", shape=(16, 128, 7, 7), groups=32, elements=196)
        comp = {"enabled": True, "bits": 2, "group_size": 196}
        with patch("deepopt.opt.backends._shared.normalization.measured_norm_device", return_value=True):
            self.assertTrue(measured_group_forward(geometry, torch.bfloat16, torch.float32, comp, False, torch.device("cuda:0")))
            self.assertFalse(measured_group_forward(geometry, torch.bfloat16, torch.bfloat16, comp, False, torch.device("cuda:0")))
            self.assertFalse(measured_group_forward(geometry, torch.bfloat16, torch.float32,
                {**comp, "group_size": 128}, False, torch.device("cuda:0")))

    def test_measured_domains_and_precision_guards(self):
        device = torch.device("cuda:0")
        packed = SimpleNamespace(bits=2, group_size=512)
        with patch("deepopt.opt.backends._shared.normalization.measured_norm_device", return_value=True):
            self.assertEqual(measured_layer_variant(100864, 768, torch.bfloat16, packed, False, device), "layer_tiled")
            self.assertEqual(measured_layer_variant(6304, 3072, torch.bfloat16, packed, False, device), "layer_fused")
            self.assertEqual(measured_layer_variant(6304, 3072, torch.bfloat16, None, False, device), "layer_tiled")
            self.assertIsNone(measured_layer_variant(6304, 768, torch.bfloat16, packed, False, device))
            self.assertIsNone(measured_layer_variant(100864, 768, torch.float32, packed, False, device))
            self.assertIsNone(measured_layer_variant(100864, 768, torch.bfloat16, packed, True, device))
            self.assertIsNone(measured_layer_variant(100864, 768, torch.bfloat16,
                              SimpleNamespace(bits=8, group_size=128), False, device))
        with patch("deepopt.opt.backends._shared.normalization.measured_norm_device", return_value=False):
            self.assertIsNone(measured_layer_variant(100864, 768, torch.bfloat16, packed, False, device))
