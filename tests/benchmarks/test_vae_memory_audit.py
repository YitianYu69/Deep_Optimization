"""Exact VAE target configuration and explicit metadata audit boundaries."""

from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from examples.vae import audit_activation_memory as audit
from examples.vae.mbconv_vae import MBConvVAE_V2


class VAEMemoryAuditTests(unittest.TestCase):
    def config(self):
        return audit.settings(Path(audit.__file__).with_name("activation_memory_local128.toml"))

    def test_exact_local128_config_preserves_training_recipe(self):
        raw = self.config()
        self.assertEqual(raw["training"]["batch_size"], 128)
        self.assertEqual(raw["training"]["optimizer"], "custom")
        self.assertEqual(raw["application"]["data"]["image_size"], 224)
        self.assertEqual(raw["runtime"]["activation_compression"]["group_size"], 1024)
        self.assertEqual(raw["schedules"]["beta"]["end"], .000001)
        self.assertEqual(raw["application"]["data"]["train_dir"], "")
        self.assertFalse(raw["runtime"]["compile"]["enabled"])

    def test_fake_initialization_restores_method_and_rejects_real_parameters(self):
        original = MBConvVAE_V2.init_weights
        with self.assertRaisesRegex(RuntimeError, "strict fake parameters"):
            with audit._fake_initialization(MBConvVAE_V2):
                MBConvVAE_V2.init_weights(None, torch.nn.Linear(2, 3))
        self.assertIs(MBConvVAE_V2.init_weights, original)
        with FakeTensorMode(allow_fallback_kernels=False), audit._fake_initialization(MBConvVAE_V2):
            MBConvVAE_V2.init_weights(None, torch.nn.Linear(2, 3))
        self.assertIs(MBConvVAE_V2.init_weights, original)

    def test_help_does_not_import_torch_or_require_custom_optimizer(self):
        result = subprocess.run([sys.executable, "-S", audit.__file__, "--help"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--optimizer-update", result.stdout)
        self.assertIn("cuda-groupnorm-emulated", result.stdout)


class VAEMemoryAuditPolicyTests(unittest.TestCase):
    def test_policy_aware_groupnorm_emulation_table_is_explicit(self):
        self.assertEqual(audit.emulated_groupnorm_dtype("torch_amp"), torch.float32)
        self.assertEqual(audit.emulated_groupnorm_dtype({"name": "bf16_storage", "version": 1}), torch.bfloat16)
        with self.assertRaises(ValueError):
            audit.emulated_groupnorm_dtype("invalid")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required: metadata emulation is not GPU dtype evidence")
class VAEMemoryAuditPolicyCUDATests(unittest.TestCase):
    def test_emulation_table_matches_actual_cuda_groupnorm_for_both_policies(self):
        import deepopt
        from deepopt.opt.policy import prepare_fp8_model

        for policy in ("torch_amp", "bf16_storage"):
            for compressed in (False, True):
                with self.subTest(policy=policy, compressed=compressed):
                    config = {"runtime": {"device": "cuda", "precision": "bf16", "bf16_policy": policy,
                        "fp8": {"enabled": False}, "activation_compression": {
                            "enabled": compressed, "bits": 2, "group_size": 1024,
                            "norm_xhat_recenter": True}}}
                    model = prepare_fp8_model(torch.nn.GroupNorm(2, 8).cuda(), config=config)[0]
                    x = torch.randn(2, 8, 17, 19, device="cuda", requires_grad=True)
                    with deepopt.autocast("cuda", dtype=torch.bfloat16, config=config):
                        y = model(x)
                    self.assertEqual(y.dtype, audit.emulated_groupnorm_dtype(policy))
                    y.float().square().mean().backward()
                    self.assertEqual(x.grad.dtype, torch.float32)
                    self.assertTrue(all(parameter.grad.dtype == torch.float32 for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
