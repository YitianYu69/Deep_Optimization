"""Owned activation compression configuration and selective linear dispatch."""

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import ActivationCompressionConfig, autocast
from deepopt._config_resolution import ResolvedConfig, resolve_config
from deepopt.config import (ConfigError, activation_compression_config_from_dict,
                            config_from_dict, load_config, reset_config)
from deepopt.engine.execution.configuration import prepare_execution_model
from deepopt.opt._autocast import _get_activation_compression
from deepopt.opt.nn.functional import do_linear, validate_fp8_plan
from deepopt.opt.policy import (fp8_plan, normalize_activation_compression_config,
                               prepare_fp8_model)


class ActivationCompressionConfigTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_configuration_import_and_normalization_are_torch_free(self):
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
        result = subprocess.run([sys.executable, "-c", """
import sys
from deepopt import ActivationCompressionConfig
from deepopt.opt.policy import normalize_activation_compression_config
assert normalize_activation_compression_config(True) == ActivationCompressionConfig(True, 512)
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
"""], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_disabled_and_partial_snapshot_provenance(self):
        self.assertEqual(normalize_activation_compression_config(), ActivationCompressionConfig())
        self.assertNotIn("activation_compression", resolve_config({"runtime": {"checks": False}})["runtime"])
        self.assertEqual(fp8_plan(True)["fp8_training"]["activation_compression"],
                         {"enabled": False, "group_size": 512, "bits": 2, "attention_bits": None, "norm_xhat_recenter": False})

    def test_validates_options_and_accepts_partial_groups(self):
        for group_size in (1, 15, 513):
            self.assertEqual(activation_compression_config_from_dict({"group_size": group_size}).group_size,
                             group_size)
        for raw in (None, True, {"enabled": 1}, {"group_size": False}, {"group_size": 0},
                    {"group_size": -1}, {"group_size": 1.5}, {"unknown": True}):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                activation_compression_config_from_dict(raw)

    def test_global_and_attention_bits_are_independent_and_validated(self):
        defaults = activation_compression_config_from_dict({})
        self.assertEqual(defaults.bits, 2)
        self.assertIsNone(defaults.attention_bits)
        for bits in (1, 2, 4, 8):
            for attention_bits in (None, 1, 2, 4, 8):
                with self.subTest(bits=bits, attention_bits=attention_bits):
                    settings = activation_compression_config_from_dict(
                        {"enabled": True, "bits": bits, "attention_bits": attention_bits})
                    self.assertEqual(settings, ActivationCompressionConfig(True, 512, bits, attention_bits))
                    self.assertEqual(normalize_activation_compression_config(settings), settings)
        for name in ("bits", "attention_bits"):
            for value in (False, True, 0, -1, 3, 16, 2.0, "2", [], {}):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ConfigError, name):
                    activation_compression_config_from_dict({name: value})
        with self.assertRaisesRegex(ConfigError, "bits"):
            activation_compression_config_from_dict({"bits": None})

    def test_sibling_configuration_round_trip(self):
        raw = {"name": "compression", "seed": 3,
               "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01},
               "runtime": {"activation_compression": {"enabled": True, "group_size": 129,
                                                       "bits": 1, "attention_bits": 2}}}
        config = config_from_dict(raw)
        self.assertEqual(config.runtime.activation_compression, ActivationCompressionConfig(True, 129, 1, 2))
        self.assertFalse(config.runtime.fp8.enabled)
        self.assertEqual(config_from_dict(asdict(config)), config)
        load_config(config)
        self.assertEqual(normalize_activation_compression_config(), config.runtime.activation_compression)
        self.assertFalse(normalize_activation_compression_config(False).enabled)

    def test_fp8_plan_copies_policy_and_preserves_activation_scale_tiles(self):
        raw = {"runtime": {"activation_compression": {"enabled": True, "group_size": 513,
                                                   "bits": 4, "attention_bits": 1}}}
        plan = fp8_plan({"enabled": True, "block_shapes": {"activation": (1, 32)}}, config=raw)
        self.assertEqual(plan["block_shapes"]["activation"], (1, 32))
        self.assertEqual(plan["fp8_training"]["activation_compression"],
                         {"enabled": True, "group_size": 513, "bits": 4, "attention_bits": 1, "norm_xhat_recenter": False})
        self.assertEqual(plan, validate_fp8_plan(plan))
        raw["runtime"]["activation_compression"]["group_size"] = 1
        self.assertEqual(plan["fp8_training"]["activation_compression"]["group_size"], 513)
        off = fp8_plan(True, config=raw, activation_compression=False)
        self.assertFalse(off["fp8_training"]["activation_compression"]["enabled"])

    def test_prepared_model_retains_and_rejects_conflicting_policy(self):
        config = {"runtime": {"activation_compression": {"enabled": True, "group_size": 32,
                                                       "bits": 1, "attention_bits": 4}}}
        model, report = prepare_fp8_model(nn.Linear(4, 2), True, config=config)
        prepared, repeated_report = prepare_fp8_model(model, True, config=ResolvedConfig())
        self.assertIs(prepared, model)
        self.assertIs(repeated_report, report)
        self.assertEqual(model.plan["fp8_training"]["activation_compression"],
                         {"enabled": True, "group_size": 32, "bits": 1, "attention_bits": 4, "norm_xhat_recenter": False})
        with self.assertRaisesRegex(ValueError, "Activation compression settings differ"):
            prepare_fp8_model(model, True, activation_compression=False)
        for override in ({"bits": 2}, {"attention_bits": 2}, {"norm_xhat_recenter": True}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "settings differ"):
                prepare_fp8_model(model, True, config=config, activation_compression=override)

    def test_execution_preparation_retains_codec_in_snapshot(self):
        model, _ = prepare_fp8_model(nn.Linear(4, 2), True,
                                    activation_compression=ActivationCompressionConfig(True, 32))
        snapshot = ResolvedConfig()
        same, settings, _ = prepare_execution_model(model, snapshot)
        self.assertIs(same, model)
        self.assertTrue(settings.enabled)
        self.assertEqual(snapshot["runtime"]["activation_compression"],
                         {"enabled": True, "group_size": 32, "bits": 2, "attention_bits": None, "norm_xhat_recenter": False})

    def test_autocast_owns_snapshot_and_nested_false_restores_outer(self):
        config = {"runtime": {"activation_compression": {"enabled": True, "group_size": 32}}}
        scope = autocast("cpu", enabled=False, config=config)
        config["runtime"]["activation_compression"]["group_size"] = 64
        with scope:
            self.assertEqual(_get_activation_compression("cpu"),
                             {"enabled": True, "group_size": 32, "bits": 2, "attention_bits": None, "norm_xhat_recenter": False})
            with autocast("cpu", enabled=False):
                self.assertTrue(_get_activation_compression("cpu")["enabled"])
            with autocast("cpu", enabled=False, activation_compression=False):
                self.assertFalse(_get_activation_compression("cpu")["enabled"])
            self.assertTrue(_get_activation_compression("cpu")["enabled"])
        self.assertIsNone(_get_activation_compression("cpu"))

    def test_nested_false_disables_compression_of_prepared_fp8(self):
        plan = fp8_plan(True, activation_compression=True)
        x, weight = torch.randn(2, 4), torch.randn(3, 4)
        with autocast("cpu", enabled=False, activation_compression=False), \
                patch("deepopt.opt.nn.functional._DOLinearFunction.apply", return_value=torch.zeros(2, 3)) as apply:
            do_linear(x, weight, plan=plan)
        self.assertFalse(apply.call_args.args[3]["fp8_training"]["activation_compression"]["enabled"])

    def test_prepared_codec_is_inherited_when_fp8_autocast_omits_compression(self):
        model, _ = prepare_fp8_model(nn.Linear(4, 3), True,
                                    activation_compression=ActivationCompressionConfig(True, 19))
        target = "deepopt.opt.nn.functional._DOLinearFunction.apply"
        with autocast("cpu", enabled=False, fp8=True, config=ResolvedConfig()), \
                patch(target, return_value=torch.zeros(2, 3)) as apply:
            model(torch.randn(2, 4))
        self.assertEqual(apply.call_args.args[3]["fp8_training"]["activation_compression"],
                         {"enabled": True, "group_size": 19, "bits": 2, "attention_bits": None, "norm_xhat_recenter": False})
        with autocast("cpu", enabled=False, fp8=True, config=ResolvedConfig(), activation_compression=False), \
                patch(target, return_value=torch.zeros(2, 3)) as apply:
            model(torch.randn(2, 4))
        self.assertFalse(apply.call_args.args[3]["fp8_training"]["activation_compression"]["enabled"])
        with autocast("cpu", enabled=False, fp8=True, config=ResolvedConfig()), \
                patch(target, return_value=torch.zeros(2, 3)) as apply:
            F.linear(torch.randn(2, 4), torch.randn(3, 4))
        self.assertFalse(apply.call_args.args[3]["fp8_training"]["activation_compression"]["enabled"])

    def test_dispatches_only_effective_bf16_linear_and_keeps_fp8_priority(self):
        x, weight = torch.randn(2, 4), torch.randn(3, 4)
        target = "deepopt.opt.nn.functional.do_linear"
        with patch(target, return_value=torch.zeros(2, 3)) as linear:
            with autocast("cpu", enabled=False, activation_compression=True):
                F.linear(x, weight)
            linear.assert_not_called()
            with autocast("cpu", dtype=torch.bfloat16, activation_compression={"enabled": True, "group_size": 19}):
                F.linear(x, weight)
            self.assertEqual(linear.call_args.kwargs["plan"],
                             {"fp8": False, "activation_compression": {
                                 "enabled": True, "group_size": 19, "bits": 2, "attention_bits": None, "norm_xhat_recenter": False}})
            linear.reset_mock()
            with autocast("cpu", dtype=torch.bfloat16, activation_compression=True):
                with autocast("cpu", activation_compression=False):
                    F.linear(x, weight)
            linear.assert_not_called()
            with autocast("cpu", enabled=False, activation_compression=True):
                F.linear(x.bfloat16(), weight.bfloat16())
            linear.assert_called_once()
            linear.reset_mock()
            with autocast("cpu", dtype=torch.bfloat16, activation_compression=True, fp8=True):
                F.linear(x, weight)
            linear.assert_called_once()
            self.assertIn("fp8_training", linear.call_args.kwargs["plan"])

    def test_disabling_prepared_fp8_still_allows_explicit_bf16_compression(self):
        model, _ = prepare_fp8_model(nn.Linear(4, 3), True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=True), \
                patch("deepopt.opt.nn.functional._native_linear", return_value=torch.zeros(2, 3)) as native:
            model(torch.randn(2, 4))
        native.assert_called_once()
        self.assertEqual(native.call_args.args[3],
                         {"enabled": True, "group_size": 512, "bits": 2, "attention_bits": None, "norm_xhat_recenter": False})

    def test_native_linear_interception_preserves_nondefault_precision_metadata(self):
        settings = {"enabled": True, "group_size": 19, "bits": 1, "attention_bits": 2, "norm_xhat_recenter": True}
        with autocast("cpu", dtype=torch.bfloat16, activation_compression=settings), \
                patch("deepopt.opt.nn.functional.do_linear", return_value=torch.zeros(2, 3)) as linear:
            F.linear(torch.randn(2, 4), torch.randn(3, 4))
        self.assertEqual(linear.call_args.kwargs["plan"],
                         {"fp8": False, "activation_compression": settings})


if __name__ == "__main__":
    unittest.main()
