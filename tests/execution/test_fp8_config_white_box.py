"""FP8 configuration ownership, lazy imports, and compute-plan translation."""

from dataclasses import asdict
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt.config import ConfigError, FP8Config, fp8_config_from_dict
from deepopt.opt.do_nn.functional import validate_fp8_plan
from deepopt.opt.policy import fp8_plan, normalize_fp8_config, prepare_fp8_model


class FP8ConfigWhiteBoxTests(unittest.TestCase):
    def test_config_and_normalization_do_not_import_torch(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
from deepopt.config import FP8Config, fp8_config_from_dict
from deepopt.opt.policy import normalize_fp8_config
assert normalize_fp8_config(True).enabled
assert fp8_config_from_dict({}).enabled is False
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
"""], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_owns_copies_of_supplied_shape_arrays(self):
        shape = [32, 64]
        raw = {"enabled": True, "block_shapes": {"activation": shape}}
        settings = fp8_config_from_dict(raw)
        shape[0] = 128
        raw["block_shapes"]["weight"] = [128, 128]
        self.assertEqual(settings.block_shapes["activation"], (32, 64))
        self.assertIsNone(settings.block_shapes["weight"])

    def test_normalizing_dataclass_revalidates_and_copies_nested_settings(self):
        source = FP8Config(enabled=True, block_shapes={"weight": (32, 64)})
        settings = normalize_fp8_config(source)
        source.block_shapes["weight"] = (64, 64)
        self.assertEqual(settings.block_shapes["weight"], (32, 64))
        self.assertEqual(set(settings.block_shapes), {"activation", "weight", "grad"})
        with self.assertRaises(ConfigError):
            normalize_fp8_config(FP8Config(enabled="yes"))

    def test_default_dataclasses_do_not_share_shape_mapping(self):
        first, second = FP8Config(), FP8Config()
        first.block_shapes["grad"] = (32, 32)
        self.assertIsNone(second.block_shapes["grad"])

    def test_plan_matches_existing_functional_contract(self):
        settings = fp8_config_from_dict({
            "enabled": True, "grad_dtype": "e5m2", "num_stages": 2,
            "out_dtype": "bf16", "block_shapes": {"activation": [32, 64]},
        })
        plan = fp8_plan(settings)
        self.assertEqual(plan, validate_fp8_plan(plan))
        self.assertEqual(plan["fp8_training"]["fp8_dtypes"], {
            "activation": torch.float8_e4m3fn, "weight": torch.float8_e4m3fn,
            "grad": torch.float8_e5m2,
        })
        self.assertIs(plan["fp8_training"]["out_dtype"], torch.bfloat16)
        self.assertEqual(plan["fp8_training"]["num_stages"], 2)
        self.assertEqual(plan["block_shapes"]["activation"], (32, 64))

    def test_output_dtype_names_are_resolved_only_in_compute_plan(self):
        for name, dtype in ((None, None), ("fp32", torch.float32),
                            ("bf16", torch.bfloat16), ("fp16", torch.float16)):
            with self.subTest(name=name):
                settings = FP8Config(out_dtype=name)
                self.assertEqual(asdict(settings)["out_dtype"], name)
                self.assertIs(fp8_plan(settings)["fp8_training"]["out_dtype"], dtype)

    def test_plan_mutation_does_not_modify_configuration(self):
        settings = FP8Config(enabled=True)
        first, second = fp8_plan(settings), fp8_plan(settings)
        first["fp8_training"]["fp8_dtypes"]["grad"] = torch.float8_e5m2
        first["block_shapes"]["weight"] = (32, 32)
        self.assertEqual(settings.grad_dtype, "e4m3fn")
        self.assertIsNone(settings.block_shapes["weight"])
        self.assertIs(second["fp8_training"]["fp8_dtypes"]["grad"], torch.float8_e4m3fn)

    def test_disabled_preparation_does_not_inspect_or_transform_model(self):
        sentinel = object()
        for settings in (None, False, {}, FP8Config()):
            with self.subTest(settings=settings), patch("deepopt.opt.transform.transform_model") as transform:
                model, report = prepare_fp8_model(sentinel, settings)
                self.assertIs(model, sentinel)
                self.assertIsNone(report)
                transform.assert_not_called()

    def test_matching_preparation_is_reused_after_compilation(self):
        model, report = prepare_fp8_model(nn.Sequential(nn.Linear(4, 2)), True)
        model.compile(backend="eager")
        with patch("deepopt.opt.transform.transform_model") as transform:
            prepared, repeated_report = prepare_fp8_model(model, FP8Config(enabled=True))
            transform.assert_not_called()
        self.assertIs(prepared, model)
        self.assertIs(repeated_report, report)

    def test_compiled_or_captured_children_fail_before_mutation(self):
        for captured in (False, True):
            with self.subTest(captured=captured):
                model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
                original = model[0]
                if captured:
                    def forward(x):
                        return x
                    forward._deepopt_cuda_capture = True
                    model[1].forward = forward
                else:
                    model[1].compile(backend="eager")
                with self.assertRaisesRegex(ValueError, "Prepare FP8 before"):
                    prepare_fp8_model(model, True)
                self.assertIs(model[0], original)
                self.assertFalse(hasattr(model, "_deepopt_fp8_config"))

    def test_prepared_marker_owns_its_normalized_configuration(self):
        settings = FP8Config(enabled=True)
        model, report = prepare_fp8_model(nn.Linear(4, 2), settings)
        settings.block_shapes["weight"] = (32, 32)
        self.assertIsNone(model._deepopt_fp8_config.block_shapes["weight"])
        self.assertIs(model._deepopt_fp8_report, report)
        self.assertEqual(tuple(model.state_dict()), ("weight", "bias"))


if __name__ == "__main__":
    unittest.main()
