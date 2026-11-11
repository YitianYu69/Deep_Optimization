"""Serializable FP8 configuration for Python, TOML, and checkpoint round trips."""

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import fx, nn
from torch.nn import functional as F

from deepopt.config import (ConfigError, FP8Config, config_from_dict, fp8_config_from_dict,
                            _read_config as load_config, reset_config)
from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.nn.attention import DOMultiheadAttention
from deepopt.opt.policy import normalize_fp8_config, prepare_fp8_model


def experiment(fp8=None):
    raw = {"name": "fp8", "seed": 7,
           "training": {"epochs": 1, "batch_size": 4, "learning_rate": .01}}
    if fp8 is not None:
        raw["runtime"] = {"fp8": fp8}
    return raw


class FP8ConfigBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_missing_configuration_preserves_disabled_behavior(self):
        config = config_from_dict(experiment())
        self.assertEqual(config.runtime.fp8, FP8Config())
        self.assertFalse(config.runtime.fp8.enabled)
        self.assertEqual(config_from_dict(asdict(config)), config)

    def test_python_shortcuts_have_explicit_enablement(self):
        for value in (None, False, {}, FP8Config()):
            with self.subTest(value=value):
                self.assertFalse(normalize_fp8_config(value).enabled)
        for value in (True, {"enabled": True}, FP8Config(enabled=True)):
            with self.subTest(value=value):
                self.assertTrue(normalize_fp8_config(value).enabled)

    def test_toml_configures_operand_dtypes_and_block_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "fp8.toml")
            path.write_text('''name = "fp8"
seed = 7
[training]
epochs = 1
batch_size = 4
learning_rate = 0.01
[runtime.fp8]
enabled = true
backend = "reference"
trace_mode = "modules"
activation_dtype = "e4m3fn"
weight_dtype = "e4m3fn"
grad_dtype = "e5m2"
num_stages = 2
out_dtype = "bf16"
[runtime.fp8.block_shapes]
activation = [16, 32]
weight = [32, 64]
''')
            config = load_config(path)
        settings = config.runtime.fp8
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.trace_mode, "modules")
        self.assertEqual(settings.grad_dtype, "e5m2")
        self.assertEqual(settings.out_dtype, "bf16")
        self.assertEqual(settings.block_shapes,
                         {"activation": (16, 32), "weight": (32, 64), "grad": None})

    def test_json_checkpoint_roundtrip_restores_tuple_shapes(self):
        config = config_from_dict(experiment({
            "enabled": True, "backend": "triton", "num_stages": 4,
            "block_shapes": {"activation": [32, 96], "grad": [64, 32]},
        }))
        restored = config_from_dict(json.loads(json.dumps(asdict(config))))
        self.assertEqual(restored, config)
        self.assertEqual(restored.runtime.fp8.block_shapes["activation"], (32, 96))

    def test_backends_allow_independent_operand_formats(self):
        for backend in ("reference", "triton"):
            for formats in product(("e4m3fn", "e5m2"), repeat=3):
                with self.subTest(backend=backend, formats=formats):
                    options = dict(zip(("activation_dtype", "weight_dtype", "grad_dtype"), formats))
                    config = fp8_config_from_dict({"backend": backend, **options})
                    self.assertEqual((config.activation_dtype, config.weight_dtype, config.grad_dtype), formats)

    def test_triton_allows_rowwise_and_unaligned_scale_blocks(self):
        for shape in ([1, 32], [32, 1], [3, 5]):
            with self.subTest(shape=shape):
                config = fp8_config_from_dict({"backend": "triton", "block_shapes": {
                    role: shape for role in ("activation", "weight", "grad")},
                })
                self.assertEqual(config.block_shapes,
                                 dict.fromkeys(("activation", "weight", "grad"), tuple(shape)))

    def test_invalid_configuration_fails_with_config_error(self):
        invalid = (
            None, True, [], {"enabled": 1}, {"enabled": "true"},
            {"backend": "unknown"}, {"trace_mode": "eager"}, {"unknown": True},
            {"activation_dtype": "float8_e4m3fn"}, {"weight_dtype": "fp32"},
            {"grad_dtype": None}, {"out_dtype": "float32"},
            {"num_stages": True}, {"num_stages": 0}, {"num_stages": 6},
            {"num_stages": "AUTO"}, {"num_stages": None}, {"num_stages": 3.0},
            {"intercept_linear": 0}, {"intercept_linear": "false"},
            {"block_shapes": []}, {"block_shapes": {"input": [32, 32]}},
            {"block_shapes": {"activation": [32]}},
            {"block_shapes": {"weight": [0, 32]}},
            {"block_shapes": {"grad": [True, 32]}},
            {"block_shapes": {"grad": [32.0, 32]}},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                fp8_config_from_dict(value)

    def test_compile_and_fp8_settings_remain_independent(self):
        raw = experiment({"enabled": True})
        raw["runtime"]["compile"] = {"enabled": True, "backend": "eager"}
        config = config_from_dict(raw)
        self.assertTrue(config.runtime.fp8.enabled)
        self.assertTrue(config.runtime.compile.enabled)
        self.assertEqual(config.runtime.compile.backend, "eager")


class FP8PreparationBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_default_module_preparation_preserves_container_methods_and_parameters(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 2)

            def forward(self, x):
                return self.layer(x)

            def describe(self):
                return "original root"

        source = Model()
        parameters = tuple(source.parameters())
        keys = tuple(source.state_dict())
        model, report = prepare_fp8_model(source, True)
        self.assertIs(model, source)
        self.assertEqual(model.describe(), "original root")
        self.assertEqual(report.trace_mode, "modules")
        self.assertIsInstance(model.layer, DOLinear)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertTrue(all(actual is expected for actual, expected in zip(model.parameters(), parameters)))
        model(torch.ones(3, 4)).sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))

    def test_root_linear_becomes_adapter_without_replacing_master_parameters(self):
        source = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(source.parameters(), lr=.1)
        model, report = prepare_fp8_model(source, True)
        self.assertIsInstance(model, DOLinear)
        self.assertIs(model.weight, source.weight)
        self.assertIs(model.bias, source.bias)
        self.assertEqual(report.converted_modules, ("<root>",))
        before = source.weight.detach().clone()
        model(torch.ones(3, 4)).sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(source.weight, before))

    def test_root_attention_uses_registered_adapter(self):
        source = nn.MultiheadAttention(8, 2, batch_first=True)
        keys = tuple(source.state_dict())
        model, report = prepare_fp8_model(source, True)
        self.assertIsInstance(model, DOMultiheadAttention)
        self.assertIs(model.in_proj_weight, source.in_proj_weight)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertEqual(report.converted_modules, ("<root>",))
        inputs = torch.randn(2, 3, 8)
        output, _ = model(inputs, inputs, inputs, need_weights=False)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(source.in_proj_weight.grad)

    def test_explicit_full_tracing_rewrites_functional_root(self):
        class Functional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.eye(4))

            def forward(self, x):
                return F.linear(x, self.weight)

        source = Functional()
        model, report = prepare_fp8_model(source, {"enabled": True, "trace_mode": "full"})
        self.assertIsInstance(model, fx.GraphModule)
        self.assertIs(model.weight, source.weight)
        self.assertEqual(report.trace_mode, "full")
        self.assertEqual(len(report.function_calls), 1)
        model(torch.ones(2, 4)).sum().backward()
        self.assertIsNotNone(source.weight.grad)

    def test_changed_prepared_configuration_requires_reconstruction(self):
        model, _ = prepare_fp8_model(nn.Linear(4, 2), True)
        with self.assertRaisesRegex(ValueError, "reconstruct the original model"):
            prepare_fp8_model(model, {"enabled": True, "grad_dtype": "e5m2"})

    def test_torch_compile_wrapper_cannot_be_prepared_after_compilation(self):
        source = nn.Linear(4, 2)
        compiled = torch.compile(source, backend="eager")
        with self.assertRaisesRegex(ValueError, "before compilation"):
            prepare_fp8_model(compiled, True)


if __name__ == "__main__":
    unittest.main()
