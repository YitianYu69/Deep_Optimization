"""Compiler configuration, construction order and unchanged ownership."""

from dataclasses import asdict
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.config import CompileConfig, ConfigError, compile_config_from_dict, config_from_dict
from deepopt.engine.execution.compilation import compile_model
from deepopt.experiment import build_experiment


class CompileWhiteBoxTests(unittest.TestCase):
    def test_configuration_defaults_roundtrip_and_remains_torch_free(self):
        config = config_from_dict({"name": "compile", "seed": 1,
                                  "training": {"epochs": 1, "batch_size": 2, "learning_rate": .1}})
        self.assertEqual(config.runtime.compile, CompileConfig())
        self.assertEqual(config_from_dict(asdict(config)), config)
        result = subprocess.run([sys.executable, "-c", "import sys; from deepopt.config import CompileConfig; "
                                 "assert 'torch' not in sys.modules"], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_settings_are_rejected_without_loading_backends(self):
        for raw in (True, None, {"enabled": 1}, {"fullgraph": "yes"}, {"dynamic": 0},
                    {"backend": ""}, {"mode": "fast"}, {"unknown": True}, {"options": []},
                    {"options": {"x": float("nan")}}, {"options": {"x": []}},
                    {"mode": "default", "options": {"x": True}}):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                compile_config_from_dict(raw)

    def test_disabled_compilation_never_calls_torch(self):
        model = nn.Linear(2, 1)
        with patch("torch.compile") as compiler:
            self.assertIs(compile_model(model, CompileConfig()), model)
        compiler.assert_not_called()

    def test_compilation_preserves_identity_and_forwards_options_exactly(self):
        model = nn.Linear(2, 1)
        parameters = tuple(model.parameters())
        settings = compile_config_from_dict({"enabled": True, "backend": "inductor", "fullgraph": True,
                                             "dynamic": False, "options": {"triton.cudagraphs": False}})
        with patch("torch.compile", return_value=Mock()) as compiler:
            self.assertIs(compile_model(model, settings), model)
        compiler.assert_called_once_with(model._call_impl, backend="inductor", fullgraph=True, dynamic=False,
                                          options={"triton.cudagraphs": False})
        self.assertEqual(tuple(model.state_dict()), ("weight", "bias"))
        self.assertTrue(all(a is b for a, b in zip(parameters, model.parameters())))
        with self.assertRaisesRegex(ValueError, "already compiled"):
            compile_model(model, settings)

    def test_mode_is_forwarded_without_options_and_capture_must_follow_compile(self):
        model = nn.Linear(2, 1)
        with patch("torch.compile", return_value=Mock()) as compiler:
            compile_model(model, CompileConfig(enabled=True, mode="reduce-overhead"))
        self.assertEqual(compiler.call_args.kwargs["mode"], "reduce-overhead")
        self.assertNotIn("options", compiler.call_args.kwargs)
        model = nn.Linear(2, 1)
        model.forward = Mock(_deepopt_cuda_capture=True)
        with self.assertRaisesRegex(ValueError, "before CUDA Graph"):
            compile_model(model, CompileConfig(enabled=True))

    def test_factory_compiles_before_optimizer_and_trainer_construction(self):
        events = []
        model = nn.Linear(2, 1)
        dataset = TensorDataset(torch.ones(4, 2), torch.ones(4, 1))
        config = config_from_dict({"name": "compile", "seed": 1,
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": .1},
            "runtime": {"compile": {"enabled": True, "backend": "eager"}}})
        def compiler(fn, **kwargs):
            events.append("compile")
            return fn
        def optimizer_factory(actual, settings):
            self.assertIs(actual, model)
            events.append("optimizer")
            return torch.optim.SGD(actual.parameters(), lr=.1)
        def trainer_factory(actual, optimizer, **kwargs):
            self.assertIs(actual, model)
            events.append("trainer")
            return Mock()
        with patch("torch.compile", side_effect=compiler):
            build_experiment(config, model_factory=lambda: model, dataset_factory=lambda: (dataset, None),
                             optimizer_factory=optimizer_factory, scheduler_factory=lambda *a: None,
                             trainer_factory=trainer_factory)
        self.assertEqual(events, ["compile", "optimizer", "trainer"])


if __name__ == "__main__":
    unittest.main()
