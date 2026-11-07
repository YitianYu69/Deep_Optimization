"""Default resolution precedence, snapshots, and side-effect boundaries."""

from contextlib import redirect_stderr
from dataclasses import asdict
import io
import logging
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt._config_resolution import ResolvedConfig, config_scope, resolve_config
from deepopt import autocast
from deepopt.config import CompileConfig, ConfigError, FP8Config, get_config, load_config, reset_config
from deepopt.data import build_loaders
from deepopt.engine.execution.compilation import compile_model
from deepopt.logger import init_logger, log_filename, logging_session
from deepopt.opt.policy import normalize_fp8_config, prepare_fp8_model, resolve_fp8_config
from deepopt.opt._autocast import _get_fp8_plan, fp8_context
from deepopt.optimizer_factory import build_optimizer
from deepopt.scheduling import ScheduleController
from deepopt.scheduling.lr import build_scheduler
from tests.application.config_support import configuration


class GlobalConsumerWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        previous = get_config()
        reset_config()
        self.addCleanup(lambda: load_config(previous) if previous is not None else reset_config())

    def test_unconfigured_defaults_do_not_invent_required_training_or_loader_values(self):
        with self.assertRaises(ConfigError):
            build_optimizer(nn.Linear(2, 1))
        with self.assertRaisesRegex(ValueError, "batch_size and seed"):
            build_loaders(TensorDataset(torch.arange(4)), None)
        with patch("torch.compile") as compiler:
            model = nn.Linear(2, 1)
            self.assertIs(compile_model(model), model)
        compiler.assert_not_called()
        self.assertEqual(len(ScheduleController()), 0)
        self.assertFalse(normalize_fp8_config().enabled)

    def test_compiler_inherits_global_and_explicit_disabled_settings_win(self):
        load_config(configuration(runtime={"compile": {"enabled": True, "backend": "eager", "fullgraph": True}}))
        model = nn.Linear(2, 1)
        with patch("torch.compile", return_value=Mock()) as compiler:
            compile_model(model)
        compiler.assert_called_once_with(model._call_impl, backend="eager", fullgraph=True, dynamic=None)
        for explicit in ({"enabled": False}, CompileConfig()):
            with self.subTest(explicit=explicit), patch("torch.compile") as disabled:
                compile_model(nn.Linear(2, 1), explicit)
            disabled.assert_not_called()

    def test_loader_validates_the_combined_configuration_after_explicit_worker_options(self):
        dataset = TensorDataset(torch.arange(4))
        loader, _, _ = build_loaders(
            dataset, None, batch_size=2, seed=3, num_workers=2,
            config={"loader": {"persistent_workers": True, "prefetch_factor": 2}},
        )
        self.assertEqual(loader.num_workers, 2)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.prefetch_factor, 2)
        # No iterator is created, so this check does not launch worker processes.

    def test_fp8_explicit_fields_complete_partial_policy_before_validation(self):
        partial = {"runtime": {"fp8": {"enabled": True, "grad_dtype": "e5m2", "backend": "triton"}}}
        settings = resolve_fp8_config(nn.Linear(2, 1), {"backend": "reference"}, config=partial)
        self.assertEqual(settings.backend, "reference")
        self.assertEqual(settings.grad_dtype, "e5m2")
        with fp8_context("cpu", {"backend": "reference"}, config=partial):
            self.assertEqual(_get_fp8_plan("cpu")["fp8_training"]["backend"], "reference")
        with autocast("cpu", enabled=False, fp8={"backend": "reference"}, config=partial):
            self.assertEqual(_get_fp8_plan("cpu")["fp8_training"]["backend"], "reference")

    def test_explicit_fp8_without_config_does_not_invent_an_amp_precision_policy(self):
        scope = autocast("cpu", fp8=True)
        self.assertNotIn("precision", scope.config.get("runtime", {}))
        with scope:
            self.assertTrue(torch.is_autocast_enabled("cpu"))

    def test_resolved_snapshot_bypasses_global_changes_for_all_section_consumers(self):
        load_config(configuration())
        snapshot = resolve_config()
        load_config(configuration(name="later", seed=71, schedules={"different": {"type": "constant", "value": .9}}))
        before = asdict(get_config())
        with config_scope(snapshot):
            optimizer = build_optimizer(nn.Linear(2, 1))
            controller = ScheduleController()
            loader, _, generators = build_loaders(TensorDataset(torch.arange(9)), None)
            self.assertEqual(optimizer.param_groups[0]["lr"], .05)
            self.assertEqual(dict(controller.values), {"coefficient": .3})
            self.assertEqual(loader.batch_size, 4)
            self.assertEqual(generators["train"].initial_seed(), 14)
        self.assertEqual(asdict(get_config()), before)

    def test_automatic_optimizer_and_scheduler_need_only_their_own_required_fields(self):
        snapshot = resolve_config({"training": {"learning_rate": .04, "optimizer": "sgd", "momentum": 0., "lr_gamma": .7}}, base=None)
        self.assertNotIn("epochs", snapshot["training"])
        self.assertNotIn("batch_size", snapshot["training"])
        optimizer = build_optimizer(nn.Linear(2, 1), snapshot)
        scheduler = build_scheduler(optimizer, snapshot)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .028)
        self.assertNotIn("epochs", snapshot["training"])
        scheduler_only = build_scheduler(optimizer, ResolvedConfig())
        self.assertEqual(scheduler_only.gamma, 1.)
        with self.assertRaises(ConfigError):
            build_optimizer(nn.Linear(2, 1), ResolvedConfig())

    def test_explicit_none_preserves_prepared_fp8_policy_after_global_change(self):
        load_config(configuration(runtime={"fp8": {"enabled": True}}))
        model, _ = prepare_fp8_model(nn.Linear(2, 1))
        load_config(configuration())
        self.assertTrue(resolve_fp8_config(model, None).enabled)
        with self.assertRaisesRegex(ValueError, "FP8 settings differ"):
            resolve_fp8_config(model)
        self.assertFalse(normalize_fp8_config(None).enabled)
        self.assertEqual(resolve_fp8_config(model, FP8Config(enabled=True)), model._deepopt_fp8_config)

    def test_autocast_owns_explicit_policy_before_first_entry(self):
        policy = {"enabled": True, "block_shapes": {"activation": [2, 3]}}
        scope = autocast("cpu", enabled=False, fp8=policy)
        policy["enabled"] = False
        policy["block_shapes"]["activation"][0] = 7
        with scope:
            self.assertEqual(_get_fp8_plan("cpu")["block_shapes"]["activation"], (2, 3))

    def test_logger_keyword_levels_override_metadata_and_global_without_mutating_either(self):
        load_config(configuration())
        before = asdict(get_config())
        metadata = {"application_run": "custom", "logging": {"level": "INFO", "file_level": "ERROR"}}
        with redirect_stderr(io.StringIO()), logging_session(metadata, level="CRITICAL", file_level="WARNING") as session:
            self.assertEqual(session.level, logging.CRITICAL)
            self.assertEqual(session.file_level, logging.WARNING)
            self.assertEqual(session.config["name"], "global-consumers")
            self.assertEqual(session.config["application_run"], "custom")
            self.assertIsNone(session.path)
        self.assertEqual(metadata["logging"], {"level": "INFO", "file_level": "ERROR"})
        self.assertEqual(asdict(get_config()), before)

    def test_explicit_none_console_level_uses_environment_instead_of_global(self):
        load_config(configuration())
        with patch.dict(os.environ, {"DEEPOPT_LOGGING_LEVEL": "ERROR"}), redirect_stderr(io.StringIO()):
            with logging_session(level=None) as session:
                self.assertEqual(session.level, logging.ERROR)

    def test_logger_binding_uses_session_snapshot_after_global_replacement(self):
        load_config(configuration())
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            with logging_session() as session:
                original = dict(session.config)
                init_logger(__name__).warning("before binding")
                load_config(configuration(name="different", seed=77))
                session.bind(directory, config={"training": {"batch_size": 2}})
            self.assertEqual(session.config["name"], original["name"])
            self.assertEqual(session.config["seed"], original["seed"])
            self.assertEqual(session.config["training"]["batch_size"], 2)
            self.assertIn("before binding", session.path.read_text())
            self.assertTrue(session.path.name.startswith("global-consumers-bs2"))
        self.assertTrue(log_filename().startswith("different-"))


if __name__ == "__main__":
    unittest.main()
