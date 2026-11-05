"""Configuration defaults flow into real consumers without changing existing objects."""

from contextlib import redirect_stderr
from dataclasses import asdict
import io
import logging
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import autocast
from deepopt.config import FP8Config, TrainingConfig, get_config, load_config, reset_config
from deepopt.data import ResumableDataLoader, build_loaders
from deepopt.logger import init_logger, logging_session
from deepopt.opt.do_nn.functional import do_linear
from deepopt.opt.policy import fp8_plan, normalize_fp8_config
from deepopt.optimizer_factory import build_optimizer
from deepopt.scheduling import Constant, ScheduleController
from deepopt.scheduling.lr import build_scheduler


def configuration(**changes):
    result = {
        "name": "global-consumers", "seed": 13,
        "training": {"epochs": 2, "batch_size": 4, "learning_rate": .05,
                     "optimizer": "sgd", "momentum": 0., "lr_gamma": .5},
        "runtime": {"device": "cpu", "precision": "fp32"},
        "loader": {"num_workers": 0, "drop_last": True},
        "logging": {"level": "WARNING", "file_level": "DEBUG"},
        "schedules": {"coefficient": {"type": "constant", "value": .3}},
    }
    result.update(changes)
    return result


class GlobalConsumerBlackBoxTests(unittest.TestCase):
    def setUp(self):
        previous = get_config()
        reset_config()
        self.addCleanup(lambda: load_config(previous) if previous is not None else reset_config())
        torch.manual_seed(832)

    def test_optimizer_and_scheduler_inherit_global_and_keep_their_settings(self):
        load_config(configuration())
        model = nn.Linear(2, 1)
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]["lr"], .05)
        changed = configuration()
        changed["training"].update(learning_rate=.2, lr_gamma=.9)
        load_config(changed)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], .025)
        self.assertEqual(build_optimizer(nn.Linear(2, 1)).param_groups[0]["lr"], .2)

    def test_explicit_training_section_overrides_global_without_mutating_it(self):
        load_config(configuration())
        before = asdict(get_config())
        raw = {"epochs": 2, "batch_size": 4, "learning_rate": .1}
        optimizer = build_optimizer(nn.Linear(2, 1), raw)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]["lr"], .1)
        typed = TrainingConfig(1, 2, .3, optimizer="adamw")
        self.assertIsInstance(build_optimizer(nn.Linear(2, 1), typed), torch.optim.AdamW)
        self.assertEqual(asdict(get_config()), before)
        self.assertEqual(raw, {"epochs": 2, "batch_size": 4, "learning_rate": .1})

    def test_loader_uses_global_batch_seed_options_and_explicit_arguments_win(self):
        load_config(configuration())
        dataset = TensorDataset(torch.arange(9))
        train, validation, generators = build_loaders(dataset, dataset)
        self.assertEqual(train.batch_size, 4)
        self.assertTrue(train.drop_last)
        self.assertEqual(generators["train"].initial_seed(), 14)
        self.assertEqual(validation.batch_size, 4)
        changed = configuration(seed=61)
        changed["training"]["batch_size"] = 2
        load_config(changed)
        self.assertEqual(train.batch_size, 4)
        self.assertEqual(generators["train"].initial_seed(), 14)

        collate = lambda samples: torch.stack([sample[0] for sample in samples])
        actual, _, _ = build_loaders(
            dataset, None, config={"training": {"batch_size": 3}},
            batch_size=2, seed=23, num_workers=0, drop_last=False, collate_fn=collate,
        )
        self.assertEqual(actual.batch_size, 2)
        self.assertIs(actual.collate_fn, collate)
        self.assertEqual(sorted(torch.cat(list(actual)).tolist()), list(range(9)))
        self.assertEqual(get_config().training.batch_size, 2)
        self.assertEqual(get_config().seed, 61)

    def test_loader_step_checkpoint_default_is_resumable_and_explicit_false_wins(self):
        config = configuration()
        config["training"]["checkpoint_every_steps"] = 2
        load_config(config)
        dataset = TensorDataset(torch.arange(9))
        loader, _, _ = build_loaders(dataset, None)
        self.assertIsInstance(loader, ResumableDataLoader)
        ordinary, _, _ = build_loaders(dataset, None, resumable=False)
        self.assertNotIsInstance(ordinary, ResumableDataLoader)

    def test_schedule_controller_snapshots_global_and_explicit_mapping_replaces_it(self):
        load_config(configuration())
        controller = ScheduleController()
        self.assertEqual(controller["coefficient"], .3)
        changed = configuration(schedules={"coefficient": {"type": "constant", "value": .8}})
        load_config(changed)
        self.assertEqual(controller["coefficient"], .3)
        self.assertEqual(ScheduleController()["coefficient"], .8)
        explicit = ScheduleController({"other": Constant(.2)})
        self.assertEqual(dict(explicit.values), {"other": .2})
        self.assertEqual(len(ScheduleController({})), 0)

    def test_autocast_and_fp8_policy_inherit_global_but_explicit_none_false_are_distinct(self):
        config = configuration(runtime={"device": "cpu", "precision": "fp32",
                                        "fp8": {"enabled": True, "grad_dtype": "e5m2"}})
        load_config(config)
        x, weight = torch.randn(5, 7), torch.randn(4, 7)
        expected = do_linear(x, weight, plan=fp8_plan(FP8Config(enabled=True, grad_dtype="e5m2")))
        dense = F.linear(x, weight)
        held_scope = autocast("cpu")
        self.assertTrue(normalize_fp8_config().enabled)
        self.assertFalse(normalize_fp8_config(False).enabled)
        self.assertEqual(fp8_plan()["fp8_training"]["fp8_dtypes"]["grad"], torch.float8_e5m2)
        with autocast("cpu", fp8=None):
            torch.testing.assert_close(F.linear(x, weight), dense, rtol=0, atol=0)
        load_config(configuration())
        with held_scope:
            torch.testing.assert_close(F.linear(x, weight), expected, rtol=0, atol=0)
            with autocast("cpu"):
                torch.testing.assert_close(F.linear(x, weight), expected, rtol=0, atol=0)
            with autocast("cpu", fp8=False):
                torch.testing.assert_close(F.linear(x, weight), dense, rtol=0, atol=0)
        with autocast("cpu"):
            torch.testing.assert_close(F.linear(x, weight), dense, rtol=0, atol=0)

    def test_autocast_explicit_device_and_amp_settings_override_selected_defaults(self):
        # CPU autocast supports BF16 independently of the trainer's CPU FP32 rule.
        load_config(configuration(runtime={"device": "cuda", "precision": "bf16"}))
        x, weight = torch.randn(5, 7), torch.randn(4, 7)
        with autocast("cpu"):
            self.assertEqual(F.linear(x, weight).dtype, torch.bfloat16)
        with autocast("cpu", enabled=False):
            self.assertEqual(F.linear(x, weight).dtype, torch.float32)
        load_config(configuration())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with autocast("cpu"):
                self.assertEqual(F.linear(x, weight).dtype, torch.bfloat16)

    def test_logging_inherits_levels_snapshots_metadata_and_writes_only_explicit_destinations(self):
        load_config(configuration())
        stderr = io.StringIO()
        with redirect_stderr(stderr), tempfile.TemporaryDirectory() as directory:
            with logging_session() as buffered:
                self.assertIsNone(buffered.path)
                init_logger(__name__).warning("buffered-only")
            with logging_session(directory=directory) as session:
                self.assertEqual(session.level, logging.WARNING)
                self.assertEqual(session.file_level, logging.DEBUG)
                load_config(configuration(name="replacement", logging={"level": "CRITICAL", "file_level": "CRITICAL"}))
                init_logger(__name__).warning("original-session-policy")
            text = session.path.read_text()
            self.assertIn("original-session-policy", text)
            self.assertIn("original-session-policy", stderr.getvalue())
            self.assertTrue(session.path.name.startswith("global-consumers-"))
            self.assertEqual(list(Path(directory).glob("*.log")), [session.path])


if __name__ == "__main__":
    unittest.main()
