"""Construction contracts preserved while consolidating setup logic."""

from contextlib import nullcontext
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from deepopt import DDPCore, Trainer, TrainingRuntime, load_config, reset_config
from deepopt._config_resolution import _UNSET
from deepopt.scheduling.lr import build_scheduler


class SetupCleanupTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_explicit_ddp_options_replace_inherited_options(self):
        inherited = {"broadcast_buffers": False, "bucket_cap_mb": 9,
                     "validation_sync_buffers": False}
        cases = (
            (_UNSET, False, 9, False),
            (None, True, None, True),
            ({}, True, None, True),
            ({"bucket_cap_mb": 3}, True, 3, True),
        )
        for options, broadcast, bucket, validation_sync in cases:
            with self.subTest(options=options), \
                    patch("deepopt.distributed.ddp.distributed.world_size", return_value=2), \
                    patch("deepopt.distributed.ddp.distributed.collective_errors", side_effect=lambda _: nullcontext()), \
                    patch("deepopt.distributed.ddp.distributed.require_equal"), \
                    patch("deepopt.distributed.ddp.DistributedDataParallel.__new__", return_value=Mock()) as wrapper:
                core = DDPCore(nn.Linear(1, 1), "cpu", options,
                               config={"runtime": {"ddp": inherited}})
                self.assertEqual(wrapper.call_args.kwargs["broadcast_buffers"], broadcast)
                self.assertEqual(wrapper.call_args.kwargs.get("bucket_cap_mb"), bucket)
                if options is not _UNSET and bucket is not None:
                    self.assertIs(type(wrapper.call_args.kwargs["bucket_cap_mb"]), int)
                self.assertEqual(core.validation_sync_buffers, validation_sync)
                self.assertEqual(core.config["runtime"]["ddp"], core.options)
        self.assertEqual(inherited, {"broadcast_buffers": False, "bucket_cap_mb": 9,
                                     "validation_sync_buffers": False})

    def test_scheduler_is_built_once_for_model_and_supplied_runtime(self):
        load_config({"name": "setup", "seed": 1,
                     "training": {"epochs": 1, "batch_size": 2, "learning_rate": .1, "lr_gamma": .9}})
        for supplied_runtime in (False, True):
            for selection in ("configured", "disabled", "supplied"):
                with self.subTest(runtime=supplied_runtime, selection=selection):
                    model = nn.Linear(1, 1)
                    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                    supplied = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                    options = ({} if selection == "configured" else
                               {"scheduler": None if selection == "disabled" else supplied})
                    owner = TrainingRuntime(model, optimizer) if supplied_runtime else model
                    with patch("deepopt.scheduling.lr.build_scheduler", wraps=build_scheduler) as build:
                        trainer = Trainer(owner, **({} if supplied_runtime else {"optimizer": optimizer}),
                                          loss_fn=nn.functional.mse_loss, **options)
                    self.assertEqual(build.call_count, int(selection == "configured"))
                    if selection == "disabled":
                        self.assertIsNone(trainer.scheduler)
                    else:
                        self.assertIs(trainer.scheduler.optimizer, optimizer)
                    if selection == "supplied":
                        self.assertIs(trainer.scheduler, supplied)


if __name__ == "__main__":
    unittest.main()
