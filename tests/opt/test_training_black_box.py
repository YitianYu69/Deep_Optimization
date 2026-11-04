"""FP8 training through the public trainer and durable checkpoint callback."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import Checkpoint, Trainer
from deepopt.data import ResumableDataLoader
from deepopt.opt import transform_model


def fp8_plan(backend="reference"):
    return {
        "fp8_training": {"fp8_dtypes": {
            "activation": torch.float8_e4m3fn,
            "weight": torch.float8_e4m3fn,
            "grad": torch.float8_e4m3fn if backend == "triton" else torch.float8_e5m2,
        }, "backend": backend},
        "block_shapes": {name: None for name in ("activation", "weight", "grad")},
    }


def make_training(mode="full", *, device="cpu", backend="reference"):
    torch.manual_seed(81)
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4)).to(device)
    model, _ = transform_model(model, plan=fp8_plan(backend), trace_mode=mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, accumulation_steps=2, device=device,
                      scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9))
    rng = torch.Generator().manual_seed(12)
    dataset = TensorDataset(torch.randn(13, 8, generator=rng), torch.randn(13, 4, generator=rng))
    loader = ResumableDataLoader(dataset, batch_size=3,
                                 generator=torch.Generator().manual_seed(27))
    return trainer, loader


def assert_state_equal(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_state_equal(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for value, reference in zip(actual, expected):
            assert_state_equal(test, value, reference)
    else:
        test.assertEqual(actual, expected)


class InterruptedTraining(Exception):
    pass


class FP8TrainingBlackBoxTests(unittest.TestCase):
    def test_step_and_epoch_resume_match_uninterrupted_fp8_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("full", "modules"):
                reference, loader = make_training(mode)
                reference.fit(loader, epochs=2, progress_bar=False, monitor=None)
                for boundary in ("step", "epoch"):
                    with self.subTest(mode=mode, boundary=boundary):
                        trainer, loader = make_training(mode)
                        checkpoint = Checkpoint(root / mode / boundary,
                                                every_steps=1 if boundary == "step" else 0)

                        def interrupt_step(current, event):
                            if current.state.optimizer_steps == 1:
                                raise InterruptedTraining()

                        def interrupt_epoch(current, row):
                            if current.state.completed_epochs == 1:
                                raise InterruptedTraining()

                        with self.assertRaises(InterruptedTraining):
                            trainer.fit(
                                loader, epochs=2, progress_bar=False, monitor=None,
                                callbacks=[checkpoint] + ([interrupt_epoch] if boundary == "epoch" else []),
                                step_callbacks=[interrupt_step] if boundary == "step" else [],
                            )
                        resumed, resumed_loader = make_training(mode)
                        restored = Checkpoint(root / mode / boundary,
                                              every_steps=1 if boundary == "step" else 0)
                        restored.restore(resumed, resumed_loader)
                        resumed.fit(resumed_loader, epochs=2, callbacks=[restored],
                                    progress_bar=False, monitor=None)
                        assert_state_equal(self, resumed.model.state_dict(), reference.model.state_dict())
                        assert_state_equal(self, resumed.optimizer.state_dict(), reference.optimizer.state_dict())
                        assert_state_equal(self, resumed.scheduler.state_dict(), reference.scheduler.state_dict())
                        self.assertEqual(resumed.state.optimizer_steps, 6)
                        self.assertEqual(resumed.state.completed_epochs, 2)


if __name__ == "__main__":
    unittest.main()
