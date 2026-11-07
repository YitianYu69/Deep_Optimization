from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt import Checkpoint, Trainer
from deepopt.config import config_from_dict
from deepopt.data import ResumableDataLoader
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint
from deepopt.experiment import build_experiment
from deepopt.run import train, evaluate
from deepopt.scheduling import ScheduleController, LinearWarmup


def make_trainer(records, duration=10):
    torch.manual_seed(3)
    schedules = ScheduleController({"beta": LinearWarmup(.1, 1., duration)})
    model = nn.Linear(1, 1)
    def step(model, batch):
        records.append(schedules["beta"])
        return model(batch[0]).square().mean() * schedules["beta"]
    return Trainer(model, torch.optim.AdamW(model.parameters(), lr=.01), step_fn=step,
                   accumulation_steps=2, schedules=schedules, execution_mode="max_speed")


def loader(batch_size=2):
    return ResumableDataLoader(TensorDataset(torch.arange(9, dtype=torch.float32).reshape(-1, 1) / 10),
                               batch_size=batch_size, generator=torch.Generator().manual_seed(8))


class SchedulingResumeTests(unittest.TestCase):
    def test_mid_epoch_resume_preserves_sequence_parameters_and_partial_window(self):
        full_records, resumed_records = [], []
        full = make_trainer(full_records)
        full.fit(loader(), epochs=2, progress_bar=False)
        for stop_at in (1, 3):
            with self.subTest(stop_at=stop_at), tempfile.TemporaryDirectory() as directory:
                resumed_records = []
                partial = make_trainer(resumed_records)
                checkpoint = Checkpoint(directory, every_steps=1)
                def stop(trainer, event):
                    if trainer.state.optimizer_steps == stop_at:
                        raise RuntimeError("interrupt")
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    partial.fit(loader(), epochs=2, callbacks=[checkpoint], step_callbacks=[stop], progress_bar=False)
                # Callback-provided checkpoint callbacks precede explicit ones.
                payload = read_checkpoint(Path(directory, "last_step.pt"))
                self.assertEqual(payload["format_version"], 5)
                self.assertEqual(payload["schedules"]["completed_update"], stop_at)
                restored, batches = make_trainer(resumed_records), loader()
                checkpoint = Checkpoint(directory, every_steps=1)
                checkpoint.restore(restored, batches)
                before = restored.schedules.state_dict()
                restored.evaluate([(torch.ones(1, 1),)])
                resumed_records.pop()  # Validation reads the last value; it is not a training sample.
                self.assertEqual(before, restored.schedules.state_dict())
                restored.fit(batches, epochs=2, callbacks=[checkpoint], progress_bar=False)
                self.assertEqual(resumed_records, full_records)
                for actual, expected in zip(restored.model.parameters(), full.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(restored.schedules.state_dict(), full.schedules.state_dict())

    def test_batch_size_change_keeps_position_and_horizon_and_rejects_policy_change(self):
        with tempfile.TemporaryDirectory() as directory:
            records = []
            original = make_trainer(records)
            def stop(trainer, event):
                raise RuntimeError("interrupt")
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                original.fit(loader(), epochs=2, callbacks=[Checkpoint(directory, every_steps=1)],
                             step_callbacks=[stop], progress_bar=False)
            changed_records = []
            changed, batches = make_trainer(changed_records), loader(3)
            checkpoint = Checkpoint(directory)
            checkpoint.restore(changed, batches)
            changed.train_epoch(batches)
            self.assertEqual(changed_records[0], LinearWarmup(.1, 1., 10).value_at(2))
            self.assertEqual(changed.schedules.get_config()["beta"]["config"]["duration"], 10)
            with self.assertRaisesRegex(ValueError, "schedule"):
                Checkpoint(directory).restore(make_trainer([], duration=11), loader())

    def test_v4_no_schedules_compatibility_and_rejection_of_new_controller(self):
        model = nn.Linear(1, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), loss_fn=nn.functional.mse_loss)
        batches = [(torch.ones(2, 1), torch.zeros(2, 1))]
        with tempfile.TemporaryDirectory() as directory:
            trainer.fit(batches, epochs=1, callbacks=[Checkpoint(directory)], progress_bar=False)
            payload = read_checkpoint(Path(directory, "last.pt"))
            payload["format_version"] = 4
            payload.pop("schedules")
            torch.save(payload, Path(directory, "last.pt"))
            self.assertEqual(read_checkpoint(Path(directory, "last.pt"))["format_version"], 4)
            restored_model = nn.Linear(1, 1)
            restored = Trainer(restored_model, torch.optim.SGD(restored_model.parameters(), lr=.01),
                               loss_fn=nn.functional.mse_loss)
            Checkpoint(directory).restore(restored, batches)
            restored.schedules = ScheduleController({"beta": LinearWarmup(0, 1, 10)})
            with self.assertRaisesRegex(ValueError, "coefficient schedules"):
                restore_checkpoint(payload, restored, {}, generators={}, train_loader=batches, batch_size_mode="manual")

    def test_configured_factory_resume_and_evaluation_restore_coefficients(self):
        raw = {"name": "scheduled", "seed": 3,
               "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01, "scheduler_interval": "step",
                            "lr_schedule": {"type": "linear_warmup", "warmup_steps": 3, "start_factor": .1}},
               "schedules": {"beta": {"type": "linear_warmup", "start": .1, "end": 1., "duration": 5}}}
        seen = []
        def factory(config):
            dataset = TensorDataset(torch.ones(6, 1), torch.zeros(6, 1))
            def step_factory(schedules):
                def step(model, batch):
                    if not model.training:
                        seen.append(schedules["beta"])
                    return model(batch[0]).square().mean() * schedules["beta"]
                return step
            return build_experiment(config, model_factory=lambda: nn.Linear(1, 1),
                                    dataset_factory=lambda: (dataset, dataset), step_factory=step_factory)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory, "run")
            train(config_from_dict(raw), output=output, experiment_factory=factory, progress_bar=False)
            payload = read_checkpoint(output / "last.pt")
            expected = payload["schedules"]["values"]["beta"]
            seen.clear()
            evaluate(output / "last.pt", experiment_factory=factory)
            self.assertEqual(seen, [expected] * 3)
            train(resume=output / "last.pt", epochs=2, experiment_factory=factory, progress_bar=False)
            self.assertEqual(read_checkpoint(output / "last.pt")["schedules"]["completed_update"], 6)


if __name__ == "__main__":
    unittest.main()
