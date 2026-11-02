from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from deepopt.scheduling import Schedule, ScheduleController, LinearWarmup, Constant


class StatefulCurve(Schedule):
    def __init__(self):
        self.calls = 0

    def value_at(self, position):
        self.calls += 1
        return float(position + self.calls)

    def get_config(self):
        return {"algorithm": 1}

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        if set(state) != {"calls"} or type(state["calls"]) is not int or state["calls"] < 0:
            raise ValueError("bad custom state")
        self.calls = state["calls"]


class ControllerWhiteBoxTests(unittest.TestCase):
    def test_prepare_complete_abort_and_read_only(self):
        schedules = ScheduleController({"x": LinearWarmup(0, 1, 3)})
        initial = schedules.state_dict()
        schedules.prepare_update(1)
        schedules.prepare_update(1)
        self.assertEqual(schedules["x"], 0)
        self.assertEqual(schedules.completed_update, 0)
        with self.assertRaises(RuntimeError):
            schedules.state_dict()
        schedules.abort_update()
        self.assertEqual(schedules.state_dict(), initial)
        schedules.prepare_update(1)
        schedules.complete_update(1)
        schedules.prepare_update(2)
        self.assertEqual(schedules["x"], .5)
        schedules.complete_update(2)
        before = schedules.state_dict()
        with schedules.read_only():
            self.assertEqual(schedules["x"], .5)
            for call in (lambda: schedules.prepare_update(3), lambda: schedules.complete_update(2),
                         schedules.abort_update, lambda: schedules.load_state_dict(before)):
                with self.assertRaisesRegex(RuntimeError, "Validation"):
                    call()
        self.assertEqual(before, schedules.state_dict())
        with self.assertRaises(TypeError):
            schedules.values["x"] = 1

    def test_tensor_storage_stays_stable_without_scalar_reads_or_sync(self):
        schedules = ScheduleController({"x": LinearWarmup(0, 1, 3)})
        scalar = schedules.tensor("x")
        pointer = scalar.data_ptr()
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")), \
                patch("torch.cuda.synchronize", side_effect=AssertionError("sync")), \
                patch("torch.distributed.all_reduce", side_effect=AssertionError("collective")):
            for update in range(1, 4):
                schedules.prepare_update(update)
                self.assertIs(schedules.tensor("x"), scalar)
                schedules.complete_update(update)
        self.assertEqual(scalar.data_ptr(), pointer)
        self.assertEqual(float(scalar), 1.)
        schedules.load_state_dict(schedules.state_dict())
        self.assertIs(schedules.tensor("x"), scalar)

    def test_epoch_schedule_evaluates_only_on_changed_epoch(self):
        curve = StatefulCurve()
        curve.interval = "epoch"
        schedules = ScheduleController({"x": curve})
        for update in range(1, 4):
            schedules.prepare_update(update, epoch=1)
            schedules.complete_update(update)
        self.assertEqual(curve.calls, 1)
        schedules.prepare_update(4, epoch=2)
        self.assertEqual(curve.calls, 2)
        schedules.abort_update()
        self.assertEqual(curve.calls, 1)

    def test_custom_state_roundtrip_and_progress_validation(self):
        schedules = ScheduleController({"x": StatefulCurve()})
        for update in (1, 2):
            schedules.prepare_update(update)
            schedules.complete_update(update)
        state = schedules.state_dict()
        restored = ScheduleController({"x": StatefulCurve()})
        restored.load_state_dict(state, completed_update=2)
        schedules.prepare_update(3)
        restored.prepare_update(3)
        self.assertEqual(restored["x"], schedules["x"])
        with self.assertRaises(ValueError):
            restored.validate_state_dict(state, completed_update=3)
        bad = deepcopy(state)
        bad["definitions"]["x"]["config"]["algorithm"] = 2
        with self.assertRaises(ValueError):
            restored.validate_state_dict(bad)

    def test_nonfinite_and_duplicate_bindings_fail(self):
        curve = Constant(1)
        with self.assertRaisesRegex(ValueError, "separate"):
            ScheduleController({"a": curve, "b": curve})
        schedules = ScheduleController({"x": LinearWarmup(1, 1e10, 2)})
        schedules.tensor("x", dtype=torch.float16)
        schedules.prepare_update(1)
        schedules.complete_update(1)
        before = schedules.state_dict()
        with self.assertRaisesRegex(ValueError, "representable"):
            schedules.prepare_update(2)
        self.assertEqual(schedules.state_dict(), before)

    def test_validation_can_materialize_a_tensor_without_changing_state(self):
        schedules = ScheduleController({"x": Constant(.5)})
        before = schedules.state_dict()
        with schedules.read_only():
            tensor = schedules.tensor("x")
        self.assertEqual(float(tensor), .5)
        self.assertEqual(before, schedules.state_dict())

    def test_memory_estimation_never_caches_fake_tensors_or_changes_schedules(self):
        from deepopt import Trainer
        from deepopt.memory import estimate_training_memory
        from torch._subclasses.fake_tensor import FakeTensor
        for prebound in (False, True):
            schedules = ScheduleController({"x": Constant(.5)})
            if prebound:
                schedules.tensor("x")
            model = torch.nn.Linear(1, 1)
            def step(model, batch):
                return model(batch).square().mean() * schedules.tensor("x", device=model.weight.device)
            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), step_fn=step, schedules=schedules)
            before = schedules.state_dict()
            estimate = estimate_training_memory(trainer, torch.ones(2, 1))
            self.assertGreater(estimate.peak_bytes_by_device["cpu"], 0)
            self.assertEqual(schedules.state_dict(), before)
            self.assertFalse(isinstance(schedules.tensor("x"), FakeTensor))
            trainer.train_epoch([torch.ones(2, 1)])


if __name__ == "__main__":
    unittest.main()
