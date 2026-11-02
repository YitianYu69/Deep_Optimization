from dataclasses import asdict
import subprocess
import sys
import unittest

from deepopt.config import ConfigError, config_from_dict
from deepopt.scheduling import Constant, LinearWarmup, LinearDecay, CosineDecay, ExponentialDecay, Piecewise, SequentialSchedule
from deepopt.scheduling.factory import config_for, schedule_from_config


class CurveWhiteBoxTests(unittest.TestCase):
    def test_endpoints_interpolation_and_saturation(self):
        for cls in (LinearWarmup, LinearDecay, CosineDecay):
            with self.subTest(cls=cls.__name__):
                curve = cls(start=2., end=4., duration=3)
                self.assertEqual([curve.value_at(k) for k in (1, 2, 3, 4, 100)], [2., 3., 4., 4., 4.])
                self.assertEqual(cls(start=2., end=4., duration=1).value_at(1), 4.)
        self.assertEqual(Constant(-2).value_at(100), -2.)
        self.assertEqual([ExponentialDecay(8, .5, duration=3).value_at(k) for k in range(1, 5)], [8., 4., 2., 2.])

    def test_piecewise_and_sequence_boundaries(self):
        curve = Piecewise([3, 5], [0., .5, 1.])
        self.assertEqual([curve.value_at(k) for k in range(1, 7)], [0., 0., .5, .5, 1., 1.])
        sequence = SequentialSchedule([LinearWarmup(0, 1, 2), LinearDecay(1, 0, 3)], [2, 3])
        self.assertEqual([sequence.value_at(k) for k in range(1, 7)], [0., 1., 1., .5, 0., 0.])
        self.assertEqual(config_for(schedule_from_config(config_for(sequence))), config_for(sequence))

    def test_invalid_parameters_and_positions(self):
        constructors = [lambda: Constant(float("nan")), lambda: Constant(True), lambda: Constant(1, "validation"),
                        lambda: LinearWarmup(0, 1, 0), lambda: LinearDecay(0, float("inf"), 3),
                        lambda: CosineDecay(0, 1, True), lambda: ExponentialDecay(1, 0),
                        lambda: Piecewise([2, 2], [0, 1, 2]), lambda: Piecewise([3, 2], [0, 1, 2]),
                        lambda: Piecewise([1], [0, 1]), lambda: Piecewise([2], [0]),
                        lambda: SequentialSchedule([], []),
                        lambda: SequentialSchedule([Constant(1, "epoch")], [2]),
                        lambda: SequentialSchedule([Constant(1)], [0])]
        for constructor in constructors:
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor()
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Constant(1).value_at(value)

    def test_config_is_normalized_strict_and_torch_free(self):
        raw = {"name": "scheduling", "seed": 1,
               "training": {"epochs": 2, "batch_size": 2, "learning_rate": .01},
               "schedules": {"beta": {"type": "linear_warmup", "start": 0, "end": 1, "duration": 10}}}
        config = config_from_dict(raw)
        self.assertEqual(config.schedules["beta"]["interval"], "update")
        self.assertEqual(config_from_dict(asdict(config)), config)
        for schedule in ({"type": "bogus"}, {"type": "constant", "value": 1, "target": "optimizer.lr"},
                         {"type": "constant", "value": 1, "interval": "validation"},
                         {"type": "linear_warmup", "start": 0, "end": 1, "duration": 0}):
            with self.subTest(schedule=schedule), self.assertRaises(ConfigError):
                config_from_dict({**raw, "schedules": {"x": schedule}})
        result = subprocess.run([sys.executable, "-c", "import sys; import deepopt.config; "
                                 "from deepopt.scheduling import LinearWarmup; assert 'torch' not in sys.modules"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
