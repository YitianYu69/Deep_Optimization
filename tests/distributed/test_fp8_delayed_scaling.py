"""Registered delayed histories are never overwritten by ordinary DDP broadcasts."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from torch import distributed as dist

if __package__:
    from .launcher import run_torchrun
else:
    from launcher import run_torchrun


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class DelayedScalingDistributedTests(unittest.TestCase):
    def test_pooling_compressed_backward_and_exact_resume_on_two_ranks(self):
        environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
        environment["OMP_NUM_THREADS"] = "1"
        for kind in ("average_pool", "maximum_pool", "power_pool", "pool_unpool"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                result = run_torchrun([str(Path(__file__).with_name("_delayed_scaling_worker.py")), directory, "auto", kind],
                                      env=environment, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                self.assertEqual(rows, [{"synchronize": value, "rank_local_checkpoint": True, "exact_next_update": True,
                                         "backward_auto": True, "activation": kind} for value in (True, False)])

    def test_recentered_layernorm_delayed_gradients_and_resume_on_two_ranks(self):
        environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
        environment["OMP_NUM_THREADS"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(Path(__file__).with_name("_delayed_scaling_worker.py")),
                                  directory, "auto", "layer_norm", "recenter"], env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(rows, [{"synchronize": value, "rank_local_checkpoint": True, "exact_next_update": True,
                                 "backward_auto": True, "activation": "layer_norm", "norm_xhat_recenter": True}
                                for value in (True, False)])

    def test_extended_activations_compression_delayed_gradients_and_exact_resume(self):
        environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
        environment["OMP_NUM_THREADS"] = "1"
        for activation in ("sigmoid", "silu", "gelu", "tanh", "relu6", "layer_norm"):
            with self.subTest(activation=activation), tempfile.TemporaryDirectory() as directory:
                result = run_torchrun([str(Path(__file__).with_name("_delayed_scaling_worker.py")),
                                      directory, "auto", activation], env=environment, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                self.assertEqual(rows, [{"synchronize": value, "rank_local_checkpoint": True, "exact_next_update": True,
                                         "backward_auto": True, "activation": activation} for value in (True, False)])

    def test_planned_gradient_histories_and_resume_on_two_installed_library_ranks(self):
        environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
        environment["OMP_NUM_THREADS"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(Path(__file__).with_name("_delayed_scaling_worker.py")), directory, "auto"],
                                  env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(rows, [{"synchronize": value, "rank_local_checkpoint": True, "exact_next_update": True,
                                 "backward_auto": True} for value in (True, False)])

    def test_synchronized_and_local_histories_restore_exactly_on_two_ranks(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(Path(__file__).with_name("_delayed_scaling_worker.py")), directory],
                                  env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"}, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(records, [{"synchronize": value, "rank_local_checkpoint": True, "exact_next_update": True}
                                   for value in (True, False)])
