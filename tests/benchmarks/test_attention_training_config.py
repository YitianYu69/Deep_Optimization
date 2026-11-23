"""The training benchmark respects supplied recipes and requires actual replay."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.benchmark_vit_attention_stability import parse_args, resolve_run, check_training_replay, training_device


class AttentionTrainingConfigTests(unittest.TestCase):
    def args(self, *options):
        return parse_args(["--output", "unused", "--optimizer", "torch.optim:SGD", *options])

    def config(self, folder):
        path = Path(folder) / "recipe.toml"
        path.write_text('''name = "test"
seed = 9
[training]
epochs = 7
batch_size = 16
learning_rate = 0.0001
[runtime]
device = "cuda"
precision = "bf16"
execution_mode = "max_speed"
deterministic = false
fill_uninitialized_memory = true
[runtime.attention]
backend = "native"
[runtime.activation_compression]
enabled = true
group_size = 1024
bits = 2
attention_bits = 8
norm_xhat_recenter = false
[runtime.fp8]
enabled = false
[loader]
num_workers = 0
drop_last = true
''')
        return path

    def test_inherits_recipe_before_any_model_or_loader_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            args, snapshot = resolve_run(self.args("--config", str(self.config(folder))))
        self.assertEqual((args.seed, args.batch_size, args.workers, args.epochs), (9, 16, 0, 7))
        self.assertEqual((args.backend, args.group_size, args.attention_bits, args.fp8), ("native", 1024, 8, False))
        self.assertEqual(args.optimizer_kwargs, {"lr": .0001})
        self.assertTrue(snapshot["runtime"]["fill_uninitialized_memory"])
        self.assertFalse(snapshot["runtime"]["deterministic"])
        self.assertFalse(snapshot["loader"]["persistent_workers"])
        self.assertIsNone(snapshot["loader"]["prefetch_factor"])

    def test_explicit_overrides_preserve_other_recipe_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            args, snapshot = resolve_run(self.args("--config", str(self.config(folder)), "--backend", "flash_attn_2",
                "--group-size", "1024", "--attention-bits", "4", "--batch-size", "32", "--fp8",
                "--optimizer-kwargs", '{"lr":0.0002,"momentum":0.9}', "--no-fill-uninitialized-memory"))
        self.assertEqual((args.backend, args.group_size, args.attention_bits, args.batch_size), ("flash_attn_2", 1024, 4, 32))
        self.assertTrue(args.fp8)
        self.assertFalse(snapshot["runtime"]["fill_uninitialized_memory"])
        self.assertFalse(snapshot["runtime"]["deterministic"])
        self.assertEqual(snapshot["training"]["learning_rate"], .0002)
        self.assertEqual(args.optimizer_kwargs["momentum"], .9)

    def test_zero_attention_bits_is_explicit_exclusion(self):
        args, snapshot = resolve_run(self.args("--learning-rate", ".0001", "--attention-bits", "0", "--workers", "0"))
        self.assertEqual(args.attention_bits, 0)
        self.assertTrue(snapshot["runtime"]["activation_compression"]["enabled"])

    def test_no_compression_keeps_attention_saves_dense(self):
        args, snapshot = resolve_run(self.args("--learning-rate", ".0001", "--no-compression"))
        self.assertEqual(args.attention_bits, 0)
        self.assertFalse(snapshot["runtime"]["activation_compression"]["enabled"])

    def test_rate_is_required_and_conflicting_explicit_rates_fail(self):
        with self.assertRaisesRegex(ValueError, "Supply"):
            resolve_run(self.args())
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_run(self.args("--learning-rate", ".0001", "--optimizer-kwargs", '{"lr":0.0004}'))

    def test_replay_required_but_evaluation_mode_fallbacks_allowed(self):
        class Capture:
            statistics = dict(forward_replays=0, eager_mode_fallbacks=0,
                              eager_autocast_fallbacks=0, eager_memory_policy_fallbacks=0)
            fallback_diagnostics = {"eager_autocast_fallbacks": {"fp8": {"captured": False, "current": None}}}
        captured = Capture()
        before = captured.statistics.copy()
        with self.assertRaisesRegex(RuntimeError, "did not replay.*fp8"):
            check_training_replay(captured, before)
        captured.statistics = {**before, "forward_replays": 1, "eager_mode_fallbacks": 3}
        self.assertEqual(check_training_replay(captured, before)["forward_replays"], 1)
        captured.statistics["eager_autocast_fallbacks"] = 1
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            check_training_replay(captured, before)
        self.assertIsNone(check_training_replay(None, None))

    def test_unspecified_cuda_index_uses_current_device(self):
        with patch("torch.cuda.current_device", return_value=2):
            self.assertEqual(str(training_device("cuda")), "cuda:2")
            self.assertEqual(str(training_device("auto")), "cuda:2")
        self.assertEqual(str(training_device("cuda:1")), "cuda:1")
        with self.assertRaisesRegex(ValueError, "requires"):
            training_device("cpu")
