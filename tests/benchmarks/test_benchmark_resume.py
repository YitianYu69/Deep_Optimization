"""Projection checkpoints survive interruption and reject incompatible results."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


linear = load("resume_linear", "benchmarks/fp8/benchmark_fp8_linear_compare.py")
unpack = load("resume_unpack", "benchmarks/activation_compression/benchmark_activation_unpack.py")


class BenchmarkResumeTests(unittest.TestCase):
    def test_codec_budgeted_rows_reuse_only_with_valid_actual_counts(self):
        args = unpack.parser().parse_args(["--execution", "eager", "--repeats", "5", "--iterations", "50", "--time-budget", "1"])
        sampling = {"requested_warmup": args.warmup, "requested_iterations": 50, "requested_repeats": 5,
                    "actual_warmup": 0, "actual_iterations_per_repeat": [1], "actual_repeats": 1,
                    "time_budget_seconds": 1.0, "budget_exhausted": True}
        row = {"exact": True, "timing": {"eager": {"wall_ms_median": 1., "wall_ms_samples": [1.], "sampling": sampling}}}
        self.assertTrue(unpack.reusable_row(row, args))
        for changes in ({"actual_iterations_per_repeat": [51]}, {"requested_iterations": 49},
                        {"requested_warmup": args.warmup + 1}, {"actual_repeats": 2}):
            changed = deepcopy(row)
            changed["timing"]["eager"]["sampling"].update(changes)
            self.assertFalse(unpack.reusable_row(changed, args))
        args.time_budget = 0
        changed = deepcopy(row)
        measured = changed["timing"]["eager"]
        measured["wall_ms_samples"] = [1.] * 5
        measured["sampling"].update(time_budget_seconds=0, actual_repeats=5,
                                     actual_iterations_per_repeat=[1] * 5, actual_warmup=args.warmup)
        self.assertFalse(unpack.reusable_row(changed, args))
        measured["sampling"]["actual_iterations_per_repeat"] = [50] * 5
        self.assertTrue(unpack.reusable_row(changed, args))

    def test_completed_linear_projections_survive_interruption_and_environment_changes_retry(self):
        args = linear.parser().parse_args(["--resume", "--profile-steps", "0", "--implementations", "bf16",
                                           "--projections", "attention", "fc1"])
        args.num_stages = 3
        settings = vars(args).copy()
        settings.pop("output")
        env = {"cuda_available": True, "capability": [9, 0], "packages": {}, "torch": "same", "gpu": "same"}
        def finish(session, args, directory, row, checkpoint):
            row["status"] = "ok"
            checkpoint()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = {"settings": settings, "directory": temporary, "run_id": "first", "resume_key": "key",
                       "implementation": "bf16", "math_policy": "amp", "batch": 1}
            with patch.object(linear, "environment", return_value=env), patch.object(linear, "configure"), \
                 patch.object(linear, "Session", return_value=object()) as constructor, patch.object(linear, "run_case", side_effect=finish):
                linear.worker(payload)
                self.assertEqual(constructor.call_count, 2)
                constructor.reset_mock()
                linear.worker({**payload, "run_id": "second"})
                constructor.assert_not_called()
                previous = json.loads((directory / "result.json").read_text())
                self.assertTrue(all(row["resumed"] for row in previous["cases"]))
                previous["cases"][1]["status"] = "running"
                linear.write_json(directory / "result.json", previous)
                with patch.object(linear, "run_case", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                    linear.worker({**payload, "run_id": "interrupted"})
                partial = json.loads((directory / "result.json").read_text())
                self.assertEqual([row["status"] for row in partial["cases"]], ["ok", "running"])
                linear.worker({**payload, "run_id": "finished"})
                constructor.reset_mock()
                different = deepcopy(env)
                different["torch"] = "changed"
                with patch.object(linear, "environment", return_value=different):
                    linear.worker({**payload, "run_id": "changed"})
                self.assertEqual(constructor.call_count, 2)

    def test_unpack_checkpoint_is_atomic_and_reuse_requires_all_requested_samples(self):
        args = unpack.parser().parse_args(["--repeats", "2"])
        measured = {"wall_ms_median": 1.0, "wall_ms_samples": [1., 1.]}
        row = {"exact": True, "timing": {"eager": measured, "graph": measured}}
        self.assertTrue(unpack.reusable_row(row, args))
        changed = deepcopy(row)
        changed["timing"]["graph"]["wall_ms_samples"] = [1.]
        self.assertFalse(unpack.reusable_row(changed, args))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            unpack.write_checkpoint(path, {"results": [row]})
            with patch.object(Path, "replace", side_effect=OSError("interrupted")), self.assertRaises(OSError):
                unpack.write_checkpoint(path, {"results": []})
            self.assertEqual(json.loads(path.read_text())["results"], [row])


if __name__ == "__main__":
    unittest.main()
