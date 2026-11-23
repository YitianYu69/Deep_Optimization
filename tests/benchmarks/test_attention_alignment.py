"""Direction metrics, controlled reconstructions and portable resume checks."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from analysis import attention_comparison as cases
from analysis import benchmark_attention_alignment as runner
from analysis.diagnose_training_stability import gradient_alignment


class AttentionAlignmentTests(unittest.TestCase):
    def test_parallel_orthogonal_opposite_and_zero_directions(self):
        reference = torch.tensor([1., 0.])
        same = gradient_alignment(reference * 2, reference)
        self.assertAlmostEqual(same["cosine"], 1)
        self.assertAlmostEqual(same["parallel_scale"], 2)
        self.assertAlmostEqual(same["orthogonal_relative_l2"], 0)
        orthogonal = gradient_alignment(torch.tensor([0., 1.]), reference)
        self.assertAlmostEqual(orthogonal["angle_degrees"], 90)
        self.assertAlmostEqual(orthogonal["orthogonal_relative_l2"], 1)
        opposite = gradient_alignment(-reference, reference)
        self.assertAlmostEqual(opposite["angle_degrees"], 180)
        self.assertEqual(opposite["sign_disagreement_fraction"], 1)
        self.assertIsNone(gradient_alignment(torch.zeros(2), reference)["cosine"])
        self.assertIsNone(gradient_alignment(reference, torch.zeros(2))["cosine"])
        self.assertFalse(gradient_alignment(torch.tensor([float("nan")]), torch.ones(1))["finite"])

    def test_head_statistics_expose_cancellation_hidden_by_global_metrics(self):
        reference = torch.ones(1, 2, 3, 4)
        actual = reference.clone()
        actual[:, 1].neg_()
        result = gradient_alignment(actual, reference, head_axis=1)
        self.assertAlmostEqual(result["cosine"], 0)
        self.assertEqual(result["negative_head_fraction"], .5)
        self.assertEqual(result["token_directions"]["negative_fraction"], .5)

    def test_shared_math_matches_explicit_saved_operand_gradient_formula(self):
        torch.manual_seed(14)
        inputs = tuple(torch.randn(1, 2, 65, 8, requires_grad=True) for _ in range(3))
        dy = torch.randn_like(inputs[0])
        output = cases.attention_case(*inputs, backend="manual", bits=(4, 8, 4), group_size=1024, mode="shared")
        output.backward(dy)
        q, k, v = inputs
        qhat, khat, vhat = [cases.reconstruct(x, b, 1024) for x, b in zip(inputs, (4, 8, 4))]
        p = (q.detach() @ k.detach().transpose(-2, -1) * (8 ** -.5)).softmax(-1)
        dp = dy @ vhat.transpose(-2, -1)
        ds = p * (dp - (p * dp).sum(-1, keepdim=True))
        for x, expected in zip(inputs, (ds @ khat * (8 ** -.5), ds.transpose(-2, -1) @ qhat * (8 ** -.5), p.transpose(-2, -1) @ dy)):
            torch.testing.assert_close(x.grad, expected)

    def test_layout_and_legacy_rules_restore_logical_shape(self):
        q = torch.randn(2, 3, 23, 8, dtype=torch.bfloat16)
        for codec in ("current", "legacy_rules"):
            for layout in ("bshd", "bhsd", "saved"):
                for role in range(3):
                    decoded = cases.reconstruct(q, 4, 1024, layout=layout, role=role, codec=codec)
                    self.assertEqual(decoded.shape, q.shape)
                    self.assertEqual(decoded.dtype, q.dtype)
                    self.assertTrue(torch.isfinite(decoded).all())
                    self.assertEqual(decoded.stride(-1), 1)

    def test_workload_roundtrip_preserves_strides_offsets_and_qkv_aliasing(self):
        for layout in ("bhsd", "bshd", "interleaved"):
            with self.subTest(layout=layout):
                original = cases.synthetic_workload(2, 3, 5, 8, dtype=torch.bfloat16, layout=layout)
                packet = {**original, "qkv": tuple(x.clone() for x in original["qkv"]),
                          "grad_output": original["grad_output"].clone()}
                qkv, dy = cases.restore_workload(packet, "cpu")
                for actual, expected in zip(qkv, original["qkv"]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.storage_offset(), expected.storage_offset())
                torch.testing.assert_close(dy, original["grad_output"], rtol=0, atol=0)
                self.assertEqual(dy.stride(), original["grad_output"].stride())
                storage_count = len({x.untyped_storage().data_ptr() for x in qkv})
                self.assertEqual(storage_count, 1 if layout == "interleaved" else 3)

    def test_resume_keeps_alignment_when_only_timing_failed(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            argv = ["--device", "cpu", "--backends", "manual", "--bits", "4", "--modes", "shared",
                    "--layouts", "bshd", "--no-individual", "--batch-size", "1", "--heads", "2",
                    "--sequence-length", "5", "--head-dim", "8", "--iterations", "0", "--output", directory]
            def fail_graph(*args, on_progress, **kwargs):
                result = {"status": "error", "step": {"median": 12.},
                          "stages": {"graph": {"status": "error", "reason": "capture failed"}}}
                on_progress(result)
                return result
            with patch.object(runner, "_timing", side_effect=fail_graph):
                self.assertEqual(runner.main(argv), 1)
            saved = json.loads((Path(directory) / "report.json").read_text())
            row = next(iter(saved["results"].values()))
            self.assertEqual(row["alignment_status"], "ok")
            self.assertEqual(row["timing"]["step"]["median"], 12.)
            self.assertIn("versus_dense", row)
            def retry(*args, existing, on_progress, **kwargs):
                self.assertEqual(existing["step"]["median"], 12.)
                return {"status": "disabled"}
            with patch.object(runner, "_step", side_effect=AssertionError("alignment repeated")), \
                    patch.object(runner, "_timing", side_effect=retry):
                self.assertEqual(runner.main([*argv, "--resume"]), 0)

    def test_real_runner_resumes_completed_cases_without_reexecution(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            argv = ["--device", "cpu", "--backends", "manual", "--bits", "4", "--modes", "shared",
                    "--layouts", "bshd", "--no-individual", "--batch-size", "1", "--heads", "2",
                    "--sequence-length", "5", "--head-dim", "8", "--group-size", "1024", "--iterations", "0",
                    "--output", directory]
            self.assertEqual(runner.main(argv), 0)
            with patch.object(runner, "_step", side_effect=AssertionError("completed case repeated")):
                self.assertEqual(runner.main([*argv, "--resume"]), 0)
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertTrue((Path(directory) / "summary.csv").is_file())
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                runner.main([*argv, "--seed", "99", "--resume"])

    def test_snapshot_replay_extracts_actual_attention_and_custom_optimizer_updates(self):
        from tests.benchmarks.test_training_stability import StabilityDiagnosticTests
        fixture = StabilityDiagnosticTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            fixture.capture(root / "snapshot.pt", attention=True)
            args = ["--device", "cpu", "--backends", "manual", "--bits", "4", "--modes", "shared",
                    "--layouts", "bshd", "--no-individual", "--heads", "2", "--iterations", "0",
                    "--snapshot", str(root / "snapshot.pt"), "--model-factory",
                    "tests.benchmarks.test_training_stability:build_attention_model", "--model-kwargs", "{}",
                    "--output", str(root / "report")]
            self.assertEqual(runner.main(args), 0)
            report = json.loads((root / "report/report.json").read_text())
            replay = report["model_replays"]["manual-qkv4"]
            self.assertEqual(replay["optimizer_update"], "matched_captured_state")
            self.assertTrue(replay["optimizer_updates"]["parameters"])
            self.assertIn("attn.qkv.weight", replay["qkv_parameter_heads"])
            self.assertTrue(list((root / "report/workloads").glob("*.pt")))
            self.assertTrue(replay["trace"])

    def test_custom_optimizer_extra_counters_are_serialized_for_long_run_resume(self):
        from analysis.benchmark_vit_attention_stability import optimizer_extra
        from tests.benchmarks.test_training_stability import CounterSGD
        model = torch.nn.Linear(2, 1)
        optimizer = CounterSGD(model.parameters())
        optimizer._global_step = 37
        self.assertEqual(optimizer_extra(optimizer, model)["_global_step"], 37)


if __name__ == "__main__":
    unittest.main()
