"""Matched batch/state, independent codec, and actual optimizer-update checks."""

from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import os
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import subprocess
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import deepopt
from deepopt.config import reset_config
from analysis import diagnose_training_stability as diagnostic
from analysis import stability_oracle as oracle
from deepopt.opt.quantization import activation as codec


class CounterSGD(torch.optim.Optimizer):
    """An actual mutable counter absent from Optimizer.state_dict()."""
    def __init__(self, params, lr=.01):
        super().__init__(params, {"lr": lr})
        self._global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._global_step += 1
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-group["lr"] * self._global_step)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.in_proj = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.out = nn.Linear(16, 3)

    def forward(self, x):
        return self.out(self.act(self.in_proj(self.norm(x))))


def build_model():
    return TinyModel()


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
        value = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(value.transpose(1, 2).reshape(b, t, 8))


class TinyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm, self.attn, self.head = nn.LayerNorm(8), TinyAttention(), nn.Linear(8, 3)

    def forward(self, x):
        return self.head(self.attn(self.norm(x)).mean(1))


def build_attention_model():
    return TinyAttentionModel()


class StabilityDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        reset_config()
        self.addCleanup(reset_config)

    def test_bounded_independent_codec_matches_all_bits_tails_and_tiles(self):
        for bits in (1, 2, 4, 8):
            for group, block in ((17, None), (17, (1, 17)), (12, (3, 4)), (1, None)):
                x = torch.randn(5, 66)[:, ::2]
                x[0].fill_(.125)
                a = codec.quantize_pack_activation(x, group, block, bits=bits)
                previous = oracle.MAX_ELEMENTS
                oracle.MAX_ELEMENTS = 20
                try:
                    b = oracle.pack(x, group, block, bits=bits)
                    for key in ("data", "step", "minimum"):
                        torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
                    for transpose in (False, True):
                        torch.testing.assert_close(oracle.unpack(b, torch.float32, transpose),
                            codec.dequantize_unpack_activation(a, torch.float32, transpose), rtol=0, atol=0)
                finally:
                    oracle.MAX_ELEMENTS = previous

    def test_oracle_restores_existing_and_lazy_aliases_after_failure(self):
        original = codec.quantize_pack_activation
        lazy = ModuleType("deepopt.opt._stability_test_lazy")
        try:
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with oracle.oracle_scope():
                    self.assertIs(codec.quantize_pack_activation, oracle.pack)
                    lazy.pack = codec.quantize_pack_activation
                    sys.modules[lazy.__name__] = lazy
                    raise RuntimeError("intentional")
            self.assertIs(codec.quantize_pack_activation, original)
            self.assertIs(lazy.pack, original)
        finally:
            sys.modules.pop(lazy.__name__, None)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA scalar-division regression")
    def test_cuda_oracle_uses_precise_division_at_bf16_rounding_boundaries(self):
        # A CUDA Python-scalar division can produce 7*(1/3) instead of 7/3.
        # The resulting one-ULP scale change moves the middle code from 2 to 1.
        x = torch.tensor([[0., 3.5, 7.]], dtype=torch.bfloat16, device="cuda")
        packed = oracle.pack(x, 3, bits=2)
        expected_step = torch.tensor([7. / 3.], dtype=torch.float32, device="cuda")
        torch.testing.assert_close(packed.step, expected_step, rtol=0, atol=0)
        self.assertEqual(int(packed.data[0, 0]), (2 << 2) | (3 << 4))
        production = codec.quantize_pack_activation(x, 3, bits=2)
        torch.testing.assert_close(packed.data, production.data, rtol=0, atol=0)

    def test_snapshot_captures_real_batch_and_extra_counter_without_refetch(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, trainer = self.capture(Path(directory) / "snapshot.pt")
        self.assertEqual(packet["completed_updates"], 1)
        self.assertEqual(packet["optimizer"]["extra"]["_global_step"], 1)
        self.assertFalse(packet["fill_uninitialized_memory"])
        self.assertEqual(trainer.optimizer._global_step, 3)
        self.assertEqual(packet["batch"][1].tolist(), [2, 0])
        rebuilt = diagnostic._factory_model(packet, __name__ + ":build_model", {}, torch.device("cpu"))
        optimizer = diagnostic._optimizer(rebuilt, packet)
        self.assertEqual(optimizer._global_step, 1)

    def capture(self, path, attention=False, resumable=False):
        model = (TinyAttentionModel() if attention else TinyModel()).bfloat16()
        trainer = deepopt.Trainer(model, CounterSGD(model.parameters()), device="cpu", precision="fp32",
            loss_fn=nn.CrossEntropyLoss(), execution_mode="max_speed", fp8=False,
            config={"runtime": {"activation_compression": {"enabled": True, "bits": 2, "group_size": 1024}}})
        diagnostic._skip_compression(trainer.model.attn.proj if attention else trainer.model.out, torch.device("cpu"))
        dataset = TensorDataset(torch.randn((6, 5, 8) if attention else (6, 8), dtype=torch.bfloat16), torch.arange(6) % 3)
        loader = DataLoader(dataset, batch_size=2)
        if resumable:
            from deepopt.data import build_loaders
            loader, _, _ = build_loaders(dataset, None, batch_size=2, seed=14, num_workers=0, resumable=True)
        wrapped = diagnostic.SnapshotLoader(loader, trainer, path, before_update=2,
            dense_modules=["attn.proj" if attention else "out"])
        trainer.fit(wrapped, epochs=1, progress_bar=False)
        return torch.load(path, weights_only=False), trainer

    def test_end_to_end_replay_separates_approximation_and_oracle_error(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, _ = self.capture(Path(directory) / "snapshot.pt")
            before = deepcopy(packet["optimizer"])
            args = Namespace(device="cpu", output=Path(directory) / "diagnostic", model_factory=__name__ + ":build_model",
                model_kwargs={}, optimizer=None, probe_modules=["norm", "in_proj", "act"],
                max_probe_bytes=1 << 20, check_fill=True, graph=False, rtol=.03, atol=1e-5)
            packet["deterministic"] = False
            rng = diagnostic._rng(torch.device("cpu"))
            deterministic = torch.are_deterministic_algorithms_enabled()
            fill = torch.utils.deterministic.fill_uninitialized_memory
            report = diagnostic.run(packet, args)
            restored = diagnostic._rng(torch.device("cpu"))
            torch.testing.assert_close(restored["torch"], rng["torch"], rtol=0, atol=0)
            self.assertEqual(restored["python"], rng["python"])
            torch.testing.assert_close(torch.from_numpy(restored["numpy"][1]), torch.from_numpy(rng["numpy"][1]), rtol=0, atol=0)
            self.assertEqual(restored["numpy"][2:], rng["numpy"][2:])
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
            self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, fill)
            self.assertTrue(all(row["status"] == "ok" for row in report["cases"].values()), report["cases"])
            self.assertTrue(all(row["optimizer_update"] == "matched_captured_state" for row in report["cases"].values()))
            self.assertFalse(report["cases"]["production"]["deterministic"])
            self.assertTrue(report["cases"]["production_fill"]["effective_fill"])
            self.assertTrue(report["cases"]["production_no_fill"]["deterministic"])
            self.assertFalse(report["cases"]["production_no_fill"]["effective_fill"])
            self.assertIn("production_vs_native", report["comparisons"])
            self.assertIn("determinism_sensitivity", report["comparisons"])
            self.assertTrue(report["comparisons"]["native_repeatability"]["gradients"]["passed"])
            self.assertTrue(report["comparisons"]["production_vs_oracle"]["gradients"]["passed"])
            self.assertTrue(report["comparisons"]["production_vs_oracle"]["updates"]["passed"])
            self.assertFalse(report["comparisons"]["expected_approximation"]["gradients"]["passed"])
            self.assertEqual(set(report["local_checks"]), {"norm", "in_proj", "act"})
            self.assertTrue(all(row["comparison"]["passed"] for row in report["local_checks"].values()))
            self.assertFalse(report["local_checks"]["in_proj"]["production_vs_native"]["passed"])
            self.assertEqual(report["diagnostic_revision"], 2)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["codec_audits"])
            self.assertTrue(all(row.get("differing_words", row.get("differing_values")) == 0 for row in report["codec_audits"]))
            self.assertEqual(packet["optimizer"]["extra"], before["extra"])
            self.assertTrue((args.output / "report.json").is_file())

    def test_missing_optimizer_state_is_not_replaced_with_sgd(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, _ = self.capture(Path(directory) / "snapshot.pt")
            packet["optimizer"]["unsupported"]["custom_state"] = "opaque object"
            model = diagnostic._factory_model(packet, __name__ + ":build_model", {}, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "Unreconstructed optimizer"):
                diagnostic._optimizer(model, packet)

    def test_error_metrics_distinguish_zero_reference_and_nonfinite(self):
        self.assertEqual(diagnostic.tensor_error(torch.zeros(2), torch.zeros(2))["cosine"], 1.)
        row = diagnostic.tensor_error(torch.ones(2), torch.zeros(2))
        self.assertIsNone(row["relative_l2"])
        self.assertFalse(row["passed"])
        self.assertFalse(diagnostic.tensor_error(torch.tensor([float("nan")]), torch.ones(1))["finite"])

    def test_codec_audit_detects_corrupt_payload_instead_of_labeling_it_approximation(self):
        original = codec.quantize_pack_activation
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            data = result.data.clone()
            data[0, 0] ^= 1
            return codec.PackedActivation(data, result.step, result.minimum, result.shape,
                result.group_size, result.block_shape, result.bits)
        records = []
        with oracle._aliases({original: corrupt}), oracle.codec_audit_scope(records):
            codec.quantize_pack_activation(torch.randn(3, 8), 8, bits=2)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["differing_words"], 1)
        self.assertIs(codec.quantize_pack_activation, original)

    def test_codec_audit_detects_decoder_corruption_on_identical_payload(self):
        original = codec.dequantize_unpack_activation
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs).clone()
            result[0, 0] += 1
            return result
        packed = codec.quantize_pack_activation(torch.randn(3, 8), 8, bits=2)
        records = []
        with oracle._aliases({original: corrupt}), oracle.codec_audit_scope(records):
            codec.dequantize_unpack_activation(packed, torch.float32)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["direction"], "unpack")
        self.assertEqual(records[0]["differing_values"], 1)
        self.assertIs(codec.dequantize_unpack_activation, original)

    def test_unmatched_and_unobserved_probes_are_reported_as_skipped(self):
        records = {}
        with diagnostic._probes(TinyModel(), ["out", "missing"], records, 1 << 20):
            pass
        self.assertEqual(records["out"]["status"], "skipped")
        self.assertEqual(records["pattern:missing"]["status"], "skipped")

    def test_optimizer_groups_and_parameter_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, _ = self.capture(Path(directory) / "snapshot.pt")
        state = packet["optimizer"]["state"]
        original = state["param_groups"][0]
        names = packet["optimizer"]["groups"][0]["params"]
        state["param_groups"] = [{**original, "params": original["params"][:2]},
                                {**original, "params": original["params"][2:], "lr": .02}]
        packet["optimizer"]["groups"] = [{**original, "params": names[:2]}, {**original, "params": names[2:], "lr": .02}]
        model = diagnostic._factory_model(packet, __name__ + ":build_model", {}, torch.device("cpu"))
        rebuilt = diagnostic._optimizer(model, packet)
        self.assertEqual([group["lr"] for group in rebuilt.param_groups], [.01, .02])
        self.assertEqual(rebuilt._global_step, 1)

    def test_standalone_help_needs_no_gpu_or_analysis_package(self):
        with tempfile.TemporaryDirectory() as directory:
            for module in (diagnostic, oracle):
                shutil.copyfile(module.__file__, Path(directory) / Path(module.__file__).name)
            result = subprocess.run([sys.executable, "-S", str(Path(directory) / "diagnose_training_stability.py"), "--help"],
                cwd=directory, capture_output=True, text=True, timeout=30)
            version = subprocess.run([sys.executable, "-S", str(Path(directory) / "diagnose_training_stability.py"), "--version"],
                cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--snapshot", result.stdout)
        self.assertEqual(version.returncode, 0, version.stderr)
        self.assertIn("diagnostic revision 2", version.stdout)

    def test_mixed_runner_and_oracle_revisions_fail_before_any_replay(self):
        with patch.object(oracle, "ORACLE_REVISION", 1):
            with self.assertRaisesRegex(ValueError, "Mixed diagnostic files"):
                diagnostic._diagnostic_sources()

    def test_standalone_cli_keeps_bundled_oracle_ahead_of_stale_launch_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.capture(root / "snapshot.pt")
            runner, launch = root / "runner", root / "launch"
            runner.mkdir(); launch.mkdir()
            for module in (diagnostic, oracle):
                shutil.copyfile(module.__file__, runner / Path(module.__file__).name)
            (launch / "stability_oracle.py").write_text('raise RuntimeError("stale oracle from launch directory")\n')
            repository = Path(__file__).resolve().parents[2]
            env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(repository / "benchmarks"), str(repository)))}
            command = [sys.executable, str(runner / "diagnose_training_stability.py"),
                "--snapshot", str(root / "snapshot.pt"), "--device", "cpu",
                "--model-factory", __name__ + ":build_model", "--model-kwargs", "{}",
                "--probe-modules", "norm", "in_proj", "act", "--output", str(root / "report")]
            result = subprocess.run(command, cwd=launch, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import json
            report = json.loads((root / "report/report.json").read_text())
            self.assertEqual(report["diagnostic_revision"], 2)
            self.assertEqual(report["diagnostic_sources"]["oracle"]["path"], str((runner / "stability_oracle.py").resolve()))

    def test_sdpa_strided_qkv_are_audited_and_reconstructed_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, _ = self.capture(Path(directory) / "snapshot.pt", attention=True)
            args = Namespace(device="cpu", output=Path(directory) / "diagnostic", model_factory=__name__ + ":build_attention_model",
                model_kwargs={}, optimizer=None, probe_modules=["norm", "attn.qkv", "head"],
                max_probe_bytes=1 << 20, check_fill=True, graph=False, rtol=.03, atol=1e-5)
            report = diagnostic.run(packet, args)
            self.assertEqual(report["status"], "complete", report)
            self.assertTrue(any(row["api"] == "strided" and row["bits"] == 4 for row in report["codec_audits"]))
            self.assertTrue(all(row.get("differing_words", row.get("differing_values")) == 0 for row in report["codec_audits"]))
            self.assertTrue(report["comparisons"]["production_vs_oracle"]["gradients"]["passed"])

    def test_real_resumable_loader_is_delegated_without_losing_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, trainer = self.capture(Path(directory) / "snapshot.pt", resumable=True)
        self.assertEqual(packet["completed_updates"], 1)
        self.assertEqual(trainer.state.optimizer_steps, 3)


if __name__ == "__main__":
    unittest.main()
