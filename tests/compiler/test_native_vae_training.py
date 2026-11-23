"""Compressed BF16 VAE training lifetimes, overlapping graphs and exact resume."""

from contextlib import nullcontext
from dataclasses import asdict
import gc
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

import deepopt
from benchmarks.analysis.measurement import SavedStorageLifetimes
from deepopt.compiler import compilation_report
from deepopt.config import config_from_dict
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from examples.vae import train_vae as recipe
from tests.application.test_vae_recipe import cpu_config, tiny_images
from tests.compiler import test_native_vae as native_fixture


class NativeVAETrainingTests(unittest.TestCase):
    device = "cpu"
    setUp = native_fixture.NativeVAETests.setUp
    pair = native_fixture.NativeVAETests.pair
    forward_loss = native_fixture.NativeVAETests.forward_loss
    assert_native = native_fixture.NativeVAETests.assert_native

    def tearDown(self):
        deepopt.reset_config()
        deepopt.shutdown_distributed()

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for value, control in zip(actual, expected):
                self.assert_tree_equal(value, control)
        else:
            self.assertEqual(actual, expected)

    def test_fifty_updates_release_saved_payloads_and_reuse_specialization(self):
        model, reference, compression = self.pair()
        del reference
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
        images = torch.randn(1, 3, 16, 16, device=self.device)
        tracker = SavedStorageLifetimes(tuple(model.parameters()) + tuple(model.buffers()))
        graphs, optimizer_shapes, allocated = [], [], []
        for update in range(50):
            optimizer.zero_grad(set_to_none=True)
            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
                outputs, loss, terms = self.forward_loss(model, images, compression, seed=1000 + update)
            self.assertGreater(tracker.snapshot()["live_nonpersistent_bytes"], 0)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, terms
            gc.collect()
            self.synchronize()
            # The hook ledger owns only metadata once autograd releases its
            # invocation tokens. Model/optimizer storage is counted separately.
            self.assertEqual(tracker.snapshot()["live_unique_bytes"], 0, f"update {update}")
            graphs.append(len(compilation_report(model).graphs))
            optimizer_shapes.append(tuple(tuple(value.shape) for state in optimizer.state.values()
                                          for value in state.values() if isinstance(value, torch.Tensor)))
            if self.device == "cuda":
                allocated.append(torch.cuda.memory_allocated())
        self.assertEqual(len(set(graphs[2:])), 1)
        self.assertEqual(len(set(optimizer_shapes)), 1)
        self.assert_native(model)
        if allocated:
            # Warmed model/optimizer allocations must not accumulate between
            # updates. This is allocated memory, independent of allocator cache.
            self.assertLessEqual(max(allocated[-10:]), max(allocated[2:5]))

    def test_interleaved_retained_graphs_accumulate_matching_eager_gradients(self):
        model, reference, compression = self.pair()
        inputs = [torch.randn(1, 3, 16, 16, device=self.device, requires_grad=True) for _ in range(2)]
        controls = [value.detach().clone().requires_grad_() for value in inputs]
        tracker = SavedStorageLifetimes(tuple(model.parameters()) + tuple(model.buffers()))
        losses, expected = [], []
        for index, (value, control) in enumerate(zip(inputs, controls)):
            expected.append(self.forward_loss(reference, control, compression, seed=2000 + index)[1])
            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
                losses.append(self.forward_loss(model, value, compression, seed=2000 + index)[1])
        losses[0].backward(retain_graph=True)
        expected[0].backward(retain_graph=True)
        self.assertGreater(tracker.snapshot()["live_nonpersistent_bytes"], 0)
        for (name, value), (_, control) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(value.grad, control.grad, rtol=0, atol=0, msg=name)
        # Keep the first contribution, then consume both outstanding graphs.
        # The first graph contributes twice and the second contributes once.
        (losses[0] + losses[1]).backward()
        (expected[0] + expected[1]).backward()
        for value, control in zip(inputs, controls):
            torch.testing.assert_close(value.grad, control.grad, rtol=0, atol=0)
        for (name, value), (_, control) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(value.grad, control.grad, rtol=0, atol=0, msg=name)
        gc.collect()
        self.synchronize()
        self.assertEqual(tracker.snapshot()["live_unique_bytes"], 0)
        self.assert_native(model)

    def test_public_mid_epoch_resume_preserves_latent_rng_optimizer_beta_and_ema(self):
        raw = asdict(cpu_config())
        # The public core deliberately supports CPU fp32 only. Its reference
        # step explicitly enters BF16 for the model; CUDA uses real runtime AMP.
        runtime_precision = "bf16" if self.device == "cuda" else "fp32"
        raw["runtime"].update(device=self.device, precision=runtime_precision, checks=False,
                               execution_mode="max_speed", fp8={"enabled": False})
        raw["runtime"]["activation_compression"] = dict(native_fixture.COMPRESSION)
        raw["runtime"]["compile"] = dict(enabled=True, backend="deepopt", fullgraph=True,
                                           dynamic=False, options={"require_full_aot": True,
                                                                   "use_native_autodiff": True})
        config = config_from_dict(raw)
        trainers = []
        bf16_scopes = []
        make_step = recipe.make_vae_step

        def step_factory(schedules):
            original = make_step(schedules)

            def step(model, images):
                scope = (deepopt.autocast("cpu", dtype=torch.bfloat16, fp8=False,
                                         activation_compression=native_fixture.COMPRESSION, config={})
                         if self.device == "cpu" else nullcontext())
                with scope:
                    bf16_scopes.append(torch.is_autocast_enabled(self.device))
                    return original(model, images)

            return step

        def trainer(*args, **kwargs):
            current = deepopt.Trainer(*args, **kwargs)
            trainers.append(current)
            return current

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(recipe, "image_datasets", side_effect=tiny_images), \
                patch.object(recipe, "make_vae_step", side_effect=step_factory), \
                patch.object(recipe, "Trainer", side_effect=trainer):
            directory = Path(directory)
            context = deepopt.initialize_distributed(self.device)
            recipe.fit_vae(config, context, directory / "uninterrupted", progress_bar=False, log_every_steps=0)
            expected = read_checkpoint(directory / "uninterrupted" / "last.pt")
            step = directory / "uninterrupted" / "last_step.pt"
            snapshot = read_checkpoint(step)
            self.assertEqual(snapshot["state"]["optimizer_steps"], 2)
            self.assertIsNotNone(snapshot["train_loader"])
            self.assertIsNotNone(snapshot["schedules"])
            self.assertTrue(snapshot["optimizer"]["state"])
            resumed = directory / "resumed"
            resumed.mkdir()
            resumed_step = resumed / "last_step.pt"
            shutil.copyfile(step, resumed_step)
            recipe.fit_vae(config, context, resumed, resume=resumed_step,
                           progress_bar=False, log_every_steps=0)
            actual = read_checkpoint(directory / "resumed" / "last.pt")
        self.assertEqual(actual["state"]["optimizer_steps"], 3)
        self.assertTrue(bf16_scopes and all(bf16_scopes))
        for key in ("state", "model", "optimizer", "scheduler", "schedules", "ema", "rng", "generators"):
            with self.subTest(component=key):
                self.assert_tree_equal(actual[key], expected[key])
        self.assertEqual(len(actual["history"]), len(expected["history"]))
        for row, control in zip(actual["history"], expected["history"]):
            # Wall time is measured again after resume; training values and
            # sample/update accounting must still match the uninterrupted run.
            def numerical_history(value):
                return {**value, "train": {key: item for key, item in value["train"].items()
                                            if key not in {"seconds", "samples_per_second"}}}
            self.assert_tree_equal(numerical_history(row), numerical_history(control))
        for current in trainers:
            self.assertEqual(current.precision, runtime_precision)
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in current.model.parameters()))
            self.assert_native(current.model)


@unittest.skipUnless(torch.cuda.is_available(), "Whole VAE BF16 training qualification requires CUDA")
class NativeVAETrainingCudaTests(NativeVAETrainingTests):
    device = "cuda"

    def setUp(self):
        super().setUp()
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("BF16 qualification requires compute capability >= 8")


if __name__ == "__main__":
    unittest.main()
