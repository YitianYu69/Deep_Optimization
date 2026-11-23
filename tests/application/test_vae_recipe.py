"""Exercise the VAE's direct Trainer.fit path, image metrics, and checkpoints."""

from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import torch

import deepopt
from deepopt.config import config_from_dict
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.scheduling import ScheduleController
from examples.vae import train_vae as recipe
from examples.vae.mbconv_vae import MBConvVAE_V2


ROOT = Path(__file__).resolve().parents[2]


def cpu_config():
    raw = tomllib.loads((ROOT / "examples/vae/imagenet.toml").read_text())
    raw["training"].update(epochs=1, batch_size=2, checkpoint_every_steps=2)
    raw["runtime"].update(device="cpu", precision="fp32")
    raw["runtime"]["activation_compression"]["enabled"] = False
    raw["loader"].update(num_workers=0, persistent_workers=False, pin_memory=False)
    raw["application"]["model"].update(
        base_channels=4, max_channels=16, blocks_per_stage=1, latent_channels=2
    )
    # Tiny CPU checks exercise real PSNR/SSIM without downloading LPIPS weights.
    raw["application"]["metrics"]["lpips"] = False
    raw["logging"].update(level="ERROR", file_level="ERROR")
    return config_from_dict(raw)


def tiny_images(options, *, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(8, 3, 16, 16, generator=generator) * 2 - 1
    return images[:5], images[5:]


class VAERecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        deepopt.reset_config()
        deepopt.shutdown_distributed()

    def test_l1_kl_objective_has_expected_values_and_gradients(self):
        reconstruction = torch.ones(2, 3, 2, 2, requires_grad=True)
        target = torch.zeros_like(reconstruction)
        mu = torch.ones(2, 2, 2, 2, requires_grad=True)
        logvar = torch.zeros_like(mu, requires_grad=True)
        loss, metrics = recipe.vae_loss(reconstruction, target, mu, logvar, beta=0.25)
        self.assertEqual(metrics["l1"].item(), 1.0)
        self.assertEqual(metrics["kl"].item(), 4.0)
        self.assertEqual(loss.item(), 2.0)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        loss.backward()
        torch.testing.assert_close(reconstruction.grad, torch.full_like(reconstruction, 1 / 24))
        torch.testing.assert_close(mu.grad, torch.full_like(mu, 0.125))
        torch.testing.assert_close(logvar.grad, torch.zeros_like(logvar))

    def test_loss_promotes_bf16_to_fp32(self):
        images = torch.ones(2, 3, 2, 2, dtype=torch.bfloat16)
        mu = torch.ones(2, 2, 2, 2, dtype=torch.bfloat16)
        loss, metrics = recipe.vae_loss(images, torch.zeros_like(images), mu, torch.zeros_like(mu))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(all(value.dtype == torch.float32 for value in metrics.values()))

    def test_model_preserves_odd_image_shape_and_stochastic_eval(self):
        model = MBConvVAE_V2(base_channels=4, max_channels=16, blocks_per_stage=1, latent_channels=2)
        images = torch.rand(2, 3, 17, 19)
        reconstructed, mu, logvar = model(images)
        self.assertEqual(reconstructed.shape, images.shape)
        self.assertEqual(mu.shape, (2, 2, 3, 3))
        loss, _ = recipe.vae_loss(reconstructed, images, mu, logvar)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        model.eval()
        with torch.no_grad():
            first, _, _ = model(images)
            second, _, _ = model(images)
        self.assertFalse(torch.equal(first, second))

    def test_schedule_reaches_launcher_beta_and_requires_configuration(self):
        schedules = ScheduleController.from_config(cpu_config().schedules)
        self.assertEqual(schedules["beta"], 0.0)
        for update in range(1, 10001):
            schedules.prepare_update(update, epoch=1)
            schedules.complete_update(update)
        self.assertEqual(schedules["beta"], 1e-6)
        with self.assertRaisesRegex(ValueError, "schedules.beta"):
            recipe.make_vae_step(None)

    def test_epoch_resume_preserves_saved_settings_and_update_counts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recipe, "image_datasets", side_effect=tiny_images
        ):
            output = Path(directory) / "run"
            context = deepopt.initialize_distributed("cpu")
            recipe.fit_vae(cpu_config(), context, output, progress_bar=False)
            before = read_checkpoint(output / "last.pt")
            # The CLI rebuilds components from the saved config, then calls fit().
            recipe.main(["--resume", str(output / "last.pt"), "--epochs", "2", "--no-progress"])
            after = read_checkpoint(output / "last.pt")
        self.assertEqual(before["state"]["optimizer_steps"], 3)
        self.assertEqual(after["state"]["completed_epochs"], 2)
        self.assertEqual(after["state"]["optimizer_steps"], 6)
        self.assertEqual(after["ema"]["updates"], 6)
        self.assertEqual(after["config"]["application"], before["config"]["application"])
        self.assertEqual(after["config"]["training"]["learning_rate"], 1e-3)
        self.assertEqual(len(after["history"]), 2)
        row = after["history"][-1]
        self.assertEqual(row["train"]["samples"], 5)
        self.assertEqual(row["validation"]["samples"], 3)
        for phase in ("train", "validation"):
            self.assertTrue({"loss", "l1", "kl", "psnr", "ssim"}.issubset(row[phase]))
            self.assertTrue(all(math.isfinite(row[phase][name]) for name in ("psnr", "ssim")))

    def test_mid_epoch_resume_matches_uninterrupted_model_ema_and_schedule(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recipe, "image_datasets", side_effect=tiny_images
        ):
            output = Path(directory) / "run"
            context = deepopt.initialize_distributed("cpu")
            recipe.fit_vae(cpu_config(), context, output, progress_bar=False)
            expected = read_checkpoint(output / "last.pt")
            snapshot = read_checkpoint(output / "last_step.pt")
            self.assertEqual(snapshot["state"]["optimizer_steps"], 2)
            resumed_output = Path(directory) / "resumed"
            resumed_output.mkdir()
            shutil.copyfile(output / "last_step.pt", resumed_output / "last_step.pt")
            recipe.main(["--resume", str(resumed_output / "last_step.pt"), "--no-progress"])
            actual = read_checkpoint(resumed_output / "last.pt")
        self.assertEqual(actual["state"], expected["state"])
        self.assertEqual(actual["schedules"], expected["schedules"])
        for name in expected["model"]:
            torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
        for name in expected["ema"]["shadow"]:
            torch.testing.assert_close(actual["ema"]["shadow"][name],
                                       expected["ema"]["shadow"][name], rtol=0, atol=0)
        for phase in ("train", "validation"):
            for name in ("loss", "l1", "kl", "psnr", "ssim", "samples"):
                self.assertEqual(actual["history"][0][phase][name], expected["history"][0][phase][name])

    def test_step_logs_confirm_completed_updates_before_epoch_checkpoint(self):
        reports = []
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recipe, "image_datasets", side_effect=tiny_images
        ):
            output = Path(directory) / "run"

            def observe_log(message, *arguments):
                if message.startswith("Epoch %s/%s | batch"):
                    # An epoch-only callback could not provide this report yet.
                    self.assertFalse((output / "last.pt").exists())
                    reports.append(arguments)

            context = deepopt.initialize_distributed("cpu")
            with patch.object(recipe.logger, "info", side_effect=observe_log):
                recipe.fit_vae(cpu_config(), context, output, progress_bar=False, log_every_steps=2)
            self.assertTrue((output / "last.pt").exists())
        self.assertEqual([row[4] for row in reports], [1, 2])
        self.assertEqual([row[2:4] for row in reports], [(1, 3), (2, 3)])

    def test_resumed_step_log_reports_the_first_new_update(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recipe, "image_datasets", side_effect=tiny_images
        ):
            output = Path(directory) / "run"
            context = deepopt.initialize_distributed("cpu")
            recipe.fit_vae(cpu_config(), context, output, progress_bar=False, log_every_steps=0)
            resumed = Path(directory) / "resumed"
            resumed.mkdir()
            shutil.copyfile(output / "last_step.pt", resumed / "last_step.pt")
            with patch.object(recipe.logger, "info") as log:
                recipe.fit_vae(cpu_config(), context, resumed, resume=resumed / "last_step.pt",
                               progress_bar=False, log_every_steps=100)
            reports = [call.args[1:] for call in log.call_args_list
                       if call.args[0].startswith("Epoch %s/%s | batch")]
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0][2:5], (3, 3, 3))

    def test_step_detaches_and_rescales_images_for_image_metrics(self):
        target = torch.zeros(2, 3, 16, 16)
        reconstruction = torch.ones_like(target, requires_grad=True)
        mu = torch.zeros(2, 2, 2, 2)
        schedules = ScheduleController.from_config(cpu_config().schedules)
        output = recipe.make_vae_step(schedules)(lambda images: (reconstruction, mu, mu), target)
        predictions, targets = recipe.image_inputs(output, target)
        torch.testing.assert_close(predictions, torch.ones_like(target))
        torch.testing.assert_close(targets, torch.full_like(target, 0.5))
        self.assertEqual(predictions.dtype, torch.float32)
        self.assertFalse(predictions.requires_grad)
        self.assertTrue(output["loss"].value.requires_grad)

    def test_psnr_averages_images_instead_of_batch_means_or_pooled_mse(self):
        metrics = recipe.make_metrics({"lpips": False})
        predictions = torch.stack([torch.full((3, 16, 16), value) for value in (0.5, 0.25, 0.125)])
        targets = torch.zeros_like(predictions)
        metrics["psnr"].update(predictions[:2], targets[:2])
        metrics["psnr"].update(predictions[2:], targets[2:])
        expected = sum(-20 * math.log10(value) for value in (0.5, 0.25, 0.125)) / 3
        self.assertAlmostEqual(metrics["psnr"].compute().item(), expected, places=5)
        metrics["ssim"].update(predictions[:2], predictions[:2])
        metrics["ssim"].update(predictions[2:], predictions[2:])
        self.assertAlmostEqual(metrics["ssim"].compute().item(), 1.0, places=5)

    def test_lpips_is_enabled_by_default_with_normalized_image_inputs(self):
        # Check wiring only; numerical LPIPS coverage below needs cached weights.
        with patch.object(recipe, "LearnedPerceptualImagePatchSimilarity") as constructor:
            metrics = recipe.make_metrics({})
        constructor.assert_called_once_with(net_type="alex", normalize=True)
        self.assertIn("lpips", metrics)

    @unittest.skipUnless(
        importlib.util.find_spec("torchvision")
        and (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth").exists(),
        "LPIPS numerical check requires torchvision and cached pretrained AlexNet weights",
    )
    def test_real_lpips_identical_images_score_zero(self):
        metric = recipe.make_metrics({"lpips": True})["lpips"]
        images = torch.rand(2, 3, 64, 64)
        metric.update(images, images)
        self.assertAlmostEqual(metric.compute().item(), 0.0, places=6)
        self.assertTrue(all(not parameter.requires_grad for parameter in metric.parameters()))

    def test_legacy_weight_import_unwraps_ddp_and_compile_and_loads_strictly(self):
        options = cpu_config().application["model"]
        source = MBConvVAE_V2(**options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"model_state_dict": {
                "module._orig_mod." + name: value for name, value in source.state_dict().items()
            }, "epoch": 25}, path)
            restored = recipe.make_model(options, init_weights=path)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            torch.save({"network": source.state_dict()}, path)
            recipe.load_initial_weights(restored, path, weights_key="network")
            torch.save({"model": {"missing.weight": torch.ones(1)}}, path)
            with self.assertRaises(RuntimeError):
                recipe.load_initial_weights(restored, path)

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "Install matching torchvision")
    def test_imagefolder_transform_normalizes_and_replays_seeded_augmentations(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            for split in ("train", "val"):
                folder = Path(directory) / split / "class"
                folder.mkdir(parents=True)
                pixels = bytes((index % 256 for index in range(23 * 31 * 3)))
                Image.frombytes("RGB", (23, 31), pixels).save(folder / "image.png")
            options = {"train_dir": str(Path(directory) / "train"),
                       "val_dir": str(Path(directory) / "val"),
                       "image_size": 16, "resize_size": 20}
            training, validation = recipe.image_datasets(options, seed=42)
            image = training[(3, 0)]
            replay = training[(3, 0)]
            torch.testing.assert_close(image, replay, rtol=0, atol=0)
            self.assertEqual(image.shape, (3, 16, 16))
            self.assertGreaterEqual(image.min().item(), -1)
            self.assertLessEqual(image.max().item(), 1)
            self.assertEqual(validation[0].shape, image.shape)

    def test_configuration_keeps_the_requested_training_and_runtime_settings(self):
        raw = tomllib.loads((ROOT / "examples/vae/imagenet.toml").read_text())
        config = config_from_dict(raw)
        self.assertEqual((config.training.epochs, config.training.batch_size), (80, 24))
        self.assertEqual(config.training.checkpoint_every_steps, 200)
        self.assertEqual(config.runtime.precision, "bf16")
        self.assertEqual(config.runtime.activation_compression.bits, 2)
        self.assertEqual(config.training.ema, {"enabled": True, "decay": 0.95})
        self.assertTrue(config.application["metrics"]["lpips"])
        self.assertEqual(asdict(config), asdict(config_from_dict(asdict(config))))


if __name__ == "__main__":
    unittest.main()
