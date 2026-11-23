"""Train the supplied MBConvVAE_V2 with DeepOpt; see the adjacent README.

torchrun --standalone --nproc-per-node=4 examples/vae/train_vae.py --output NEW_RUN
torchrun --standalone --nproc-per-node=4 examples/vae/train_vae.py --resume RUN/last_step.pt
"""

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import traceback

import deepopt
import torch
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from deepopt import Checkpoint, LossResult, ScheduleController, Trainer
from deepopt import initialize_distributed, shutdown_distributed
from deepopt.data import SeededTransformDataset, build_loaders
from deepopt.logger import init_logger, logging_session
from deepopt.runtime import seed_everything
from deepopt.scheduling.lr import build_scheduler

if __package__:
    from .mbconv_vae import MBConvVAE_V2
else:
    from mbconv_vae import MBConvVAE_V2


logger = init_logger(__name__)


@dataclass(frozen=True)
class ImageTransform:
    """All augmentation randomness uses DeepOpt's per-epoch/sample generator."""

    image_size: int
    resize_size: int
    augment: bool

    def __call__(self, sample, *, generator):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        image, _label = sample
        image = TF.resize(image, self.resize_size, InterpolationMode.BILINEAR, antialias=True)
        if self.augment:
            width, height = image.size
            top = int(torch.randint(height - self.image_size + 1, (), generator=generator))
            left = int(torch.randint(width - self.image_size + 1, (), generator=generator))
            image = TF.crop(image, top, left, self.image_size, self.image_size)
            if torch.rand((), generator=generator) < 0.5:
                image = TF.hflip(image)
        else:
            image = TF.center_crop(image, self.image_size)
        # The decoder ends in Tanh, so use [-1, 1] rather than ImageNet class statistics.
        return TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)


def image_datasets(options, *, seed):
    from torchvision.datasets import ImageFolder

    size, resize = options["image_size"], options["resize_size"]
    if type(size) is not int or size < 16 or type(resize) is not int or resize < size:
        raise ValueError("image_size must be >= 16 and resize_size must be >= image_size")
    datasets = []
    for split, augment in (("train_dir", True), ("val_dir", False)):
        transform = ImageTransform(size, resize, augment)
        datasets.append(SeededTransformDataset(
            ImageFolder(options[split]), transform, seed=seed,
            transform_id=f"vae-resize-crop-flip-v1:{size}:{resize}:{augment}",
        ))
    return tuple(datasets)


def load_initial_weights(model, path, *, weights_key=None):
    """Import legacy model weights; optimizer/epoch/RNG state starts fresh."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload
    if weights_key is not None:
        state = payload[weights_key]
    elif isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(payload.get(key), Mapping):
                state = payload[key]
                break
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("Expected a state_dict; use --weights-key for its top-level checkpoint key")
    normalized = {}
    for key, value in state.items():
        # torchrun and torch.compile may wrap the legacy model in either order.
        name = key
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name in normalized:
            raise ValueError(f"Checkpoint parameter names collide after removing wrappers: {name}")
        normalized[name] = value
    model.load_state_dict(normalized, strict=True)
    logger.info("Initialized model weights from %s; starting a new DeepOpt run", path)


def make_model(options, *, init_weights=None, weights_key=None):
    model = MBConvVAE_V2(**options)
    if init_weights is not None:
        load_initial_weights(model, init_weights, weights_key=weights_key)
    return model


def vae_loss(reconstruction, target, mu, logvar, beta=1e-4):
    """The supplied objective; the training step supplies the scheduled beta."""
    reconstruction_loss = F.l1_loss(reconstruction.float(), target.float())
    mu, logvar = mu.float(), logvar.float()
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=(1, 2, 3)).mean()
    loss = reconstruction_loss + beta * kl
    return loss, {"loss": loss.detach(), "l1": reconstruction_loss.detach(), "kl": kl.detach()}


def make_vae_step(schedules):
    if schedules is None or "beta" not in schedules.values:
        raise ValueError("Configure the KL coefficient in [schedules.beta]")

    def step(model, images):
        reconstructed, mu, logvar = model(images)
        # BF16 is used for the model; loss reductions and exp(logvar) use FP32.
        with torch.autocast(device_type=images.device.type, enabled=False):
            loss, values = vae_loss(reconstructed, images, mu, logvar, beta=schedules["beta"])
        with torch.no_grad():
            # Share the same detached FP32 pair across all three image metrics.
            metric_images = (
                ((reconstructed.detach().float() + 1) / 2).clamp(0, 1),
                ((images.detach().float() + 1) / 2).clamp(0, 1),
            )
        return {
            "loss": LossResult.from_mean(loss, weight=images.shape[0]),
            "sample_count": images.shape[0],
            "l1": values["l1"],
            "kl": values["kl"],
            "metric_images": metric_images,
        }

    return step


def l1_inputs(output, batch):
    return {"value": output["l1"], "weight": batch.shape[0]}


def kl_inputs(output, batch):
    return {"value": output["kl"], "weight": batch.shape[0]}


def image_inputs(output, batch):
    return output["metric_images"]


def make_metrics(options):
    metrics = {
        "l1": MeanMetric(),
        "kl": MeanMetric(),
        "psnr": PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3)),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
    }
    if type(options.get("lpips", True)) is not bool:
        raise ValueError("application.metrics.lpips must be a boolean")
    if options.get("lpips", True):
        metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(
            net_type=options.get("lpips_net_type", "alex"), normalize=True,
        )
    return metrics


class StepLog:
    """Report completed updates without synchronizing every training batch."""

    def __init__(self, train_loader, *, every, first_step, epochs):
        self.train_loader = train_loader
        self.every, self.first_step, self.epochs = every, first_step, epochs

    def should_run_step(self, optimizer_step):
        return optimizer_step == self.first_step or optimizer_step % self.every == 0

    def __call__(self, trainer, event):
        if not deepopt.is_main_process():
            return
        batches = event["batches_processed"]
        # Account for a restored cursor, including a changed microbatch size.
        total = batches + self.train_loader.remaining_batches()
        logger.info(
            "Epoch %s/%s | batch %s/%s | optimizer step %s | beta=%s",
            event["epoch"], self.epochs, batches, total, event["optimizer_step"],
            event.get("scheduled_values", {}).get("beta"),
        )


def fit_vae(config, context, directory, *, resume=None, init_weights=None, weights_key=None,
            progress_bar=True, log_every_steps=10, progress_bar_every_steps=1):
    """Build ordinary components explicitly, then train with Trainer.fit()."""
    if type(log_every_steps) is not int or log_every_steps < 0:
        raise ValueError("log_every_steps must be a nonnegative integer")
    with logging_session(config, directory=Path(directory) / "logs"):
        seed_everything(config.seed, config.runtime.deterministic)
        train_dataset, valid_dataset = image_datasets(config.application["data"], seed=config.seed)
        train_loader, valid_loader, _ = build_loaders(
            train_dataset, valid_dataset,
            batch_size=config.training.batch_size, seed=config.seed,
            resumable=True, config=config,
        )

        model = make_model(config.application["model"], init_weights=init_weights,
                           weights_key=weights_key).to(context.device)
        if config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        elif config.training.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay, momentum=config.training.momentum,
            )
        else:
            raise ValueError("This VAE recipe supports training.optimizer = adamw or sgd")

        schedules = ScheduleController.from_config(config.schedules)
        metrics = make_metrics(config.application.get("metrics", {}))
        metric_inputs = {name: image_inputs for name in metrics}
        metric_inputs.update(l1=l1_inputs, kl=kl_inputs)
        trainer = Trainer(
            model, optimizer,
            device=context.device, precision=config.runtime.precision,
            step_fn=make_vae_step(schedules),
            metrics=metrics, metric_inputs=metric_inputs,
            accumulation_steps=config.training.accumulation_steps,
            scheduler=build_scheduler(optimizer, config.training),
            schedules=schedules, config=config,
        )
        # Identical initialization on every rank; independent latent draws afterward.
        if context.is_distributed:
            seed_everything(config.seed + context.global_rank, config.runtime.deterministic)

        checkpoint = Checkpoint(
            directory, every_steps=config.training.checkpoint_every_steps, config=config,
        )
        if resume is not None:
            checkpoint.restore(trainer, train_loader, valid_loader, path=Path(resume).resolve())
            if config.training.epochs < trainer.state.completed_epochs:
                raise ValueError("Total epochs cannot be less than the number already completed")
            if config.training.epochs == trainer.state.completed_epochs:
                if trainer.epoch_progress is not None:
                    raise ValueError("Total epochs must allow the active checkpoint epoch to finish")
                logger.info("Requested %s epochs are already complete", config.training.epochs)
                return

        step_callbacks = []
        if log_every_steps:
            step_callbacks.append(StepLog(
                train_loader, every=log_every_steps,
                first_step=trainer.state.optimizer_steps + 1, epochs=config.training.epochs,
            ))
        if deepopt.is_main_process():
            logger.info("Training loader has %s batches per rank per full epoch",
                        len(train_loader))
        trainer.fit(
            train_loader, valid_loader,
            epochs=config.training.epochs,
            monitor=config.training.monitor, monitor_mode=config.training.monitor_mode,
            callbacks=[checkpoint], step_callbacks=step_callbacks, progress_bar=progress_bar,
            progress_bar_every_steps=progress_bar_every_steps,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path, help="Directory for new checkpoints and logs")
    location.add_argument("--resume", type=Path, help="DeepOpt last.pt or last_step.pt")
    parser.add_argument("--config", type=Path, help="New-run TOML; defaults to adjacent imagenet.toml")
    parser.add_argument("--epochs", type=int, help="Total target epochs, including completed epochs")
    parser.add_argument("--batch-size", type=int, help="Microbatch size PER GPU")
    parser.add_argument("--init-weights", type=Path, help="Start a new run from legacy model weights")
    parser.add_argument("--weights-key", help="Optional top-level state_dict key in --init-weights")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-bar-every-steps", type=int, default=1,
                        help="Refresh progress loss/metrics every N optimizer updates (default: 1)")
    parser.add_argument("--log-every-steps", type=int, default=10,
                        help="Log first completed update and every N updates (default: 10; 0 disables)")
    args = parser.parse_args(argv)
    if args.resume is not None and (args.config is not None or args.init_weights is not None):
        parser.error("--resume restores the saved config and weights; omit --config and --init-weights")
    if args.weights_key is not None and args.init_weights is None:
        parser.error("--weights-key requires --init-weights")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.log_every_steps < 0:
        parser.error("--log-every-steps must be nonnegative")
    if args.progress_bar_every_steps < 1:
        parser.error("--progress-bar-every-steps must be positive")

    if args.resume is not None:
        if args.resume.name not in ("last.pt", "last_step.pt"):
            parser.error("--resume requires a DeepOpt last.pt or last_step.pt")
        # Rebuild ordinary components from the saved config before restore().
        # The full payload is released before the model/optimizer are allocated.
        source = torch.load(args.resume, map_location="cpu", weights_only=True)["config"]
        directory = args.resume.resolve().parent
    else:
        source = args.config or Path(__file__).with_name("imagenet.toml")
        directory = args.output.resolve()
    config = deepopt.load_config(source)
    if args.epochs is not None or args.batch_size is not None:
        options = asdict(config)
        if args.epochs is not None:
            options["training"]["epochs"] = args.epochs
        if args.batch_size is not None:
            options["training"]["batch_size"] = args.batch_size
        config = deepopt.load_config(options)

    try:
        context = initialize_distributed(config.runtime.device)
        fit_vae(config, context, directory, resume=args.resume, init_weights=args.init_weights,
                weights_key=args.weights_key, progress_bar=not args.no_progress,
                log_every_steps=args.log_every_steps,
                progress_bar_every_steps=args.progress_bar_every_steps)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gc.collect()
        shutdown_distributed()


if __name__ == "__main__":
    main()
