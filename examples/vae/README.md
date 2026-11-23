# Train an ImageNet VAE with DeepOpt

The [training example](train_vae.py) builds the model, optimizer, loaders,
schedules, and checkpoint callback, then calls `Trainer.fit()` directly. The
[MBConv VAE](mbconv_vae.py) uses an L1 reconstruction objective with a scheduled
KL term. It reports PSNR and SSIM, with optional LPIPS.

## Install and configure

```sh
python -m pip install -e .
python -m pip install torchvision
```

Use a torchvision build that matches your installed PyTorch. The
[example configuration](imagenet.toml) defaults to the relative
`data/imagenet/train` and `data/imagenet/val` directories, each with
ImageFolder layout. Copy the configuration to a local file to set your own
dataset roots, crop size, batch size, precision, or checkpoint interval.
Set `application.metrics.lpips = false` to run without pretrained LPIPS
backbone weights.

## Start a training run

From the repository root, choose one process per GPU:

```sh
NPROC_PER_NODE=2 VAE_CONFIG=work/vae.toml OUT_DIR=runs/vae \
  bash examples/vae/run_VAE.sh
```

The [launcher](run_VAE.sh) defaults to the bundled example configuration and
`runs/vae`. The output directory holds checkpoints and runtime logs. For
direct control over training flags, run `python examples/vae/train_vae.py --help`.

The [custom optimizer adapter](train_vae_custom.py) demonstrates wiring an
external optimizer and model factory into the same training pattern. It
requires the external Python modules it imports.

## Progress during a long epoch

Use `--progress-bar-every-steps N` to choose how often the terminal bar updates
and `--log-every-steps N` for completed-update log messages. Both options can
be passed through the launcher.

```sh
bash examples/vae/run_VAE.sh --progress-bar-every-steps 50 --log-every-steps 50
```

## Resume a run

```sh
RESUME=runs/vae/last_step.pt bash examples/vae/run_VAE.sh
```

Resume restores the recorded configuration and training state. To initialize a
fresh run from model weights, use `INIT_WEIGHTS` with a fresh `OUT_DIR`. The
library's [checkpoint guide](../../docs/checkpoints/README.md) covers the
checkpoint and restore contracts.

## Verification tools

Use the [VAE benchmark](benchmark_vae.py) to compare eager and compiled
training with matching inputs and settings. The [Inductor capture guide](INDUCTOR_VAE_V2.md)
covers code capture using an isolated native Torch model. Run results and
compiler artifacts belong in local output directories.
