# Saving, resuming, and configured runs

This guide is now organized into [configured runs and data](runs/README.md)
and [checkpoints and resume](checkpoints/README.md). Start with the
[documentation index](README.md) for other features. Earlier section links are
preserved below.

## Save and resume an existing trainer

Use the [direct checkpoint APIs](checkpoints/README.md) and the runnable
[checkpoint recipe](../examples/checkpointing/checkpoint_resume.py).

## Configured demo with managed artifacts

The [configured-runs README](runs/README.md) covers TOML settings, every CLI
command, reconstruction factories, evaluation, and generated artifacts.

### Choose a batch size

See [memory and batch sizing](memory/README.md) for auto/manual modes on fresh
and resumed runs, representative sample factories, decision logs and limits.

### Step checkpoints and resume

See [checkpoints and resume](checkpoints/README.md) for optimizer boundaries,
consumed-data position, worker/prefetch contracts, changed batching and recovery.

## Own your data and application settings

See [configured runs and data](runs/README.md) for caller-owned application
settings, dataset/loader factories and generator state.

## Replace a component

See [configured runs and data](runs/README.md) for the component factory
interfaces and custom reconstruction. See [training APIs](training/README.md)
when you want to use ready-made components without configuration.

### Use different optimizers for different layers

See [losses and optimization](optimization/README.md) and the
[multiple-optimizer recipe](../examples/optimization/multiple_optimizers.py).
