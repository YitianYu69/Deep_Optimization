# Advanced training reference

The reference is now split into focused READMEs. Start with the
[documentation index](README.md) or [examples index](../examples/README.md).
These section names remain available for earlier links.

## Step outputs and metric inputs

See [training computations](training/README.md) for supported output forms and
[metrics](metrics/README.md) for `metric_data`, `metric_inputs`, count reporting,
and scalar metric results.

## Manual runtime details

See [training APIs](training/README.md) for `TrainingRuntime`, manual update
ownership, runtime reuse, and recovery after a failed update.

## Optional model wrappers and tasks

See [training APIs](training/README.md) for `TrainingModule`, the structural
`Task` protocol, and built-in task presets. These are optional reuse tools.

## Supply a custom loss

See [losses and optimization](optimization/README.md) for scalar loss callables,
custom steps, built-in reductions, and trainable loss-module ownership.

## Loss weighting and reporting

See [losses and optimization](optimization/README.md) for `LossResult`,
`LossReport`, weighted accumulation, partial windows and legacy `BatchOutput`
shorthands. See [metrics](metrics/README.md) for phase reporting.

## Replace a training strategy

See [losses and optimization](optimization/README.md) for loss/gradient
strategies, custom optimizer construction, and strategy state.

## Optional runtime observers

See [observers and checks](observers/README.md) for finite checks, custom events,
activation/weight inspection, and the passive observer contract.

## Autograd and performance

See [architecture and development](architecture/README.md#autograd-and-performance)
for detached reporting, output lifetime, CPU benchmarks and profiling limits.

## Verification and current boundaries

See [architecture and development](architecture/README.md#supported-boundaries)
for scope and verification commands, [metrics](metrics/README.md) for
synchronization contracts, and [checkpoints](checkpoints/README.md) for resume
compatibility and failure handling.
