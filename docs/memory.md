# Memory estimates from caller-supplied batches

The maintained guide is [memory and batch sizing](memory/README.md), with a
[standalone estimator example](../examples/memory/estimate_memory.py) and a
[fresh/resume sizing example](../examples/memory/batch_size_modes.py).

## Estimate one representative step

See the [memory README](memory/README.md) for `estimate_training_memory`, its
per-device report, storage accounting and two-window execution scope.

## Provide batches for automatic batch sizing

See the [memory README](memory/README.md) for sample factories, candidate
selection, budgets and auto/manual modes on fresh and resumed runs.

## Limits to account for

See the [memory README](memory/README.md) for unsupported tracing and the
allocator, kernel, metric, callback and validation memory exclusions.
