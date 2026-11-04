# Scheduling and warmup

[Training](../training/README.md) · [Configuration](../runs/README.md) · [Checkpoints](../checkpoints/README.md)

Coefficient schedules change caller-named values at training boundaries. They
do not infer a loss, change arbitrary attributes, or own an optimizer. Ordinary
PyTorch LR schedulers remain independent and use their existing timing.

## Direct training

```python
from deepopt import Trainer
from deepopt.scheduling import ScheduleController, LinearWarmup

schedules = ScheduleController({
    "auxiliary_weight": LinearWarmup(start=0.0, end=0.1, duration=1000),
})

def step(model, batch):
    prediction = model(batch["inputs"])
    primary = primary_loss(prediction, batch["targets"])
    auxiliary = auxiliary_loss(prediction)
    return primary + schedules["auxiliary_weight"] * auxiliary

trainer = Trainer(model, optimizer, step_fn=step, schedules=schedules,
                  accumulation_steps=4)
```

The same controller is shared by the step and trainer. Values are prepared
before the first forward of each accumulation window, held across all its
microbatches, then committed after a successful optimizer update. A final
partial window commits once. A failure rolls preparation back; recovering
partially changed optimizer/model state still requires a checkpoint.

`trainer.state.optimizer_steps` is the authoritative counter. The controller
records the last committed boundary to validate state, but never increments a
clock independently. Diagnostic modes do not change coefficient values or timing.
Validation can read floats or tensor values; it cannot prepare, complete, abort
or restore schedules. Repeated evaluation does not change checkpointable schedule
state. An empty loader does not advance an update schedule.

## Curves and exact indexing

Every schedule implements `value_at(position)` and `get_config()`. Curve imports
do not import Torch. Built-in configurations are frozen and positions are
one-based positive integers. All results must be finite; the consumer owns
domain constraints such as requiring a particular loss weight to be nonnegative.

| Curve | Constructor fields and behavior |
| --- | --- |
| `Constant` | `value`; fixed for every position |
| `LinearWarmup`, `LinearDecay` | `start`, `end`, `duration`; interpolate between endpoints and hold the end |
| `CosineDecay` | `start`, `end`, `duration`; cosine interpolation with the same endpoints |
| `ExponentialDecay` | `start`, `gamma` in `(0, 1]`, optional `duration`; `start * gamma**(position-1)`, capped at the duration if supplied |
| `Piecewise` | Strictly increasing `milestones >= 2` and one more `values` entry; a milestone activates the next value at that position |
| `SequentialSchedule` | `schedules`, positive `durations` of equal length; each child receives local positions starting at 1; hold the last child's value at its assigned duration afterward |

All accept `interval="update"` (default) or `interval="epoch"`. Epoch coefficients
are prepared at the first training window of that epoch, then held for all its
updates. Sequential children must use the same interval as their parent. Segment
durations explicitly define boundaries, independently of a child's own horizon.

For linear curves of duration N > 1:

```text
alpha = min((position - 1) / (N - 1), 1)
value = (1 - alpha) * start + alpha * end
```

Update 1 uses `start`, update N uses `end`, and later positions hold `end`.
For N = 1, update 1 uses `end`. This differs from warmups whose first update
already uses `end/N`; existing formulas are not silently rewritten.

Custom schedules subclass `Schedule`, declare their interval, and return
immutable JSON-serializable configuration. Stateful implementations additionally
implement `state_dict()` and `load_state_dict()` with finite JSON-safe state.
Each coefficient owns a separate instance. Value evaluation must be deterministic
across ranks and must not access rank-local data, random state, or external I/O.
Custom state is restored before the next prepared update, including on abort.

## Configured experiments

```toml
[schedules.auxiliary_weight]
type = "linear_warmup"
interval = "update"
start = 0.0
end = 0.1
duration = 1000
```

Other type names are `constant`, `linear_decay`, `cosine_decay`,
`exponential_decay`, `piecewise`, and `sequential`. Their fields match the table
above. Sequential `schedules` is a list of child configurations. Unknown types
and fields fail validation. Names are caller-defined; no task-specific bindings
are inferred. Missing schedules preserve ordinary training behavior.

`build_experiment` constructs the controller and exposes it through
`trainer.schedules`. A `step_factory` receives the controller and returns the
existing `step_fn(model, batch)` callable:

```python
def make_step(schedules):
    def step(model, batch):
        prediction = model(batch[0])
        return prediction.square().mean() * schedules["auxiliary_weight"]
    return step

trainer, train_loader, valid_loader, generators = build_experiment(
    config, model_factory=make_model, dataset_factory=make_datasets,
    step_factory=make_step,
)
```

Alternatively, build a controller with `ScheduleController.from_config(config.schedules)`
and supply both `schedules=controller` and a step closing over it. If configuration
and a supplied controller both define schedules, their definitions must match.
Custom experiment factories must wire their own controller. The synthetic CLI
recipe does not invent uses for arbitrary coefficients; use a custom recipe to
consume them. See [example config](../../configs/scheduling.toml) and
[runnable direct example](../../examples/scheduling/coefficients.py).

## Learning-rate policies

Optimizer construction lives in `deepopt.optimizer_factory`; native scheduler
construction lives in `deepopt.scheduling.lr`. `deepopt.optim` is reserved for
shipped optimizer algorithms, with lazy aliases preserving old builder imports.
`MultiOptimizer` remains in its existing package.

The existing `Trainer(..., scheduler=lr_scheduler, scheduler_interval="step")`
API is unchanged. Step schedulers advance after optimizer success. Epoch
schedulers advance after a training epoch that had updates. Neither runs during
validation. Metric-based `ReduceLROnPlateau` remains a manual-loop concern.

Without `training.lr_schedule`, the factory retains legacy ExponentialLR with
`training.lr_gamma`. New policies use native PyTorch classes:

| `training.lr_schedule.type` | Fields | Native implementation |
| --- | --- | --- |
| `constant` | none | ConstantLR with factor 1 |
| `step_decay` | `step_size`, `gamma=0.1` | StepLR |
| `exponential_decay` | `gamma` | ExponentialLR |
| `piecewise` | Increasing positive `milestones`, `gamma=0.1` | MultiStepLR |
| `linear_warmup` | `warmup_steps`, `start_factor=0.1` | LinearLR |
| `cosine_decay` | `total_steps >= 2`, `eta_min=0` | CosineAnnealingLR |
| `warmup_cosine` | `warmup_steps < total_steps`, `start_factor=0.1`, `eta_min=0` | SequentialLR of LinearLR and CosineAnnealingLR |

`gamma` and `start_factor` must be in `(0, 1]`; `eta_min` must be nonnegative.
Specify either a new policy or a non-default legacy `lr_gamma`, not both.
The horizon is explicit and measured in the selected scheduler interval: optimizer
updates for `step`, training epochs for `epoch`. Never infer it from an unknown
stream. Batch-size or epoch-budget changes do not rewrite it.

Linear warmup uses the start factor on update 1 and reaches base LR on update N;
N = 1 immediately uses base LR. Cosine reaches `eta_min` on the final planned
position. As with native CosineAnnealingLR, running beyond the horizon can raise
LR again: choose an appropriate explicit horizon. LR milestone positions count
completed scheduler steps, unlike one-based coefficient milestones.

```toml
[training]
epochs = 10
batch_size = 32
learning_rate = 0.001
scheduler_interval = "step"

[training.lr_schedule]
type = "warmup_cosine"
warmup_steps = 100
total_steps = 1000
start_factor = 0.1
eta_min = 0.00001
```

One successful update of a `MultiOptimizer` wrapper is one global schedule tick.
Independent alternating-optimizer clocks and optimizer-attribute bindings are
not supported. There is no coefficient binding that can compete with an LR
scheduler for an optimizer's `lr`; configuration rejects such binding fields.

## Manual loops and compiled computations

Manual code owns the clock and calls the same controller around an update:

```python
schedules.prepare_update(next_update, epoch=current_epoch)
try:
    # Forward/backward over all microbatches.
    runtime.step()
except BaseException:
    schedules.abort_update()
    raise
else:
    schedules.complete_update(next_update)
```

For compiled computation, bind a scalar tensor once and pass it as an input:

```python
beta = schedules.tensor("auxiliary_weight", device=device, dtype=torch.float32)
compiled_loss = torch.compile(lambda prediction, coefficient: prediction.square().mean() * coefficient)
# After prepare_update, outside compiled_loss:
loss = compiled_loss(prediction, beta)
```

The controller fills the same storage in place without `.item()` or CUDA
synchronization. Bind scalars on their final device; do not move or overwrite
borrowed tensors afterward. Python float coefficients are appropriate for eager
losses. Calling schedule evaluation inside compiled code or capturing changing
Python floats can cause recompilation. Tensor caching may occur during validation
without changing schedule state, but bind before graph capture for fixed addresses.
Existing CUDA Graph constraints still apply. Default float32 storage can round
values; request a suitable floating dtype for the computation.

DDP validates initial definitions/state once with trainer settings. It uses the
shared update boundary to stay consistent, adding no per-update collectives.
Checkpoints validate schedule equality across ranks. Arbitrary data-dependent
custom schedule divergence is unsupported.

## Reports and restoration

Epoch callback rows add `scheduled_values_start` (first applied values in that
epoch) and `scheduled_values` (last applied values). Step events include the
values just applied, so existing explicitly sampled step callbacks can report
them. Reporting uses cached Python values and does not read device scalars.
One final value is not presented as an epoch average.

Checkpoint format 5 records names, types, immutable configurations, custom state,
last applied positions/values and the committed update. Restore checks these
against `TrainingState.optimizer_steps`. After update 4000, validation reads the
last applied value and the next training window prepares position 4001. Physical
rebatching changes neither that position nor the configured duration.

Version-4 files remain readable and resumable without coefficient schedules.
They cannot acquire a new controller during exact resume, even a constant one.
Changing names, types, intervals or horizons requires a new experiment. The
configured evaluation entry point restores last applied values without stepping.

## Verification and performance

Both white-box and black-box coverage live in [tests/scheduling](../../tests/scheduling/).
CUDA replay coverage lives in [tests/cuda](../../tests/cuda/).

```bash
python -m unittest discover -s tests/scheduling -v
DEEPOPT_EXTENDED_DDP=1 python -m unittest discover -s tests/scheduling -p test_ddp_black_box.py -v
python benchmarks/benchmark_scheduling.py --updates 10000 --repeats 5
```

The compilation test counts actual Dynamo compilations while coefficients change.
The two/four-rank Gloo test compares collective counts with unscheduled reference
training. The benchmark measures controller-only CPU cost, not GPU throughput.

To investigate worker aborts, keep per-rank logs and repeat the four-rank case:

```bash
export DEEPOPT_EXTENDED_DDP=1
export DEEPOPT_DDP_LOG_DIR="$PWD/work/ddp-logs"
export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
set -e
for attempt in $(seq 1 20); do
  python -m unittest tests.scheduling.test_ddp_black_box.ScheduleDistributedTests.test_four_rank_consistency_and_collective_count -v
done
```

Each invocation writes a distinct directory containing each rank's Python/native
stdout and stderr, fatal-signal tracebacks, PyTorch version, and flushed phase
markers. On failure the parent prints rank exit codes and log tails before
cleanup, and preserves the failing exit status. With no log directory configured,
logs are temporary but their tails still appear in a failed test's output.

The multiprocessing CI job repeats the four-rank case three additional times
and uploads `distributed-worker-logs` even on failure. Repetitions are stress
checks, not retries that turn a failure into success. A parent `join()` traceback
only reports a worker failure; inspect the earliest rank error and last phase.
`TORCH_DISTRIBUTED_DEBUG=DETAIL` can be enabled separately when diagnosing a
collective mismatch, but it changes PyTorch's process-group implementation and
can expose version-specific native failures. Compare with a normal-mode run.

The worker releases its model/trainer scope and collects remaining test-owned
cycles before destroying the process group. On failure it first records the
original traceback, then clears completed traceback frames so their local
variables cannot retain DDP objects through teardown. Production runtime
diagnostic callbacks borrow the runtime through weak references; they do not
extend the lifetime of its native execution core. No collection or extra
collective is added to the production training loop.

One local run on macOS arm64 / Python 3.11.17, with 10,000 updates and five
repeats, measured median costs of **1.22 µs/update** for one Python coefficient
and **2.18 µs/update** for one bound CPU tensor. These are isolated controller
measurements, not an end-to-end training speedup or GPU performance claim.
