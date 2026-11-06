"""Estimate training tensor storage from a metadata-only fake execution.

This is not a prediction of total CUDA memory. Kernels, the allocator and
external libraries can reserve memory that tensor dispatch cannot observe.
"""

from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
import random
import sys
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensor, FakeTensorMode, unset_fake_temporarily
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from deepopt.engine.execution.optimization import OptimizationWindow
from deepopt.engine.core import TrainingCore
from deepopt.engine.execution.policy import ModeController
from deepopt.distributed import _process as distributed
from deepopt.observers import CheckObservers
from deepopt.optimizers import MultiOptimizer
from deepopt.scheduling.controller import fake_tensor_reads


class MemoryEstimationError(RuntimeError):
    """The supplied training computation cannot be traced safely with fake tensors."""


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    peak_bytes_by_device: dict[str, int]
    persistent_bytes_by_device: dict[str, int]
    initial_bytes_by_device: dict[str, int]
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"peak_bytes_by_device": dict(self.peak_bytes_by_device),
                "persistent_bytes_by_device": dict(self.persistent_bytes_by_device),
                "initial_bytes_by_device": dict(self.initial_bytes_by_device),
                "assumptions": list(self.assumptions)}


def _objects(roots):
    """Walk owned Python state without traversing function globals or modules."""
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        yield value
        if isinstance(value, Tensor):
            if value.is_leaf and value.grad is not None:
                yield from visit(value.grad)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                yield from visit(key)
                yield from visit(item)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                yield from visit(item)
        elif isinstance(value, MethodType):
            yield from visit(value.__self__)
        elif not isinstance(value, (type, FunctionType, BuiltinFunctionType, ModuleType)):
            if hasattr(value, "__dict__"):
                yield from visit(vars(value))
            for cls in type(value).__mro__:
                slots = cls.__dict__.get("__slots__", ())
                for name in (slots,) if isinstance(slots, str) else slots:
                    if name not in ("__dict__", "__weakref__") and hasattr(value, name):
                        yield from visit(getattr(value, name))

    yield from visit(roots)


def _storage(tensor):
    if tensor.layout != torch.strided:
        raise MemoryEstimationError("Memory tracing currently requires strided tensor storage; sparse layouts are unsupported")
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage._cdata), storage, int(storage.nbytes())


def _owned_bytes(roots):
    sizes, seen = {}, set()
    for value in _objects(roots):
        if isinstance(value, Tensor):
            key, _, size = _storage(value)
            if key not in seen:
                seen.add(key)
                sizes[key[0]] = sizes.get(key[0], 0) + size
    return sizes


class _StorageTrace(TorchDispatchMode):
    """Weak storage references include aliases and tensors saved by autograd."""

    def __init__(self):
        super().__init__()
        self.storages = {}
        self.peak = {}

    def track(self, values, *, scalar_factory=False):
        for key, (reference, _) in tuple(self.storages.items()):
            if reference.expired():
                del self.storages[key]
        for value in _objects(values):
            if isinstance(value, Tensor):
                if not isinstance(value, FakeTensor):
                    if scalar_factory and value.device.type == "cpu" and value.ndim == 0:
                        # torch.tensor(0.) lifts a tiny literal before FakeTensor
                        # dispatch; Adam uses this for its lazy step counter.
                        continue
                    raise MemoryEstimationError("A real tensor escaped the fake execution")
                key, storage, size = _storage(value)
                previous = self.storages.get(key)
                # In-place resize can change storage size without a new key.
                self.storages[key] = (previous[0] if previous is not None else StorageWeakRef(storage), size)
        totals = {}
        for (device, _), (_, size) in self.storages.items():
            totals[device] = totals.get(device, 0) + size
        for device, size in totals.items():
            self.peak[device] = max(self.peak.get(device, 0), size)

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        scalar_factory = function in (torch.ops.aten.lift_fresh.default, torch.ops.aten.lift_fresh_copy.default)
        self.track((args, kwargs), scalar_factory=scalar_factory)
        result = function(*args, **kwargs)
        self.track((args, kwargs), scalar_factory=scalar_factory)
        self.track(result)
        return result


def _snapshot_rng():
    # State tensors and optimizer scalar control constants are the only real
    # copies made here; model, optimizer payload and sample tensors remain fake.
    with unset_fake_temporarily():
        numpy = sys.modules.get("numpy")
        return (random.getstate(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                numpy.random.get_state() if numpy is not None else None)


def _restore_rng(state):
    with unset_fake_temporarily():
        python, cpu, cuda, numpy = state
        random.setstate(python)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if numpy is not None:
            sys.modules["numpy"].random.set_state(numpy)


def _clone_training(trainer, sample_batch, mode):
    # Shallow shells remove diagnostics and metric routing before recursive
    # copying, so their callbacks/state are neither cloned nor executed.
    runtime = copy(trainer.runtime)
    if runtime is trainer.runtime:
        raise MemoryEstimationError("The runtime must support independent copying")
    runtime.checks, runtime._failed = CheckObservers(False), False
    runtime.mode_controller = ModeController(execution_mode="max_speed")
    runtime._checks_spec = False
    # Trace only local model/optimizer execution. Never copy a process group,
    # reducer or collective into FakeTensorMode.
    from deepopt._config_resolution import ResolvedConfig
    runtime.core = TrainingCore(runtime.model, runtime.device, fp8=runtime.fp8,
                                config=ResolvedConfig())
    if runtime._stepper is not None:
        runtime._stepper = copy(runtime._stepper)
        runtime._stepper.checks = runtime.checks
    steps = copy(trainer._steps)
    steps.metrics, steps.metric_inputs = None, None
    roots = (runtime, steps, trainer.loss_strategy)
    initial = _owned_bytes(roots)
    memo = {}

    # Existing Adam-family step counters are known control state, not unknown
    # input data. Preserve those scalar values on independent CPU constants.
    for value in _objects(roots):
        if isinstance(value, torch.optim.Optimizer):
            for state in value.state.values():
                counter = state.get("step") if isinstance(state, Mapping) else None
                if isinstance(counter, Tensor) and counter.ndim == 0 and counter.device.type == "cpu":
                    with unset_fake_temporarily():
                        constant = torch.tensor(counter.item(), dtype=counter.dtype)
                    memo[id(counter)] = mode.fake_tensor_converter.from_real_tensor(mode, constant, make_constant=True)
    converter = mode.fake_tensor_converter
    previous_export = converter.export
    converter.export = True  # Do not specialize real scalar model/input values.
    try:
        for value in _objects((roots, sample_batch)):
            if isinstance(value, Tensor) and id(value) not in memo:
                if isinstance(value, FakeTensor) and value.fake_mode is not mode:
                    raise MemoryEstimationError("Sample tensors must belong to one FakeTensorMode")
                memo[id(value)] = mode.from_tensor(value)
        # Optimizer.__getstate__ omits subclass attributes, including the child
        # optimizers in MultiOptimizer. Preserve those with the shared memo.
        optimizers = [value for value in _objects(roots) if isinstance(value, torch.optim.Optimizer)]
        for optimizer in optimizers:
            memo[id(optimizer)] = object.__new__(type(optimizer))
        # Protect tensors hidden behind custom __getstate__ implementations
        # from Tensor's real-storage deepcopy implementation as well.
        with mode, FakeCopyMode(mode):
            for optimizer in optimizers:
                memo[id(optimizer)].__dict__ = deepcopy(vars(optimizer), memo)
            for optimizer in optimizers:
                if isinstance(optimizer, MultiOptimizer):
                    memo[id(optimizer)]._bind_groups()
            cloned, batch = deepcopy((roots, sample_batch), memo)
    finally:
        converter.export = previous_export
    runtime, steps, loss_strategy = cloned
    # LRScheduler installs a closure with a weak reference to the ORIGINAL
    # optimizer. Rebind the copied optimizer's method instead of executing it.
    for value in _objects(cloned):
        if isinstance(value, torch.optim.Optimizer) and getattr(value.step, "_wrapped_by_lr_sched", False):
            value.step = type(value).step.__get__(value, type(value))
    return runtime, steps, loss_strategy, batch, initial


def estimate_training_memory(trainer, sample_batch) -> TrainingMemoryEstimate:
    """Trace two training windows without allocating model/input tensor payloads.

    The caller supplies one representative batch; no batch dimension is
    inferred. Existing fake sample tensors reuse their strict FakeTensorMode.
    User training functions must be free of external side effects and I/O.
    Unsupported data-dependent operations raise MemoryEstimationError rather
    than falling back to real execution or guessing their output shapes.
    """
    if torch.is_inference_mode_enabled():
        raise MemoryEstimationError("Call training memory estimation outside torch.inference_mode()")
    if distributed.world_size() > 1 and not trainer.runtime.distributed:
        raise MemoryEstimationError("Distributed multi-rank estimation requires a DDPCore runtime created after group setup")
    if any(isinstance(module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
           for module in trainer.model.modules()):
        raise MemoryEstimationError("Distributed/DataParallel wrappers are unsupported; estimate an unwrapped single-device model")
    fake_modes = {value.fake_mode for value in _objects(sample_batch) if isinstance(value, FakeTensor)}
    if len(fake_modes) > 1:
        raise MemoryEstimationError("Sample tensors must belong to one FakeTensorMode")
    mode = next(iter(fake_modes)) if fake_modes else FakeTensorMode(allow_fallback_kernels=False)
    if mode.allow_fallback_kernels or mode.allow_non_fake_inputs or mode.propagate_real_tensors:
        raise MemoryEstimationError("Memory tracing requires a strict FakeTensorMode with fallback kernels "
                                    "and real tensor propagation disabled")
    rng = _snapshot_rng()
    try:
        runtime, steps, strategy, batch, initial = _clone_training(trainer, sample_batch, mode)
        trace = _StorageTrace()
        with torch.enable_grad(), mode, trace, fake_tensor_reads(mode):
            # A custom runtime can accept already-placed inputs without any
            # transfer dispatch. Every live batch field still occupies storage,
            # including fields this particular step does not consume.
            trace.track((runtime, steps, strategy, batch))
            runtime.model.train()
            runtime.zero_grad()
            for _ in range(2):
                window = OptimizationWindow(runtime, accumulation_steps=trainer.accumulation_steps,
                                            loss_strategy=strategy)
                for _ in range(trainer.accumulation_steps):
                    output = runtime.compute(batch, steps.training_step)
                    window.backward(output)
                    del output
                window.flush()
            trace.track((runtime, steps, strategy))
        persistent = _owned_bytes((runtime, steps, strategy))
        device = str(trainer.device)
        for values in (initial, persistent, trace.peak):
            values.setdefault(device, 0)
        if trainer.runtime.distributed:
            # Conservatively reserve parameter-sized bucket storage in addition
            # to the local gradient allocations seen by the trace.
            buckets = sum(parameter.numel() * parameter.element_size()
                          for parameter in trainer.model.parameters() if parameter.requires_grad)
            trace.peak[device] += buckets
            persistent[device] += buckets
            initial[device] += buckets
        ema = getattr(trainer, "ema", None)
        if ema is not None:
            for shadow in ema.shadow.values():
                shadow_device = str(shadow.device)
                shadow_bytes = shadow.numel() * shadow.element_size()
                for values in (initial, persistent, trace.peak):
                    values[shadow_device] = values.get(shadow_device, 0) + shadow_bytes
        return TrainingMemoryEstimate(
            trace.peak, persistent, initial,
            ("Tensor-storage estimate for this sample's shapes, dtypes and training execution path; input values are assumed valid.",
             "Two full accumulation windows include backward, gradient processing, optimizer updates and lazy optimizer state.",
             "DDP includes a parameter-sized gradient-bucket allowance; communication workspaces and external allocations remain excluded."
             if trainer.runtime.distributed else
             "Kernel workspaces, allocator fragmentation, CUDA context, communication and external allocations are excluded.",
             "Metrics, observers, scheduler and Trainer callbacks are excluded; user computation must not mutate external Python state or perform I/O.",
             "In-place compiled modules are copied and traced eagerly; compiler workspaces and graph pools are not estimated.",
             "Existing scalar CPU optimizer step counters are copied as known control constants; model and input values remain unknown.")
            + (("EMA parameter shadows are included; EMA update temporaries and validation parameter/buffer backups are excluded.",)
               if ema is not None else ()))
    except MemoryEstimationError:
        raise
    except Exception as exc:
        raise MemoryEstimationError(f"Training memory trace is unsupported: {type(exc).__name__}: {exc}") from exc
    finally:
        _restore_rng(rng)
