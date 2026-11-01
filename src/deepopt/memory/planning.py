"""Static batch-size planning for new and resumed training."""

import random
import sys
from contextlib import contextmanager

import torch
from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily
from torch.utils.data import BatchSampler, DataLoader
from deepopt.logger import init_logger
from deepopt.distributed import _process as distributed


logger = init_logger("deepopt.engine.batch_size")


def _positive_size(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def loader_batch_size(loader) -> int | None:
    """Read the actual size, or None for an absent/unknown batching contract."""
    sampler_size = getattr(getattr(loader, "batch_sampler", None), "batch_size", None)
    size = sampler_size if sampler_size is not None else getattr(loader, "batch_size", None)
    if size is None:
        return None
    return _positive_size(size, "Loader batch_size")


def set_loader_batch_size(loader, size: int, *, require_resume_support: bool = False) -> None:
    """Resize before iteration, or through a loader's explicit cursor-safe API.

    A custom setter owns its consumed/prefetched-data contract. Step restoration
    additionally requires supports_batch_size_change=True. Ordinary DataLoader
    resizing is only supported with its exact standard BatchSampler, before
    creating an iterator; active worker iterators must not be reused afterwards.
    """
    _positive_size(size, "batch_size")
    if loader_batch_size(loader) == size:
        return
    if require_resume_support and getattr(loader, "supports_batch_size_change", False) is not True:
        raise ValueError("Changing batch size during step resume requires a loader with "
                         "supports_batch_size_change=True and a cursor-preserving set_batch_size()")
    setter = getattr(loader, "set_batch_size", None)
    if callable(setter):
        setter(size)
    elif type(loader) is DataLoader and type(loader.batch_sampler) is BatchSampler and not require_resume_support:
        if getattr(loader, "_iterator", None) is not None:
            raise ValueError("Resize a standard DataLoader before creating its worker iterator")
        loader.batch_sampler.batch_size = size
        # DataLoader forbids assignment after initialization, although the
        # standard batch sampler owns actual iteration. Keep this public value
        # consistent without rebuilding a loader or touching its sampler/RNG.
        object.__setattr__(loader, "batch_size", size)
    else:
        raise ValueError("This loader must implement set_batch_size(size) to support batch-size changes")
    if loader_batch_size(loader) != size:
        raise ValueError("Loader set_batch_size() did not apply the requested batch size")


def _device_bytes(report, field, device):
    values = report[field]
    if not isinstance(values, dict) or any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError(f"Memory estimate {field} must contain nonnegative integer byte counts")
    if device not in values:
        raise ValueError(f"Memory estimate {field} did not account for training device {device}")
    return values[device]


@contextmanager
def _preserve_rng_state():
    with unset_fake_temporarily():
        python_state, cpu_state = random.getstate(), torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        numpy = sys.modules.get("numpy")
        numpy_state = numpy.random.get_state() if numpy is not None else None
    try:
        yield
    finally:
        with unset_fake_temporarily():
            random.setstate(python_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            if numpy is not None:
                numpy.random.set_state(numpy_state)


def plan_batch_size(trainer, *, requested_batch_size: int, mode: str = "auto",
                    sample_batch_fn=None, memory_budget_bytes: int | None = None) -> dict:
    """Choose a statically estimated candidate, without running real training.

    The caller supplies a complete representative batch for each size; no batch
    axes, dataset lengths, or task shapes are inferred. Candidates are the
    requested size followed by integer halving down to one. The first estimated
    fit is selected, which is not a claim of the largest feasible size. Failures
    in shape tracing propagate; allocation failures never trigger a retry.
    """
    _positive_size(requested_batch_size, "requested_batch_size")
    if mode not in ("auto", "manual"):
        raise ValueError("Batch-size mode must be auto or manual")
    if memory_budget_bytes is not None:
        _positive_size(memory_budget_bytes, "memory_budget_bytes")
    if sample_batch_fn is not None and not callable(sample_batch_fn):
        raise TypeError("sample_batch_fn must be callable")
    result = {"requested_batch_size": requested_batch_size, "new_batch_size": requested_batch_size,
              "mode": mode, "status": "manual", "reason": "Using the requested batch size",
              "memory_budget_bytes": memory_budget_bytes, "candidates": [], "report": None}
    if mode == "manual":
        return result
    device = torch.device(trainer.device)
    if device.type != "cuda" and memory_budget_bytes is None:
        result.update(status="skipped", reason=f"Automatic memory planning on {device} requires an explicit memory budget")
        return result
    if distributed.world_size() > 1 and not getattr(getattr(trainer, "runtime", None), "distributed", False):
        raise ValueError("Automatic multi-rank planning requires a DDPCore runtime created after group setup")
    if sample_batch_fn is None:
        raise ValueError("Automatic memory planning requires sample_batch_fn(size) to create a representative batch; "
                         "provide it or use batch_size_mode='manual'")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device_name = str(device)
    free_bytes = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" and memory_budget_bytes is None else None

    from deepopt.memory import estimate_training_memory

    candidate = requested_batch_size
    while True:
        # The factory itself is traced so torch.empty/randn factories do not
        # allocate candidate-sized payloads, even for CPU-shaped sample data.
        with distributed.collective_errors("memory planning"):
            with _preserve_rng_state():
                fake_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=False)
                if fake_mode.propagate_real_tensors:
                    raise ValueError("Memory planning requires real tensor propagation disabled in FakeTensorMode")
                with fake_mode:
                    batch = sample_batch_fn(candidate)
                    estimate = estimate_training_memory(trainer, batch)
        report = estimate.as_dict()
        peak = _device_bytes(report, "peak_bytes_by_device", device_name)
        initial = _device_bytes(report, "initial_bytes_by_device", device_name)
        budget = memory_budget_bytes if memory_budget_bytes is not None else (4 * (free_bytes + initial)) // 5
        fits = bool(distributed.reduce_number(int(peak <= budget), torch.distributed.ReduceOp.MIN))
        result["candidates"].append({"batch_size": candidate, "fits": fits,
                                     "memory_budget_bytes": budget, "report": report})
        if fits:
            result.update(new_batch_size=candidate, status="planned", memory_budget_bytes=budget, report=report,
                          reason="Selected the first statically estimated fit among descending halved candidates")
            return result
        if candidate == 1:
            raise ValueError(f"No statically estimated batch size fits the memory budget on {device_name}; "
                             f"batch size 1 needs {peak} bytes, budget is {budget} bytes")
        candidate = max(1, candidate // 2)


def apply_batch_size_plan(trainer, train_loader, validation_loader, plan, *, step_resume: bool = False) -> None:
    """Apply a completed plan before iteration and expose its actual batching.

    Validation follows the selected size only when it used the same requested
    size as training. An independently configured validation size stays intact.
    Checkpoint callers add old_batch_size before applying the plan and own the
    separate saved-to-selected resume log.
    """
    requested = _positive_size(plan["requested_batch_size"], "requested_batch_size")
    selected = _positive_size(plan["new_batch_size"], "new_batch_size")
    actual = loader_batch_size(train_loader)
    if actual is not None and actual != requested:
        raise ValueError("Requested batch-size plan differs from the training loader")
    if selected != requested:
        if isinstance(getattr(trainer, "scheduler", None), torch.optim.lr_scheduler.OneCycleLR):
            raise ValueError("Automatic batch resizing changes update counts for OneCycleLR; "
                             "plan its schedule explicitly and use batch_size_mode='manual'")
        set_loader_batch_size(train_loader, selected, require_resume_support=step_resume)
        if loader_batch_size(validation_loader) == requested:
            set_loader_batch_size(validation_loader, selected)
    plan["validation_batch_size"] = loader_batch_size(validation_loader)
    trainer.batch_size_adjustment = plan
    if plan["status"] == "skipped":
        logger.warning("Automatic batch-size planning skipped: %s", plan["reason"])
    elif plan["status"] == "planned":
        report = plan.get("report") or {}
        logger.info("Batch-size plan selected %s; estimated peak tensor bytes=%s, budget=%s bytes",
                    selected, report.get("peak_bytes_by_device"), plan.get("memory_budget_bytes"))
    if selected != requested and "old_batch_size" not in plan:
        logger.warning("Training batch size changed from requested %s to %s (%s mode); "
                       "LR and accumulation are unchanged", requested, selected, plan["mode"])


def prepare_batch_size(trainer, train_loader, validation_loader=None, *, mode="auto",
                       sample_batch_fn=None, memory_budget_bytes=None):
    """Plan/apply batching before a fresh direct fit; return its logged decision.

    Read the requested size from the supplied loader, without inspecting data.
    For within-epoch resume, pass these options to Checkpoint.restore instead.
    """
    if not trainer.at_epoch_boundary or getattr(trainer, "_resume_pending", False):
        raise ValueError("Prepare fresh batching at an epoch boundary; use Checkpoint.restore for step resume")
    size = loader_batch_size(train_loader)
    if size is None:
        raise ValueError("Batch preparation requires a loader exposing batch_size; use plan_batch_size for custom batching")
    plan = plan_batch_size(trainer, requested_batch_size=size, mode=mode,
                           sample_batch_fn=sample_batch_fn, memory_budget_bytes=memory_budget_bytes)
    apply_batch_size_plan(trainer, train_loader, validation_loader, plan)
    return plan
