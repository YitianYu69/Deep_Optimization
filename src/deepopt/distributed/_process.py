"""Internal default-process-group setup and rank-coordinated operations.

Launch one process per device with torchrun. All ranks must enter training,
evaluation and checkpoint calls in the same order. User callbacks run on every
rank; guard noncollective external side effects with is_main_process().
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
import os

import torch
from torch import distributed as dist


_owns_group = False


@dataclass(frozen=True)
class DistributedContext:
    """Initialization metadata snapshot, not a live process-group handle.

    Global rank/world size come from the active group. Local topology comes
    from torchrun's environment; it is unknown (None) for externally initialized
    multi-rank groups without LOCAL_RANK/LOCAL_WORLD_SIZE. Device indices are
    deliberately not treated as process ranks.
    """

    device: torch.device
    global_rank: int
    local_rank: int | None
    world_size: int
    local_world_size: int | None
    backend: str | None
    is_initialized: bool
    owns_process_group: bool

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.global_rank == 0

    def as_dict(self) -> dict:
        """Return JSON-compatible metadata, including a string device."""
        return {**asdict(self), "device": str(self.device),
                "is_distributed": self.is_distributed, "is_main_process": self.is_main_process}


def _environment_integer(name, *, minimum, default=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def is_main_process():
    return rank() == 0


def initialize_distributed(device="auto", *, backend=None, timeout=180) -> DistributedContext:
    """Join torchrun's group (if present) and return device/rank metadata.

    CUDA uses LOCAL_RANK and NCCL; CPU uses Gloo. An existing default group is
    reused, and is never destroyed by shutdown_distributed(). Use result.device
    for model placement and result.global_rank/local_rank/world_size for routing.
    """
    global _owns_group
    local_rank = _environment_integer("LOCAL_RANK", minimum=0)
    local_world_size = _environment_integer("LOCAL_WORLD_SIZE", minimum=1)
    requested_world = _environment_integer("WORLD_SIZE", minimum=1, default=1)
    if local_rank is not None and local_world_size is not None and local_rank >= local_world_size:
        raise ValueError("LOCAL_RANK must be smaller than LOCAL_WORLD_SIZE")
    requested = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        index = local_rank if local_rank is not None else (device.index if device.index is not None else torch.cuda.current_device())
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    available = dist.is_available()
    initialized = available and dist.is_initialized()
    if requested_world > 1 and not initialized:
        if not available:
            raise RuntimeError("This PyTorch build does not support distributed training")
        dist.init_process_group(backend=backend or ("nccl" if device.type == "cuda" else "gloo"),
                                timeout=timedelta(seconds=timeout))
        _owns_group = True
    initialized = available and dist.is_initialized()
    size = dist.get_world_size() if initialized else 1
    return DistributedContext(
        device=device, global_rank=dist.get_rank() if initialized else 0,
        local_rank=local_rank if local_rank is not None else (0 if size == 1 else None),
        world_size=size, local_world_size=local_world_size if local_world_size is not None else (1 if size == 1 else None),
        backend=str(dist.get_backend()) if initialized else None,
        is_initialized=initialized, owns_process_group=bool(initialized and _owns_group),
    )


def shutdown_distributed():
    """Release only the process group initialized by DeepOpt."""
    global _owns_group
    if _owns_group and dist.is_initialized():
        dist.destroy_process_group()
    _owns_group = False


def collective_device():
    return torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")


def reduce_number(value, op=dist.ReduceOp.SUM):
    if world_size() == 1:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=collective_device())
    dist.all_reduce(tensor, op=op)
    return tensor.item()


def require_equal(value, description):
    if world_size() > 1:
        values = [None] * world_size()
        dist.all_gather_object(values, value)
        if any(item != values[0] for item in values[1:]):
            raise ValueError(f"DDP requires matching {description} on every rank")


@contextmanager
def collective_errors(operation):
    """Propagate local Python failures at a shared, noncollective boundary."""
    if world_size() == 1:
        yield
        return
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    failed = reduce_number(int(error is not None), dist.ReduceOp.MAX)
    if failed:
        messages = [None] * world_size()
        dist.all_gather_object(messages, None if error is None else f"rank {rank()}: {type(error).__name__}: {error}")
        if error is not None:
            raise error
        raise RuntimeError(f"Distributed {operation} failed: " + "; ".join(message for message in messages if message))


def rank_zero_call(function):
    """Run shared file I/O once, and report its failure to all participating ranks."""
    with collective_errors("rank-zero operation"):
        if is_main_process():
            return_value = function()
        else:
            return_value = None
    return return_value


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def training_batches(loader):
    """Reject unequal numbers of training batches before backward can hang."""
    if world_size() == 1:
        yield from loader
        return
    with collective_errors("iterator construction"):
        iterator = iter(loader)
    while True:
        with collective_errors("data loading"):
            try:
                batch, available = next(iterator), 1
            except StopIteration:
                batch, available = None, 0
        count = reduce_number(available)
        if count == 0:
            return
        if count != world_size():
            raise ValueError("DDP training requires equal batch counts on all ranks; use a distributed sampler")
        yield batch


def set_loader_epoch(loader, epoch):
    seen, owners = set(), [loader]
    while owners:
        owner = owners.pop()
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        setter = getattr(owner, "set_epoch", None)
        if callable(setter):
            setter(epoch)
        owners.extend(getattr(owner, name, None) for name in ("sampler", "batch_sampler"))
