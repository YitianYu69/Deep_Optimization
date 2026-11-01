"""Read-only model/optimizer storage accounting, independent of fake execution."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn.parameter import is_lazy
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.logger import init_logger
from deepopt.optimizers import MultiOptimizer


_GB = 1_000_000_000
logger = init_logger("deepopt.memory_stats")


@dataclass(frozen=True)
class ModelMemorySummary:
    total_parameters: int | None
    trainable_parameters: int | None
    frozen_parameters: int | None
    parameter_bytes: int | None
    buffer_bytes: int | None
    model_bytes: int | None
    optimizer_bytes: int | None
    notes: tuple[str, ...] = ()

    def _percentage(self, count) -> float | None:
        if count is None or self.total_parameters is None or self.total_parameters == 0:
            return None
        return 100.0 * count / self.total_parameters

    @property
    def total_percent(self) -> float | None:
        return self._percentage(self.total_parameters)

    @property
    def trainable_percent(self) -> float | None:
        return self._percentage(self.trainable_parameters)

    @property
    def frozen_percent(self) -> float | None:
        return self._percentage(self.frozen_parameters)

    @property
    def model_gb(self) -> float | None:
        return None if self.model_bytes is None else self.model_bytes / _GB

    @property
    def optimizer_gb(self) -> float | None:
        return None if self.optimizer_bytes is None else self.optimizer_bytes / _GB

    def as_dict(self) -> dict:
        return {**asdict(self), "total_percent": self.total_percent,
                "trainable_percent": self.trainable_percent, "frozen_percent": self.frozen_percent,
                "model_gb": self.model_gb, "optimizer_gb": self.optimizer_gb}


class _StorageSet:
    def __init__(self):
        self.storages = {}
        self.notes = set()
        self.complete = True
        self.seen = {}

    def visit(self, value):
        if id(value) in self.seen:
            return
        # Retain temporary sparse views/containers so Python cannot reuse IDs.
        self.seen[id(value)] = value
        if isinstance(value, Tensor):
            self.tensor(value)
        elif isinstance(value, Mapping):
            # Optimizer state keys are parameter references, not owned state.
            for item in value.values():
                self.visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                self.visit(item)

    def tensor(self, tensor):
        if is_lazy(tensor):
            self.complete = False
            self.notes.add("Uninitialized tensors: materialize the model and summarize again for complete counts/storage.")
            return
        if tensor.device.type == "meta" or isinstance(tensor, FakeTensor):
            self.notes.add("Meta/fake tensors have no allocated payload; their physical storage is counted as zero.")
            return
        try:
            if tensor.layout == torch.sparse_coo:
                self.visit(tensor._indices())
                self.visit(tensor._values())
            elif tensor.layout in (torch.sparse_csr, torch.sparse_bsr):
                self.visit(tensor.crow_indices())
                self.visit(tensor.col_indices())
                self.visit(tensor.values())
            elif tensor.layout in (torch.sparse_csc, torch.sparse_bsc):
                self.visit(tensor.ccol_indices())
                self.visit(tensor.row_indices())
                self.visit(tensor.values())
            else:
                storage = tensor.untyped_storage()
                # Keep storage alive, use identity rather than data_ptr (empty/meta
                # tensors and CUDA address reuse make pointer-only counts unsafe).
                key = (str(tensor.device), storage._cdata)
                self.storages[key] = storage
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            self.complete = False
            self.notes.add("Some tensor storage is unavailable for inspection; affected byte totals are unknown.")

    def size(self):
        return sum(storage.nbytes() for storage in self.storages.values()) if self.complete else None


def _count(parameters):
    if any(is_lazy(parameter) for parameter in parameters):
        return None
    try:
        return sum(parameter.numel() for parameter in parameters)
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        return None


def summarize_model_memory(model: nn.Module, optimizer=None) -> ModelMemorySummary:
    """Count distinct parameters and currently owned tensor storage (decimal GB).

    Model storage is the union of registered parameters and buffers. Optimizer
    storage includes nested state and tensor-valued group settings, excluding
    model storage already counted. MultiOptimizer children are included. No
    forward, backward, optimizer step, state_dict, copy, or tensor read is used.
    Unknown counts/storage are None, never fabricated estimates.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    frozen = [parameter for parameter in parameters if not parameter.requires_grad]
    weights, buffers, state = _StorageSet(), _StorageSet(), _StorageSet()
    weights.visit(parameters)
    buffers.visit(list(model.buffers()))
    model_storage = {**weights.storages, **buffers.storages}
    seen_optimizers = set()

    def visit_optimizer(current):
        if current is None or id(current) in seen_optimizers:
            return
        if not isinstance(current, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer or None")
        seen_optimizers.add(id(current))
        state.visit(current.state)
        for group in current.param_groups:
            state.visit({key: value for key, value in group.items() if key != "params"})
        if isinstance(current, MultiOptimizer):
            for child in current.optimizers.values():
                visit_optimizer(child)

    visit_optimizer(optimizer)
    # Parameter views/tied weights and state aliases share backing allocations.
    for key in model_storage:
        state.storages.pop(key, None)
    notes = weights.notes | buffers.notes | state.notes
    if optimizer is not None:
        notes.add("Optimizer storage is currently allocated tensor state/settings; lazy state may grow after the first update.")
    return ModelMemorySummary(
        total_parameters=_count(parameters), trainable_parameters=_count(trainable), frozen_parameters=_count(frozen),
        parameter_bytes=weights.size(), buffer_bytes=buffers.size(),
        model_bytes=sum(s.nbytes() for s in model_storage.values()) if weights.complete and buffers.complete else None,
        optimizer_bytes=state.size(), notes=tuple(sorted(notes)))


def log_model_memory(model: nn.Module, optimizer=None) -> ModelMemorySummary:
    """Log a fresh summary through the shared logger and return it for inspection."""
    summary = summarize_model_memory(model, optimizer)
    def count(value, percent):
        number = "unknown" if value is None else f"{value:,}"
        ratio = "N/A" if percent is None else f"{percent:.2f}%"
        return f"{number} ({ratio})"
    logger.info("Model parameters: total=%s trainable=%s frozen=%s",
                count(summary.total_parameters, summary.total_percent),
                count(summary.trainable_parameters, summary.trainable_percent),
                count(summary.frozen_parameters, summary.frozen_percent))
    def gb(value):
        return "unknown" if value is None else f"{value:.9f} GB"
    logger.info("Model tensor storage (parameters + buffers): %s", gb(summary.model_gb))
    logger.info("Optimizer tensor storage (currently allocated): %s", gb(summary.optimizer_gb))
    for note in summary.notes:
        logger.info("Memory accounting: %s", note)
    return summary
