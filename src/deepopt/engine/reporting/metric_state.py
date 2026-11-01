"""Persist nested TorchMetrics states and counters for exact step resume."""

from collections.abc import Mapping
from copy import deepcopy

from torch import Tensor
from torchmetrics import Metric


def _metrics(root):
    return {name: module for name, module in root.named_modules() if isinstance(module, Metric)}


def _copy_state(state):
    """Own detached tensors, including buffers and tensors in extra state.

    TorchMetrics copies its add_state values, but ordinary module buffers are
    still shared by state_dict. Seeding deepcopy's memo also supports non-leaf
    tensors in extra state while preserving OrderedDict metadata and aliases.
    """
    memo = {}
    visited = set()

    def detach(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, Tensor):
            memo[id(value)] = value.detach().clone()
        elif isinstance(value, Mapping):
            for item in value.values():
                detach(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                detach(item)

    detach(state)
    return deepcopy(state, memo)


def snapshot_metrics(metric: Metric) -> dict:
    modules = _metrics(metric)
    for module in modules.values():
        module.persistent(True)
    return {
        "types": {name: f"{type(module).__module__}.{type(module).__qualname__}"
                  for name, module in modules.items()},
        "state": _copy_state(metric.state_dict()),
        "updates": {name: module.update_count for name, module in modules.items()},
    }


def _restore_state(metric: Metric, saved_state, current_state):
    """Copy state, upgrading only the known optional-count schema addition.

    The exact older MetricAccumulator required a sample count for every batch.
    Its saved batch count therefore also equals the new counted-batch count.
    Subclasses can change those semantics and are deliberately not migrated.
    """
    missing = set(current_state) - set(saved_state)
    if set(saved_state) - set(current_state):
        raise ValueError("Step resume requires the same registered metric states")
    additions = {}
    if missing:
        # MetricAccumulator imports this adapter, so resolve its class lazily.
        from deepopt.engine.reporting.metrics import MetricAccumulator

        for name, module in _metrics(metric).items():
            if type(module) is not MetricAccumulator:
                continue
            prefix = f"{name}." if name else ""
            counted_key, batches_key = prefix + "counted_batches", prefix + "batches"
            if counted_key not in missing or batches_key not in saved_state:
                continue
            batches = saved_state[batches_key]
            if (not isinstance(batches, Tensor) or batches.ndim != 0
                    or batches.dtype != current_state[batches_key].dtype or batches.item() < 0):
                raise ValueError("Invalid saved batch count for metric-state migration")
            additions[counted_key] = batches_key
    if missing != set(additions):
        raise ValueError("Step resume requires the same registered metric states")
    state = _copy_state(saved_state)
    for counted_key, batches_key in additions.items():
        # Separate tensors are essential: both counters are updated in place.
        state[counted_key] = state[batches_key].clone()
    from deepopt.engine.reporting.metrics import MetricAccumulator

    for name, module in _metrics(metric).items():
        if type(module) is not MetricAccumulator:
            continue
        prefix = f"{name}." if name else ""
        extra = state.get(prefix + "_extra_state")
        if isinstance(extra, dict) and "report_presence" not in extra:
            # Legacy snapshots derive this host-side flag once at restoration,
            # never by reading CUDA counters on each subsequent metric update.
            batches = state[prefix + "batches"].item()
            loss_batches = state[prefix + "loss_batches"].item()
            extra["report_presence"] = None if batches == 0 else loss_batches > 0
    return state


def restore_metrics(metric: Metric, saved: dict, device) -> None:
    current = snapshot_metrics(metric)
    if saved["types"] != current["types"]:
        raise ValueError("Step resume requires the same metric class for every registered metric")
    if set(saved["updates"]) != set(current["updates"]):
        raise ValueError("Step resume requires the same metric update counters")
    if any(type(count) is not int or count < 0 for count in saved["updates"].values()):
        raise ValueError("Invalid saved metric update counter")
    # TorchMetrics assigns add_state tensors/lists directly when loading.
    # Copy first so resumed updates cannot mutate the checkpoint in memory.
    metric.load_state_dict(_restore_state(metric, saved["state"], current["state"]))
    metric.to(device)
    # TorchMetrics does not serialize its read-only update_count. This adapter
    # is guarded by the exact TorchMetrics version check in checkpoint restore.
    for name, module in _metrics(metric).items():
        # load_state_dict does not invalidate compute/forward or sync caches.
        # Clear those caches without calling user reset() or erasing the newly
        # restored states. A subsequent compute must use the loaded values.
        module._computed = None
        module._forward_cache = None
        module._cache = None
        module._is_synced = False
        module._update_count = saved["updates"][name]
