"""Conservative CUDA capacity admission without changing a requested workload."""

from copy import deepcopy
import math


def admit_cuda_capacity(*, requested_workload, device_available, total_bytes, free_bytes,
                        already_live_bytes, additional_tensor_bytes, workspace_budget_bytes,
                        margin_fraction=0.1, margin_bytes=2**30):
    """Require known capacity, incremental demand, workspace and safety margin.

    ``free_bytes`` already excludes live state, so live bytes are not subtracted
    twice. A second total-capacity check includes them explicitly. The default
    margin is max(10% of device capacity, 1 GiB); it is an admission policy, not
    a prediction of allocator fragmentation or opaque library workspace.
    ``workspace_budget_bytes=None`` means unknown, never zero.
    """
    if not math.isfinite(margin_fraction) or not 0 <= margin_fraction <= 1:
        raise ValueError("margin_fraction must be finite and between zero and one")
    values = {"total_bytes": total_bytes, "free_bytes": free_bytes,
              "already_live_bytes": already_live_bytes, "additional_tensor_bytes": additional_tensor_bytes,
              "workspace_budget_bytes": workspace_budget_bytes, "margin_bytes": margin_bytes}
    for name, value in values.items():
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")
    reasons = []
    if device_available is not True:
        reasons.append("CUDA device availability is false or unknown")
    reasons.extend(f"{name} is unknown" for name, value in values.items() if value is None)
    margin = required = None
    if not reasons:
        if total_bytes == 0 or free_bytes > total_bytes or already_live_bytes > total_bytes:
            reasons.append("capacity counters are inconsistent")
        margin = max(margin_bytes, math.ceil(total_bytes * margin_fraction))
        required = additional_tensor_bytes + workspace_budget_bytes + margin
        if required > free_bytes:
            reasons.append("additional tensors, workspace and margin exceed currently free capacity")
        if already_live_bytes + required > total_bytes:
            reasons.append("already live state plus requested demand exceeds total capacity")
    return {"admitted": not reasons, "status": "admitted" if not reasons else "rejected",
            "requested_workload": deepcopy(requested_workload), "device_available": device_available,
            **values, "margin_fraction": margin_fraction, "effective_margin_bytes": margin,
            "required_additional_bytes_including_margin": required, "reasons": reasons,
            "policy": "Free capacity excludes already live state; require both free and total checks. "
                      "Unknown values reject admission. No batch resizing, microbatching or extrapolation."}


def cuda_capacity_admission(torch, device, *, requested_workload, additional_tensor_bytes,
                            workspace_budget_bytes, margin_fraction=0.1, margin_bytes=2**30):
    """Query current counters; unavailable/failed queries yield rejected evidence."""
    available = device.type == "cuda" and torch.cuda.is_available()
    free = total = live = None
    query_error = None
    if available:
        try:
            free, total = map(int, torch.cuda.mem_get_info(device))
            live = int(torch.cuda.memory_allocated(device))
        except (RuntimeError, OSError) as error:
            query_error = f"{type(error).__name__}: {error}"
    result = admit_cuda_capacity(requested_workload=requested_workload, device_available=available,
                                 total_bytes=total, free_bytes=free, already_live_bytes=live,
                                 additional_tensor_bytes=additional_tensor_bytes,
                                 workspace_budget_bytes=workspace_budget_bytes,
                                 margin_fraction=margin_fraction, margin_bytes=margin_bytes)
    result["query_error"] = query_error
    return result
