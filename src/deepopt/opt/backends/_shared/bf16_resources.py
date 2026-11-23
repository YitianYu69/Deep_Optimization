"""Actual compiler resource admission, independent of Triton imports on CPU."""

import re
from collections import OrderedDict

import torch


_admissions = OrderedDict()
_MAX_ADMISSIONS = 256


def bf16_scratch_requirements(m, n, k, *, route="direct", panel_size=1024):
    """Explicit invocation scratch, excluding outputs and opaque workspaces."""
    if any(type(x) is not int or x < 0 for x in (m, n, k)):
        raise ValueError("GEMM dimensions must be nonnegative integers")
    if type(panel_size) is not int or panel_size <= 0:
        raise ValueError("panel_size must be a positive integer")
    if route not in ("direct", "dense", "panel", "materialized"):
        raise ValueError("Unknown BF16 scratch route")
    sizes = ([("decoded_panel", min(k, panel_size) * n * 2),
              ("fp32_accumulator", m * n * 4)] if route == "panel" and m * n * k else
             [("decoded_operand", k * n * 2)] if route == "materialized" else [])
    return tuple(dict(name=name, bytes=size, lifetime="call", owner="invocation",
                      reuse="invocation_stream_ordered", capture_safe_after_warmup=True)
                 for name, size in sizes)


def _specialization(value):
    if isinstance(value, torch.Tensor):
        # Triton pointer specialization uses 16-byte alignment, not identity.
        return ("tensor", str(value.device), value.dtype, value.data_ptr() % 16)
    if isinstance(value, dict):
        return tuple((key, _specialization(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_specialization(item) for item in value)
    return value


class BF16ResourceError(RuntimeError):
    def __init__(self, evidence):
        self.evidence = dict(evidence)
        super().__init__(f"BF16 resource rejection: {self.evidence}")


def admit_resources(required, limit, *, role, route, config, registers=None, spills=None):
    evidence = dict(role=role, route=route, config=config, shared_memory_bytes=required,
                    shared_memory_limit_bytes=limit, registers_per_thread=registers,
                    spills=spills, status="admitted", qualification="resource_only",
                    fallback=False)
    known = type(required) is int and required >= 0 and type(limit) is int and limit > 0
    if not known or required > limit:
        evidence.update(status="rejected", qualification="unqualified",
                        reason="unknown_resource_limit" if not known else "shared_memory_limit")
        raise BF16ResourceError(evidence)
    return evidence


def launch_admitted(kernel, grid, args, kwargs, *, device, role, route, config):
    """Compile without executing, check actual target demand, then launch.

    Triton's specialization cache owns compiled artifacts; no invocation tensor
    is retained here. Repeated calls use that cache, including capture warmup.
    Compile and execution errors never select another route implicitly.
    """
    properties = torch.cuda.get_device_properties(device)
    limit = getattr(properties, "shared_memory_per_block_optin", None)
    if limit is None:
        limit = getattr(properties, "shared_memory_per_block", None)
    key = (kernel, str(device), properties.major, properties.minor, limit,
           _specialization(args), _specialization(kwargs), role, route)
    cached = _admissions.get(key)
    if cached is not None:
        kernel[grid](*args, **kwargs)
        return None, dict(cached)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("BF16 specialization must be warmed and resource-admitted before CUDA capture")
    try:
        artifact = kernel.warmup(*args, grid=grid, **kwargs)
    except Exception as error:
        if type(error).__name__ != "OutOfResources":
            raise
        required = getattr(error, "required", None)
        match = re.search(r"Required:\s*(\d+)", str(error))
        if required is None and match:
            required = int(match[1])
        resource_name = getattr(error, "name", "shared memory" if "shared memory" in str(error) else "unknown")
        evidence = dict(role=role, route=route, config=config, status="rejected",
                        qualification="unqualified", reason="compiler_out_of_resources",
                        resource_name=resource_name, required=required,
                        compiler_resource_limit=getattr(error, "limit", None),
                        shared_memory_bytes=required if resource_name == "shared memory" else None,
                        shared_memory_limit_bytes=limit,
                        compiler_error=str(error), fallback=False)
        raise BF16ResourceError(evidence) from error
    evidence = admit_resources(getattr(getattr(artifact, "metadata", None), "shared", None), limit,
        role=role, route=route, config=config, registers=getattr(artifact, "n_regs", None),
        spills=getattr(artifact, "n_spills", None))
    # Some drivers expose register/spill counts only after handle creation.
    # Record those after the admitted launch without weakening preflight.
    launched = kernel[grid](*args, **kwargs)
    artifact = launched if launched is not None else artifact
    evidence.update(registers_per_thread=getattr(artifact, "n_regs", evidence["registers_per_thread"]),
                    spills=getattr(artifact, "n_spills", evidence["spills"]))
    _admissions[key] = dict(evidence)
    if len(_admissions) > _MAX_ADMISSIONS:
        _admissions.popitem(last=False)
    return artifact, evidence
