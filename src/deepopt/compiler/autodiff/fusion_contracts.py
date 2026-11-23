"""Serializable replacement contracts and candidate-specific execution ranges."""

from contextlib import nullcontext
from collections.abc import Mapping
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path

import torch


def candidate(graph, node, spec, phase, pattern, *, requested=True):
    policy = getattr(spec, "bf16_policy", None)
    policy = policy.as_dict() if hasattr(policy, "as_dict") else policy
    signature = tuple((getattr(item, "op", type(item).__name__),
                       tuple((value.name, value.meta.shape, str(value.meta.dtype))
                             for value in item.outputs() if value.meta is not None),
                       repr(getattr(item, "kwargs", {}))) for item in graph.nodes)
    graph_id = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    site = getattr(spec, "node_name", None) or node.outputs()[0].name
    identity = (graph_id, site, phase, pattern)
    return {"candidate_id": hashlib.sha256(repr(identity).encode()).hexdigest()[:24],
            "graph_id": graph_id, "node_name": site, "module_path": getattr(spec, "module_path", None),
            "phase": "train" if getattr(spec, "module_training", True) else "eval",
            "execution_phase": phase, "pattern": pattern, "requested": requested,
            "effective_enabled": requested, "discovered": True, "eligible": None,
            "selected": False, "status": "disabled" if not requested else "rejected",
            "reason": "disabled" if not requested else "unsupported_contract",
            "observed_cuda_launches": None, "bf16_policy": policy}


def select(node, record, *, roles, consumed=(), source_files=()):
    """Bind an exact profiler range to this implementation, never its siblings."""
    function = node.target
    numerical = getattr(function, "numerical_function", function)
    digest = hashlib.sha256()
    identity = (record["graph_id"], record["node_name"], record["execution_phase"], record["pattern"],
                json.dumps(record.get("bf16_policy"), sort_keys=True),
                json.dumps(record.get("native_statistics_contract"), sort_keys=True))
    record["candidate_id"] = hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
    digest.update(repr(identity).encode())
    if getattr(numerical, "keywords", None):
        digest.update(repr(sorted(numerical.keywords.items())).encode())
    try:
        digest.update(inspect.getsource(numerical).encode())
    except (OSError, TypeError):
        digest.update(repr(getattr(numerical, "__qualname__", type(numerical).__name__)).encode())
    spec = getattr(function, "specification", getattr(node, "specification", None))
    if isinstance(spec, Mapping):
        digest.update(repr(sorted(spec.items())).encode())
    for name in ("kernel_policy_fingerprint", "tile", "num_stages", "epilogue", "activation", "compression",
                 "execution", "decode", "backward_order", "producer_activation", "producer_derivative", "need_dx", "need_dw"):
        digest.update(repr((name, getattr(spec, name, None))).encode())
    plan = getattr(spec, "plan", None)
    if isinstance(plan, Mapping):
        digest.update(repr(plan.get("activation_compression")).encode())
    root = Path(__file__).resolve().parents[2]
    for relative in source_files:
        path = root / relative
        digest.update(path.read_bytes())
    label = "deepopt.fusion::" + record["candidate_id"]

    @wraps(function)
    def execute(*args, **kwargs):
        profile = getattr(spec, "profile_ranges", False) or torch.autograd._profiler_enabled()
        with torch.profiler.record_function(label) if profile else nullcontext():
            return function(*args, **kwargs)

    execute.numerical_function = numerical
    execute.specification = spec
    node.target = execute
    record.update(selected=True, eligible=True, status="selected", reason=None,
                  profile_range=label, implementation_fingerprint=digest.hexdigest(),
                  kernel_evidence={"required_kernel_roles": roles, "forbidden_kernel_symbols": []},
                  replacement_contract={"original_nodes": tuple(consumed),
                      "consumed_values": tuple(value.name for value in node.inputs()),
                      "outputs": tuple(value.name for value in node.outputs()),
                      "dependencies": tuple(value.name for value in node.inputs()),
                      "effects": "pure", "rounding": "preserve_original_cast_boundaries",
                      "estimated_scratch_bytes": None})
    record["legality_signature"] = {
        "inputs": [{"name": value.name, "shape": value.meta.shape, "dtype": str(value.meta.dtype),
                    "device": str(value.meta.device), "stride": value.meta.stride,
                    "requires_grad": value.meta.requires_grad, "representation": value.meta.representation}
                   for value in node.inputs() if value.meta is not None],
        "outputs": [{"name": value.name, "shape": value.meta.shape, "dtype": str(value.meta.dtype),
                     "representation": value.meta.representation}
                    for value in node.outputs() if value.meta is not None],
        "selection_control": "explicit_fusion_toggle", "measured_default_promoted": False}
    # Fail at compilation if a report accidentally retains executable state.
    json.dumps(record, allow_nan=False)


def ordered_operations(schedule):
    """Describe the final executable order, including pointwise partitions."""
    rows = []
    for index, item in enumerate(schedule.items):
        nodes = getattr(item, "nodes", None)
        if nodes is None:
            nodes = (getattr(item, "node", item),)
        for node in nodes:
            rows.append({"position": index, "operation": getattr(node, "op", type(node).__name__),
                         "outputs": tuple(value.name for value in node.outputs()),
                         "dependencies": tuple(value.name for value in node.inputs()),
                         "constraints": node.scheduling_constraints})
    return tuple(rows)
