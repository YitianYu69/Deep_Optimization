"""Share equivalent explicit saved-operand packs after native differentiation.

This is common subexpression elimination, not numerical kernel fusion. A pack
can be shared only for the same SSA source in one effect interval and with an
identical codec, cast, traversal, correction and physical result contract.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from functools import wraps
import hashlib
import json
from math import prod
import os
from pathlib import Path

import torch

from ..ir import CallNode, GetAttrNode, InputNode, OutputNode, PointwiseNode, TensorValue, register_uses


_CONTRACT_FIELDS = frozenset(("bits", "group_size", "logical_shape", "logical_dtype", "logical_stride",
                              "restored_stride", "traversal", "rounding_domain", "source_cast"))
_CONVOLUTION_FIELDS = frozenset(("source_shape", "matrix_shape", "codec", "unbatched", "transposed",
                                "explicit_padding", "padding_mode", "synthetic_padding_right"))
_PACK_OPERATIONS = frozenset(("codec.save_activation", "codec.save_convolution_source"))
_PURE_POINTWISE = frozenset(("add", "sub", "mul", "div", "neg", "relu", "sigmoid", "tanh", "gelu", "silu",
                             "leaky_relu", "hardtanh", "hardswish", "hardsigmoid", "softplus", "mish"))


def _freeze(value):
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze(child)) for key, child in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    if value is None or isinstance(value, (str, bool, int, float, torch.dtype, torch.device)):
        return value
    raise TypeError("Pack compatibility metadata must contain immutable scalar descriptors")


def _metadata(meta):
    if meta is None:
        raise ValueError("Pack sharing requires complete source and result metadata")
    return tuple(getattr(meta, field, None) for field in (
        "shape", "dtype", "device", "stride", "representation", "storage_dtype", "block_shape", "layout"))


def _compatibility(node, version):
    if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], TensorValue):
        raise ValueError("unsupported_pack_arguments")
    source = node.args[0]
    options = node.specification
    if not isinstance(options, Mapping) or _CONTRACT_FIELDS - options.keys():
        raise ValueError("incomplete_codec_contract")
    spatial = node.op == "codec.save_convolution_source"
    if spatial and _CONVOLUTION_FIELDS - options.keys():
        raise ValueError("incomplete_convolution_contract")
    if (type(options["bits"]) is not int or options["bits"] not in (1, 2, 4, 8)
            or type(options["group_size"]) is not int or options["group_size"] < 1):
        raise ValueError("invalid_codec_recipe")
    if len(node.values) != 3:
        raise ValueError("physical_component_schema")
    if source.meta is None or any(v.meta is None for v in node.values):
        raise ValueError("metadata")
    if (tuple(options["logical_shape"]) != source.meta.shape or options["logical_dtype"] != source.meta.dtype
            or options["logical_stride"] != source.meta.stride):
        raise ValueError("source_contract_mismatch")
    group, bits = options["group_size"], options["bits"]
    shape = tuple(options["source_shape"]) if spatial else source.meta.shape
    if spatial and (not shape or any(type(size) is not int or size < 0 for size in shape)
                    or prod(shape) != prod(options["matrix_shape"])):
        raise ValueError("invalid_convolution_source_geometry")
    groups = (prod(shape) + group - 1) // group
    words = (group + 32 // bits - 1) // (32 // bits)
    expected = (((groups, words), torch.int32, "packed"),
                ((groups,), torch.float32, "dense"), ((groups,), torch.float32, "dense"))
    if any((value.meta.shape, value.meta.dtype, value.meta.representation) != contract
           or value.meta.requires_grad or value.meta.device != source.meta.device
           for value, contract in zip(node.values, expected)):
        raise ValueError("physical_component_schema")
    # These defaults are intrinsic to codec.save_activation. Normalization
    # xhat, derivative packs and spatial encodings are different operations or
    # must declare their distinct representation/correction options explicitly.
    from deepopt._bf16_policy import resolve_bf16_policy

    policy = resolve_bf16_policy(options.get("bf16_policy", node.bf16_policy))
    if policy != node.bf16_policy:
        raise ValueError("pack_policy_contract_mismatch")
    recipe = {"bf16_policy": policy.as_dict(),
              "representation_role": "convolution_source" if spatial else "raw_saved_operand",
              "correction_recipe": None, **options}
    key = (source, version, _metadata(source.meta), (node.op, _freeze(recipe)),
           tuple(_metadata(v.meta) for v in node.values))
    description = {"source": source.name, "source_version": version,
                   "bf16_policy": policy.as_dict(),
                   "cast_boundary": str(options["source_cast"]),
                   "dtype": str(source.meta.dtype), "shape": source.meta.shape,
                   "stride": source.meta.stride, "layout": source.meta.layout,
                   "orientation": options["traversal"], "bits": options["bits"],
                   "group_size": options["group_size"],
                   "tail_elements": prod(shape) % options["group_size"],
                   "representation_role": str(recipe["representation_role"]),
                   "correction_recipe": repr(recipe["correction_recipe"]),
                   "rounding_domain": options["rounding_domain"]}
    return key, description


def _pure_between_packs(node):
    """Known source-preserving operations may separate shared projections.

    In particular gate convolution, SiLU, then up convolution is the VAE's
    public projection order. RNG, stateful FP8, running statistics, arbitrary
    extension calls and mutation remain barriers.
    """
    if isinstance(node, (InputNode, GetAttrNode, OutputNode)):
        return True
    if isinstance(node, PointwiseNode) and node.op in _PURE_POINTWISE:
        return True
    from .spatial import Convolution
    from .semantic import NativeActivation

    specification = getattr(node, "specification", None)
    return (type(specification) is Convolution or
            type(specification) is NativeActivation and not getattr(specification.module, "inplace", False))


def _tree(value, mapping):
    if isinstance(value, TensorValue):
        return mapping.get(value, value)
    if isinstance(value, tuple):
        return tuple(_tree(child, mapping) for child in value)
    if isinstance(value, list):
        return [_tree(child, mapping) for child in value]
    if isinstance(value, dict):
        return {key: _tree(child, mapping) for key, child in value.items()}
    return value


def _rewrite_node(node, mapping):
    for attribute in ("args", "kwargs", "result", "autograd_inputs", "autograd_buffers", "saved_tensors_for_bwd"):
        if hasattr(node, attribute):
            setattr(node, attribute, _tree(getattr(node, attribute), mapping))


def _rebuild(graph):
    for node in graph.nodes:
        for value in node.outputs():
            value.users.clear()
            value.producer = node
    for node in graph.nodes:
        register_uses(node)
    graph.validate()


def _record_names(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, tuple):
        return tuple(_record_names(child, mapping) for child in value)
    if isinstance(value, list):
        return [_record_names(child, mapping) for child in value]
    if isinstance(value, dict):
        return {key: _record_names(child, mapping) for key, child in value.items()}
    return value


def _profile_pack(node, encoding_id):
    """Retain causal encode evidence without marking CSE as kernel fusion."""
    label = "deepopt.pack::" + encoding_id
    function = node.target
    profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"

    @wraps(function)
    def execute(*args, **kwargs):
        scope = torch.profiler.record_function(label) if profile_ranges or torch.autograd._profiler_enabled() else nullcontext()
        with scope:
            return function(*args, **kwargs)

    node.target = execute
    digest = hashlib.sha256(repr(_freeze(node.specification)).encode())
    root = Path(__file__).resolve().parents[2]
    for relative in ("compiler/autodiff/aot_auto_diff.py", "compiler/autodiff/pack_fusion.py",
                     "opt/quantization/activation.py", "opt/backends/triton/quantization/activation/affine.py"):
        digest.update((root / relative).read_bytes())
    return {"profile_range": label, "implementation_fingerprint": digest.hexdigest(),
            "kernel_evidence": {"required_kernel_roles": {"encode": ["_quantize_pack"]},
                                "maximum_kernel_role_launches": {"encode": 1},
                                "forbidden_kernel_symbols": []}}


def share_packs(graph, trace, *, enabled=True):
    """Return an updated trace and metadata decisions after safe pack CSE.

    Both graphs are updated in place. Saved argument bindings are deduplicated
    and renumbered together, so code generation's ordinary use/lifetime logic
    keeps the physical payload and metadata alive for every remaining reader.
    Unknown operations are effect barriers; aliases and equal source names do
    not establish logical source identity.
    """
    graph.validate()
    trace.graph.validate()
    signature = tuple((getattr(node, "op", type(node).__name__), tuple(v.name for v in node.outputs()))
                      for node in graph.nodes)
    available, seen_sources, replacements, removed, decisions = {}, set(), {}, set(), []
    version = 0
    for position, node in enumerate(graph.nodes):
        if not isinstance(node, CallNode) or node.op not in _PACK_OPERATIONS:
            if not _pure_between_packs(node):
                available.clear()
                version += 1
            continue
        record = {"transformation": "pack_cse", "kernel_fusion": False, "execution_phase": "forward",
                  "node_name": node.values[0].name if node.values else "pack", "requested": bool(enabled),
                  "discovered": True, "eligible": None, "selected": False, "status": "disabled" if not enabled else "retained",
                  "reason": "disabled" if not enabled else "no_compatible_predecessor", "produced": True,
                  "original_components": tuple(v.name for v in node.values),
                  "produced_components": tuple(v.name for v in node.values), "estimated_removed_component_bytes": 0,
                  "observed_cuda_launches": None}
        decisions.append(record)
        try:
            key, description = _compatibility(node, version)
        except (TypeError, ValueError) as error:
            record.update(eligible=False, status="rejected" if enabled else "disabled", reason=str(error))
            continue
        record["compatibility"] = description
        record["encoding_id"] = hashlib.sha256(repr((signature, position, node.args[0].name, key[2:])).encode()).hexdigest()[:24]
        record["candidate_id"] = record["encoding_id"]
        if not enabled:
            previous = available.get(key)
            if previous is not None:
                record["would_share_with"] = previous[1]["encoding_id"]
            else:
                available[key] = (node, record)
            continue
        record["eligible"] = True
        previous = available.get(key)
        if previous is None:
            if node.args[0] in seen_sources:
                record["reason"] = "incompatible_representation_or_effect_interval"
            available[key] = (node, record)
            seen_sources.add(node.args[0])
            continue
        canonical, canonical_record = previous
        replacements.update(zip(node.values, canonical.values))
        removed.add(id(node))
        if "profile_range" not in canonical_record:
            canonical_record.update(_profile_pack(canonical, canonical_record["encoding_id"]))
            canonical_record.update(status="shared_producer", reason=None, expected_encode_operations=1)
        for name in ("profile_range", "implementation_fingerprint", "kernel_evidence"):
            record[name] = canonical_record[name]
        record.update(selected=True, status="shared", reason=None, produced=False,
                      shared_with=canonical_record["encoding_id"], expected_encode_operations=1,
                      produced_components=tuple(v.name for v in canonical.values),
                      estimated_removed_component_bytes=sum(prod(v.meta.shape) * v.meta.dtype.itemsize for v in node.values))
    if replacements:
        graph.nodes[:] = [node for node in graph.nodes if id(node) not in removed]
        for node in graph.nodes:
            _rewrite_node(node, replacements)
        saved = tuple(dict.fromkeys(replacements.get(value, value) for value in trace.saved_tensors))
        seed_count = len(trace.grad_output_source)
        indices = {value: seed_count + index for index, value in enumerate(saved)}
        aliases, bindings, discarded = {}, {}, set()
        for node in trace.graph.nodes:
            if not isinstance(node, InputNode) or node.index < seed_count:
                continue
            original = trace.saved_tensors[node.index - seed_count]
            source = replacements.get(original, original)
            if source in bindings:
                aliases[node.value] = bindings[source].value
                discarded.add(id(node))
            else:
                bindings[source] = node
                node.index = indices[source]
        trace.graph.nodes[:] = [node for node in trace.graph.nodes if id(node) not in discarded]
        for node in trace.graph.nodes:
            _rewrite_node(node, aliases)
        names = {old.name: new.name for old, new in replacements.items()}
        dependency_names = {**names, **{old.name: new.name for old, new in aliases.items()}}
        for node in (*graph.nodes, *trace.graph.nodes):
            node.scheduling_constraints = _record_names(node.scheduling_constraints, dependency_names)
        records = tuple(_record_names(record, names) for record in trace.saved_value_specs)
        trace = replace(trace, saved_tensors=saved, saved_value_specs=records)
        _rebuild(graph)
        _rebuild(trace.graph)
    # Reports own only serializable decisions, never source values/callables.
    json.dumps(decisions, allow_nan=False)
    return trace, tuple(decisions)
