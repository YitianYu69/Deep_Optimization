"""Cold-path saved-state classification. Only metadata enters this module.

Logical tensor bytes are not allocator measurements. Runtime storage ledgers
remain responsible for pointer identity, tied storage, offsets and release.
"""

from collections import defaultdict
from math import prod

import torch


SAVED_STATE_SCHEMA_VERSION = 3
OFFLOADED_SAVED_STATE_SCHEMA_VERSION = 4
PACKED_TRANSPORT_SCHEMA_VERSION = 1
_MODEL_STATE = {"parameter", "persistent_buffer", "constant"}
_CODECS = {
    "packed_saved_operand": "codec.save_activation",
    "packed_convolution_source": "codec.save_convolution_source",
    "packed_normalized_value": {"autodiff_spatial", "autodiff_native_layer_norm"},
    "packed_activation_derivative": "autodiff_native_activation",
    "packed_linear_source": {"autodiff_bf16_linear", "autodiff_dense_linear"},
    "packed_storage_exp_coefficient": "autodiff_storage_exp",
}


def _shape(value):
    if not isinstance(value, (list, tuple)) or any(type(n) is not int or n < 0 for n in value):
        raise ValueError("missing or invalid shape")
    return tuple(value)


def _recipe(record):
    value = record.get("recipe", ())
    if isinstance(value, dict):
        return value
    result = dict(value)
    if len(result) != len(value):
        raise ValueError("duplicate recipe fields")
    return result


def _normalization_geometry(recipe):
    shape = _shape(recipe.get("source_shape"))
    if recipe.get("normalization_kind") == "layer":
        from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry

        geometry = LayerNormGeometry(shape, _shape(recipe.get("normalized_shape")))
        return geometry.matrix_shape, {(geometry.rows,)}
    from deepopt.opt.nn.normalization.geometry import NormGeometry

    geometry = NormGeometry(recipe.get("normalization_kind"), shape,
                            recipe.get("normalization_groups"), batch_stats=recipe.get("batch_stats", True))
    shapes = {(geometry.domains,)}
    if geometry.kind == "group":
        shapes.add((shape[0], geometry.groups))
    return geometry.matrix_shape, shapes


def _components(record, values, *, legacy=False):
    names = record.get("components", ())
    if not isinstance(names, (list, tuple)) or len(set(names)) != len(names):
        raise ValueError("invalid physical component names")
    descriptions = record.get("physical_components", ())
    if descriptions and tuple(item["name"] for item in descriptions) != tuple(names):
        raise ValueError("physical component ABI mismatch")
    if not descriptions and not legacy:
        raise ValueError("missing physical component contract")
    for description in descriptions:
        actual = values.get(description["name"])
        if actual is None:
            raise ValueError("missing saved physical component")
        if actual.get("absent"):
            if description.get("presence") != "optional":
                raise ValueError("required physical component is absent")
            continue
        if (description.get("shape") is not None and _shape(description["shape"]) != _shape(actual.get("shape"))
                or description.get("dtype") not in (None, "None") and description["dtype"] != actual.get("dtype")
                or description.get("device") is not None and description["device"] != actual.get("device")):
            raise ValueError("physical component shape/dtype/device mismatch")
    if any(name not in values for name in names):
        raise ValueError("missing saved physical component")
    return tuple(values[name] for name in names), descriptions


def _codec(record, values, legacy):
    from deepopt.opt.quantization.activation import _dimensions

    recipe = _recipe(record)
    bits, group = recipe.get("bits"), recipe.get("group_size")
    if type(bits) is not int or bits not in (1, 2, 4, 8) or type(group) is not int or group < 1:
        raise ValueError("invalid codec bits/group_size")
    representation = record["representation"]
    source = _shape(record.get("logical_shape"))
    if record.get("logical_dtype") not in ("torch.bfloat16", "torch.float32", "torch.float16", "torch.float64"):
        raise ValueError("invalid codec source dtype")
    matrix = _shape(recipe.get("matrix_shape", (prod(source[:-1]), source[-1]) if source else (1, 1)))
    if len(matrix) != 2:
        raise ValueError("invalid codec matrix geometry")
    block = recipe.get("block_shape")
    block = None if block is None else _shape(block)
    if block is not None and block != (1, group):
        raise ValueError("invalid codec block geometry")
    if representation == "packed_normalized_value":
        expected_matrix, _ = _normalization_geometry(recipe)
        if matrix != expected_matrix or _shape(recipe.get("source_shape")) != source or block != (1, group):
            raise ValueError("normalization codec geometry mismatch")
    elif representation == "packed_convolution_source":
        shape = _shape(recipe.get("source_shape"))
        if len(shape) < 3 or matrix != (shape[0] * prod(shape[2:]), shape[1]):
            raise ValueError("convolution codec geometry mismatch")
        if recipe.get("traversal") != "channels" or recipe.get("codec") not in ("auto", "direct", "materialized"):
            raise ValueError("invalid convolution codec traversal")
        unbatched = recipe.get("unbatched")
        if type(unbatched) is not bool or type(recipe.get("transposed")) is not bool:
            raise ValueError("missing convolution batch/transpose contract")
        original = (1, *source) if unbatched else source
        if len(original) != len(shape) or original[:2] != shape[:2]:
            raise ValueError("convolution logical/source geometry mismatch")
        padding = recipe.get("explicit_padding", ())
        synthetic = recipe.get("synthetic_padding_right", ())
        if (len(synthetic) != len(shape) - 2 or any(type(n) is not int or n < 0 for n in synthetic)
                or padding and (len(padding) != 2 * (len(shape) - 2) or any(type(n) is not int for n in padding))):
            raise ValueError("invalid convolution padding geometry")
        spatial = tuple(size + (padding[2 * (len(shape) - axis - 3)] + padding[2 * (len(shape) - axis - 3) + 1]
                                if padding else synthetic[axis]) for axis, size in enumerate(original[2:]))
        if shape[2:] != spatial or recipe.get("logical_dtype") != record["logical_dtype"]:
            raise ValueError("convolution padded source geometry mismatch")
    elif prod(matrix) != prod(source) or block is not None:
        raise ValueError("codec source geometry mismatch")
    if representation == "packed_saved_operand":
        if (_shape(recipe.get("logical_shape")) != source
                or recipe.get("logical_dtype") != record.get("logical_dtype")
                or recipe.get("traversal") != "logical_row_major"):
            raise ValueError("saved operand codec source mismatch")
    physical, descriptions = _components(record, values, legacy=legacy)
    if len(physical) != 3:
        raise ValueError("codec requires payload, step and minimum")
    if not legacy and record.get("logical_device") != physical[0].get("device"):
        raise ValueError("codec source/component device mismatch")
    groups, _, words = _dimensions(matrix, group, block, bits)
    roles = ("payload", "codec_step", "codec_minimum")
    expected = (((groups, words), "torch.int32"), ((groups,), "torch.float32"), ((groups,), "torch.float32"))
    constructors = _CODECS[representation]
    constructors = {constructors} if isinstance(constructors, str) else constructors
    for index, (actual, (shape, dtype)) in enumerate(zip(physical, expected)):
        if _shape(actual["shape"]) != shape or actual["dtype"] != dtype:
            raise ValueError(f"invalid {roles[index]} shape/dtype")
        if actual.get("device") is None or actual["device"] != physical[0].get("device"):
            raise ValueError("codec device mismatch")
        if descriptions and (descriptions[index].get("role") != roles[index]
                             or descriptions[index].get("constructor") not in constructors):
            raise ValueError(f"invalid {roles[index]} constructor/role")
    # Every membership must describe the same physical encoding, including its
    # source and traversal. Exact self-records for metadata do not shadow it.
    identity = (record.get("physical_source", record["source"]), representation, matrix, block, bits, group)
    return {actual["name"]: ("packed_payload" if index == 0 else "codec_metadata", roles[index], identity)
            for index, actual in enumerate(physical)}


def _auxiliary(record, values):
    recipe = _recipe(record)
    _, shapes = _normalization_geometry(recipe)
    physical, descriptions = _components(record, values)
    if len(physical) != 1:
        raise ValueError("normalization auxiliary requires one physical value")
    value, description = physical[0], descriptions[0]
    role = description.get("role")
    dtype = recipe.get("statistics_dtype", "torch.float32")
    if (role not in {"mean", "rstd", "center_shift", "moment_gain"}
            or description.get("constructor") not in {"autodiff_spatial", "autodiff_native_layer_norm"}
            or _shape(value["shape"]) not in shapes
            or dtype not in ("torch.bfloat16", "torch.float32", "torch.float64") or value["dtype"] != dtype):
        raise ValueError("invalid normalization auxiliary role/geometry/dtype")
    return {value["name"]: ("normalization_auxiliary", role, None)}


def _attention(record, values, compressed):
    from deepopt.opt.quantization.activation import _dimensions

    recipe = _recipe(record)
    physical, descriptions = _components(record, values)
    roles = ("output", "q_data", "q_step", "q_min", "k_data", "k_step", "k_min", "v_data", "v_step", "v_min",
             "lse", "probability", "drop_probability", "drop_mask", "seed", "offset", "cu_q", "cu_k", "implementation")
    if len(physical) != len(roles) or any(description.get("role") != role or description.get("constructor") != "autodiff_sdpa"
                                        for description, role in zip(descriptions, roles)):
        raise ValueError("invalid attention constructor/physical ABI")
    q, k, v = (_shape(recipe.get(name + "_shape")) for name in ("q", "k", "v"))
    if any(len(shape) != 4 for shape in (q, k, v)):
        raise ValueError("attention requires rank-four source geometry")
    state = dict(zip(roles, physical))
    implementation = state["implementation"].get("scalar")
    if type(implementation) is not int or implementation not in (0, 1, 2, 3, 10, 11):
        raise ValueError("missing or unsupported actual attention implementation")
    if (recipe.get("backend") not in ("native", "flash_attn_2")
            or (recipe["backend"] == "flash_attn_2") != (implementation == 10)):
        raise ValueError("attention backend/implementation mismatch")
    bits, group = recipe.get("bits"), recipe.get("group_size")
    if type(bits) is not int or bits not in (0, 1, 2, 4, 8) or type(group) is not int or group < 1:
        raise ValueError("invalid effective attention codec recipe")
    dtype = recipe.get("output_dtype")
    device = record.get("logical_device")
    if dtype not in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64") or not isinstance(device, str):
        raise ValueError("invalid attention output dtype/device")
    if compressed and bits == 0 and dtype in ("torch.bfloat16", "torch.float32") and any(prod(shape) >= 1024 for shape in (q, k, v)):
        raise ValueError("eligible attention operands bypass the requested compression")
    shapes = (q, k, v)
    if implementation == 10:
        shapes = tuple((shape[0], shape[2], shape[1], shape[3]) for shape in shapes)
    elif implementation == 0:
        if recipe.get("enable_gqa"):
            k, v = (k[0], q[1], *k[2:]), (v[0], q[1], *v[2:])
        batch = tuple(torch.broadcast_shapes(q[:-2], k[:-2]))
        score = (*batch, q[-2], k[-2])
        if recipe.get("mask_shape") is not None:
            score = tuple(torch.broadcast_shapes(score, _shape(recipe["mask_shape"])))
        value_batch = tuple(torch.broadcast_shapes(score[:-2], v[:-2]))
        shapes = ((*batch, *q[-2:]), (*batch, k[-1], k[-2]), (*value_batch, *v[-2:]))
    proof = {}
    for prefix, shape in zip(("q", "k", "v"), shapes):
        data, step, minimum = (state[prefix + suffix] for suffix in ("_data", "_step", "_min"))
        if any(value.get("device") != device for value in (data, step, minimum)):
            raise ValueError("attention operand/component device mismatch")
        if bits and prod(shape):
            groups, _, words = _dimensions((prod(shape[:-1]), shape[-1]), group, None, bits)
            if (_shape(data["shape"]) != (groups, words) or data["dtype"] != "torch.int32"
                    or any(_shape(value["shape"]) != (groups,) or value["dtype"] != "torch.float32" for value in (step, minimum))):
                raise ValueError(f"invalid packed attention {prefix.upper()} geometry/dtype")
            proof[data["name"]] = ("packed_payload", f"attention.{prefix}.payload", None)
            for value, role in ((step, "step"), (minimum, "minimum")):
                proof[value["name"]] = ("codec_metadata", f"attention.{prefix}.{role}", None)
        else:
            expected_dtype = "torch.float64" if dtype == "torch.float64" else "torch.float32" if implementation == 0 else dtype
            if _shape(data["shape"]) != shape or data["dtype"] != expected_dtype:
                raise ValueError("invalid exact attention operand")
            proof[data["name"]] = ("attention_exact_state", f"attention.{prefix}.exact_operand", None)
            if any(prod(_shape(value["shape"])) != 0 for value in (step, minimum)):
                raise ValueError("uncompressed attention contains unexpected codec metadata")
    output_batch = torch.broadcast_shapes(score[:-2], v[:-2]) if implementation == 0 else torch.broadcast_shapes(q[:-2], k[:-2], v[:-2])
    output_shape = (*tuple(output_batch), q[-2], v[-1])
    if (_shape(state["output"]["shape"]) != output_shape or state["output"]["dtype"] != dtype
            or state["output"].get("device") != device):
        raise ValueError("attention exact output geometry/dtype mismatch")
    score_shape = tuple(score) if implementation == 0 else (*tuple(torch.broadcast_shapes(q[:-2], k[:-2])), q[-2], k[-2])
    for role, value in state.items():
        if role in ("seed", "offset", "cu_q", "cu_k") and value.get("absent"):
            if (role in ("seed", "offset") and recipe.get("dropout_p", 0) and implementation in (1, 2, 3)
                    or role == "seed" and recipe.get("dropout_p", 0) and implementation == 10
                    or role in ("cu_q", "cu_k") and implementation == 3):
                raise ValueError(f"required attention {role} is absent")
        elif value.get("absent"):
            raise ValueError("required attention state is absent")
        elif role in ("probability", "drop_probability", "drop_mask"):
            expected = score_shape if implementation == 0 and (role != "drop_mask" or recipe.get("dropout_p", 0)) else (0,)
            if _shape(value["shape"]) != expected:
                raise ValueError(f"invalid attention {role} geometry")
            expected_dtype = "torch.bool" if role == "drop_mask" and prod(expected) else "torch.float64" if dtype == "torch.float64" else "torch.float32" if implementation == 0 and prod(expected) else dtype
            if value["dtype"] != expected_dtype or value.get("device") != device:
                raise ValueError(f"invalid attention {role} dtype")
        elif role == "lse":
            allowed = {(0,)} if implementation == 0 else {(*q[:-1],), (*q[:-1], 1)}
            if (_shape(value["shape"]) not in allowed or value.get("device") != device
                    or value["dtype"] != (dtype if implementation == 0 else "torch.float64" if dtype == "torch.float64" else "torch.float32")):
                raise ValueError("invalid attention LSE geometry/dtype")
        elif role == "implementation" and (_shape(value["shape"]) != () or value["dtype"] != "torch.int64" or value["device"] != "cpu"):
            raise ValueError("invalid attention implementation selector")
        elif role in ("seed", "offset") and (value["dtype"] != "torch.int64"
                or _shape(value["shape"]) not in ((), (1,), (2,) if role == "seed" and implementation == 10 else ())
                or value.get("device") not in ("cpu", device)):
            raise ValueError("invalid attention RNG state")
        elif role in ("cu_q", "cu_k") and (_shape(value["shape"]) not in ((0,), (q[0] + 1,))
                or value["dtype"] != "torch.int32" or value.get("device") != device):
            raise ValueError("invalid attention cumulative lengths")
        if value["name"] not in proof:
            proof[value["name"]] = ("attention_exact_state", f"attention.{role}", None)
    return proof


def _exact_exp(record, value, policy):
    consumers = record.get("consumers", ())
    try:
        shape_matches = _shape(record.get("logical_shape")) == _shape(value["shape"])
    except (ValueError, TypeError):
        return False
    return (record.get("producer") == "autodiff_primitive_exp"
            and policy == record.get("bf16_policy") == {"name": "torch_amp", "version": 1}
            and record.get("source") == record.get("physical_source") == value["name"]
            and shape_matches
            and record.get("logical_dtype") == value["dtype"] in ("torch.bfloat16", "torch.float32")
            and bool(consumers) and all(
                consumer.get("operation") == "autodiff_primitive_exp"
                and tuple(consumer.get("node", ())) == (value["name"],)
                and consumer.get("role") == "exp.forward_result_coefficient"
                and consumer.get("allow_approximation") is False
                and consumer.get("rounding_domain") == "forward_result_dtype"
                and consumer.get("retention_reason") == "consumer_requires_exact"
                for consumer in consumers))


def _exact_stochastic_mask(record, value):
    if record.get("producer") != "stochastic_depth.snapshot":
        return False
    recipe = _recipe(record)
    shape = _shape(value["shape"])
    consumers = record.get("consumers", ())
    survival = recipe.get("survival_probability")
    return (record.get("physical_source") == record.get("source") == value["name"]
            and _shape(recipe.get("mask_shape")) == _shape(record.get("logical_shape")) == shape
            and len(shape) >= 1 and all(size == 1 for size in shape[1:])
            and record.get("logical_dtype") == value["dtype"] in ("torch.bfloat16", "torch.float32")
            and type(survival) in (int, float) and 0 <= survival <= 1 and type(recipe.get("scaled")) is bool
            and bool(consumers) and all(consumer.get("role") == "stochastic_depth.scaled_mask"
                and consumer.get("allow_approximation") is False
                and consumer.get("rounding_domain") == "native_mask_dtype" for consumer in consumers))


def _model_copy(record, value, sources):
    origin, transforms = record.get("model_origin"), record.get("model_transforms")
    if not origin or not transforms or record.get("storage_source"):
        return False
    if any(operation not in {"clone", "to", "float", "bfloat16", "half", "double", "type_as", "contiguous",
                              "reshape", "view", "flatten", "unflatten", "squeeze", "unsqueeze", "transpose",
                              "t", "permute", "detach"} for operation in transforms):
        return False
    declarations = sources.get(origin.get("source"), ())
    constructors = {"InputNode", "autodiff_spatial", transforms[-1], "autodiff_primitive_" + transforms[-1]}
    if (any(component.get("constructor") not in constructors for component in record.get("physical_components", ()))
            or origin.get("dtype") != value["dtype"] and not any(operation in {
                "to", "float", "bfloat16", "half", "double", "type_as"} for operation in transforms)):
        return False
    return (bool(declarations) and all(
        source.get("provenance") in _MODEL_STATE and _shape(source.get("shape")) == _shape(origin.get("shape"))
        and source.get("dtype") == origin.get("dtype") and source.get("device") == origin.get("device")
        for source in declarations) and value.get("device") == origin.get("device")
        and prod(value["shape"]) == prod(origin["shape"]))


def _lossless_mask(record, values, legacy):
    physical, descriptions = _components(record, values, legacy=legacy)
    recipe = _recipe(record)
    if (len(physical) != 1 or physical[0]["dtype"] != "torch.int32"
            or _shape(physical[0]["shape"]) != ((prod(_shape(record.get("logical_shape"))) + 31) // 32,)
            or descriptions and (descriptions[0].get("role") != "mask"
                                 or descriptions[0].get("constructor") != "autodiff_scalar_clamp")
            or not descriptions and not any(consumer.get("operation") == "autodiff_scalar_clamp"
                and consumer.get("role") == "input_gradient_branch" for consumer in record.get("consumers", ()))
            or type(recipe.get("include_minimum")) is not bool or type(recipe.get("include_maximum")) is not bool
            or recipe.get("minimum") is None and recipe.get("maximum") is None):
        raise ValueError("invalid lossless clamp mask contract")
    return {physical[0]["name"]: ("lossless_residual", "clamp_mask", None)}


def _transport_consumer_observation(static, observation, transport_components):
    """Validate host residency separately from the unchanged CUDA consumer ABI.

    This is a metadata/report reader. It neither restores tensors nor grants a
    CPU tensor permission to bypass a generated CUDA input guard.
    """
    envelope = observation.get("transport")
    if not isinstance(envelope, dict) or type(envelope.get("version")) is not int or envelope["version"] != PACKED_TRANSPORT_SCHEMA_VERSION:
        raise AssertionError("Incompatible packed activation transport envelope")
    producer, consumer = envelope.get("producer_abi"), envelope.get("consumer_abi")
    if not isinstance(producer, dict) or not isinstance(consumer, dict):
        raise AssertionError("Packed transport requires producer and consumer ABI metadata")
    component = transport_components.get(static.get("name"))
    if (component is None or envelope.get("bundle_id") != component[0]
            or envelope.get("saved_index") != component[1].get("saved_index")
            or type(envelope.get("generation")) is not int or envelope["generation"] < 1):
        raise AssertionError("Packed transport contradicts its compiled bundle/slot/generation")
    for abi in (producer, consumer, component[1]):
        for key in ("shape", "dtype", "device"):
            expected, actual = static.get(key), abi.get(key)
            if expected is None or (tuple(expected) if key == "shape" else expected) != (tuple(actual) if key == "shape" and actual is not None else actual):
                raise AssertionError(f"Packed transport contradicts static {key}: {static.get('name')}")
        expected_stride = component[1].get("stride")
        if expected_stride is not None and tuple(abi.get("stride") or ()) != tuple(expected_stride):
            raise AssertionError(f"Packed transport contradicts compiled stride: {static.get('name')}")
    if (not isinstance(producer.get("device"), str) or torch.device(producer["device"]).type != "cuda"
            or observation.get("device") != "cpu"
            or envelope.get("residency") not in ("host_pinned", "host_pageable")
            or type(envelope.get("saved_index")) is not int or envelope["saved_index"] < 0
            or not isinstance(envelope.get("bundle_id"), (str, int))
            or isinstance(envelope.get("bundle_id"), bool)):
        raise AssertionError("Invalid packed transport residency or identity")
    for key in ("shape", "dtype"):
        actual = observation.get(key)
        if (tuple(actual) if key == "shape" and actual is not None else actual) != (tuple(static[key]) if key == "shape" else static[key]):
            raise AssertionError(f"Packed host snapshot contradicts static {key}: {static.get('name')}")
    if component[1].get("stride") is not None and tuple(observation.get("stride") or ()) != tuple(component[1]["stride"]):
        raise AssertionError(f"Packed host snapshot contradicts compiled stride: {static.get('name')}")
    return {**observation, "device": consumer["device"],
            "storage_residency": envelope["residency"], "storage_device": "cpu"}


def classify_saved_state(region, compressed):
    """Validate each physical save against all its owners, independent of order.

    Schema 1 codecs can be read when their existing recipe proves geometry.
    Ownership and exact-coefficient exceptions require the explicit schema 2
    metadata. Unknown versions are rejected rather than treated as cache hits.
    """
    version = region.get("saved_state_schema_version", 1)
    offload = region.get("packed_activation_offload", {})
    if (type(version) is not int or version not in (1, 2, SAVED_STATE_SCHEMA_VERSION, OFFLOADED_SAVED_STATE_SCHEMA_VERSION)
            or version == OFFLOADED_SAVED_STATE_SCHEMA_VERSION and
            (not isinstance(offload, dict) or type(offload.get("transport_envelope_version")) is not int
             or offload["transport_envelope_version"] != PACKED_TRANSPORT_SCHEMA_VERSION)):
        raise AssertionError(f"Incompatible native saved-state schema: {version}")
    transport_components = {}
    if version == OFFLOADED_SAVED_STATE_SCHEMA_VERSION:
        plan = offload.get("plan")
        if not isinstance(plan, dict) or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
            raise AssertionError("Incompatible packed transport plan schema")
        for bundle in plan.get("bundles", ()):
            if not isinstance(bundle, dict) or not isinstance(bundle.get("components"), (list, tuple)) or len(bundle["components"]) != 3:
                raise AssertionError("Invalid compiled packed transport bundle")
            for component in bundle["components"]:
                if (not isinstance(component, dict) or not isinstance(component.get("name"), str)
                        or component["name"] in transport_components):
                    raise AssertionError("Invalid compiled packed transport component membership")
                transport_components[component["name"]] = bundle.get("bundle_id"), component
    values, members, proofs = {}, defaultdict(list), defaultdict(list)
    records = region.get("saved_value_bindings", ())
    if not isinstance(records, (list, tuple)) or any(not isinstance(record, dict) for record in records):
        raise AssertionError("Invalid native saved-state binding schema")
    model_sources = defaultdict(list)
    for source in region.get("model_state_sources", ()):
        model_sources[source.get("source")].append(source)
    runtime = region.get("runtime_saved_state", {})
    observed = {value["name"]: value for value in runtime.get("bindings", ())} if runtime.get("metadata_only") is True else {}
    for static in region.get("saved_state", ()):
        value = dict(static)
        observation = observed.get(value.get("name"))
        if observation is not None:
            if observation.get("transport") is not None:
                if version != OFFLOADED_SAVED_STATE_SCHEMA_VERSION:
                    raise AssertionError("Packed transport requires native saved-state schema 4")
                observation = _transport_consumer_observation(value, observation, transport_components)
            if observation.get("presence") == "absent":
                if value.get("presence") != "optional":
                    raise AssertionError(f"Required native save is absent: {value.get('name')}")
                value.update(observation, absent=True)
                values[value["name"]] = value
                continue
            for field in ("shape", "dtype", "device"):
                if value.get(field) is not None and (tuple(value[field]) if field == "shape" else value[field]) != (tuple(observation[field]) if field == "shape" else observation[field]):
                    raise AssertionError(f"Actual native save contradicts static {field}: {value.get('name')}")
            value.update(observation)
        name = value.get("name")
        try:
            _shape(value.get("shape"))
            dtype = getattr(torch, value.get("dtype", "").removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("missing or invalid dtype")
            if not isinstance(name, str) or not name or name in values:
                raise ValueError("missing or duplicate physical name")
        except (ValueError, TypeError, AttributeError) as error:
            raise AssertionError(f"Untyped native VAE save: {name}: {error}") from error
        values[name] = value
    for record in records:
        representation = record.get("representation")
        if not representation:
            raise AssertionError(f"Unclassified native VAE save: {record.get('source')}")
        names = record.get("components", (record.get("source"),))
        if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) for name in names):
            raise AssertionError("Invalid native saved-state component schema")
        for name in names:
            members[name].append(record)
        try:
            if record.get("physical_components"):
                _components(record, values)
            classified = (_codec(record, values, legacy=version == 1) if representation in _CODECS else
                          _auxiliary(record, values) if representation == "normalization_auxiliary" else
                          _attention(record, values, compressed) if representation == "attention_state" else
                          _lossless_mask(record, values, legacy=version == 1) if representation == "lossless_clamp_mask" else {})
            for name, proof in classified.items():
                proofs[name].append(proof)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise AssertionError(f"Invalid native saved-state contract ({representation}, {record.get('source')}): {error}") from error
    result = []
    for name, value in values.items():
        bindings = members[name]
        if not bindings:
            raise AssertionError(f"Unclassified native VAE save: {name}")
        category, role = "unexplained_dense_activation", "value"
        explained_copy = False
        candidates = proofs[name]
        if candidates:
            if len(set(candidates)) != 1:
                raise AssertionError(f"Conflicting native saved-state component ownership: {name}")
            category, role, _ = candidates[0]
        if value.get("absent") and category != "attention_exact_state":
            raise AssertionError(f"Unverified optional native save: {name}")
        aliases = {record["storage_source"] for record in bindings if record.get("storage_source")
                   and record.get("provenance") in _MODEL_STATE
                   and record.get("physical_source") == name and tuple(record.get("components", ())) == (name,)}
        if aliases:
            if len(aliases) != 1 or candidates or any(
                    record.get("physical_source") == name and record.get("storage_source") not in aliases
                    for record in bindings):
                raise AssertionError(f"Conflicting native model-state ownership: {name}")
            declarations = model_sources[next(iter(aliases))]
            if not declarations or any(
                    source.get("provenance") not in _MODEL_STATE
                    or source.get("dtype") != value["dtype"] or source.get("device") != value.get("device")
                    for source in declarations):
                raise AssertionError(f"Invalid native model-state source: {name}")
            for binding in bindings:
                if binding.get("storage_source"):
                    try:
                        _, descriptions = _components(binding, values)
                        for declaration in declarations:
                            _shape(declaration.get("shape"))
                        if any(description.get("constructor") not in {
                                "InputNode", "GetAttrNode", "autodiff_spatial", "autodiff_primitive_view",
                                "autodiff_primitive_transpose", "autodiff_primitive_t", "autodiff_primitive_permute",
                                "autodiff_primitive_squeeze", "autodiff_primitive_unsqueeze", "autodiff_primitive_expand",
                                "autodiff_primitive_expand_as", "autodiff_primitive_narrow", "autodiff_primitive_detach",
                                "autodiff_primitive_to", "autodiff_primitive_float", "autodiff_primitive_bfloat16",
                                "autodiff_primitive_contiguous", "to", "view", "transpose", "t", "permute"}
                               for description in descriptions):
                            raise ValueError("unproven alias constructor")
                    except (ValueError, TypeError, KeyError) as error:
                        raise AssertionError(f"Invalid native model-state alias: {name}: {error}") from error
            category, role = "persistent_model_state_reference", "model_alias"
        elif not candidates:
            direct = [record for record in bindings if record.get("physical_source", record.get("source")) == name]
            if any(record.get("provenance") in _MODEL_STATE for record in direct):
                category = "derived_model_state_copy"
                try:
                    explained_copy = all(_model_copy(record, value, model_sources) for record in direct)
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    raise AssertionError(f"Invalid native model-state copy contract: {name}: {error}") from error
            elif direct and all(_exact_stochastic_mask(record, value) for record in direct):
                category, role = "lossless_residual", "stochastic_depth.scaled_mask"
            elif direct and all(_exact_exp(record, value, region.get("bf16_policy")) for record in direct):
                try:
                    for record in direct:
                        _, descriptions = _components(record, values)
                        if any(description.get("constructor") != "autodiff_primitive_exp" for description in descriptions):
                            raise ValueError("unproven exact-coefficient constructor")
                except (ValueError, TypeError, KeyError) as error:
                    raise AssertionError(f"Invalid native exact-coefficient contract: {name}: {error}") from error
                category, role = "approved_exact_coefficient", "exp.forward_result_coefficient"
        count = 0 if value.get("absent") else prod(value["shape"])
        if (compressed and value["dtype"] in ("torch.bfloat16", "torch.float32") and count >= 1024
                and (category == "unexplained_dense_activation" or category == "derived_model_state_copy" and not explained_copy)):
            raise AssertionError(f"Unexplained large dense native VAE save: {name} ({category})")
        dtype = None if value.get("absent") else getattr(torch, value["dtype"].removeprefix("torch."))
        result.append({"name": name, "category": category, "role": role,
                       "bytes": count * dtype.itemsize if dtype is not None else 0,
                       "presence": "absent" if value.get("absent") else "present", "storage_source": next(iter(aliases), None),
                       "representations": sorted({record["representation"] for record in bindings}),
                       "consumers": sorted({(consumer.get("operation"), tuple(consumer.get("node", ())), consumer.get("role"))
                                            for record in bindings for consumer in record.get("consumers", ())}, key=repr)})
        if value.get("transport") is not None:
            result[-1].update(storage_residency=value["storage_residency"], storage_device=value["storage_device"],
                              consumer_device=value["device"], transport=value["transport"])
    return tuple(result)
