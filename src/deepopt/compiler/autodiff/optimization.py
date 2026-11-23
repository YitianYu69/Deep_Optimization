"""Fusion over differentiated graphs, preserving forward residual identities."""

from functools import partial
from dataclasses import replace
from math import prod

import torch

from ..ir import CallNode, ExternKernelNode, GetAttrNode, InputNode, OutputNode, PointwiseNode, register_uses
from .semantic import BF16Linear, NativeActivation, recorded
from .fusion_contracts import candidate as fusion_candidate, select


def rebuild(graph):
    for node in graph.nodes:
        for value in node.outputs():
            value.users = []
            value.producer = node
    for node in graph.nodes:
        register_uses(node)
    graph.validate()


def _activation(node):
    if isinstance(node, PointwiseNode) and node.op in ("gelu", "silu"):
        return node.op, node.kwargs.get("approximate", "none")
    if isinstance(node, CallNode) and type(node.specification) is NativeActivation:
        spec = node.specification
        if spec.kind in ("gelu", "silu"):
            return spec.kind, spec.approximate
    return None


def _producer_residual_contract(producer, linear):
    """An operand pack does not imply that an activation VJP needs state."""
    from .semantic import packed_metas

    activation = producer.specification
    residuals = producer.values[1:]
    if activation.need_dx != producer.args[0].meta.requires_grad:
        return False, None, "producer_residual_gradient_contract"
    if len(residuals) != len(activation.saved_components):
        return False, None, "producer_residual_cardinality"
    if not activation.need_dx:
        return (not residuals), None, "producer_residual_gradient_contract"
    options = activation.compression
    if not (type(linear) is BF16Linear and linear.compressed and options["enabled"] and
            all(options[key] == linear.plan["activation_compression"][key] for key in ("bits", "group_size"))):
        return False, None, "derivative_codec_contract"
    expected = packed_metas(activation.geometry.matrix_shape, options, producer.args[0].meta.device)
    if len(residuals) != len(expected) or any(
            value.meta is None or value.meta.shape != metadata.shape or value.meta.dtype != metadata.dtype or
            value.meta.device != metadata.device or value.meta.requires_grad or
            value.meta.representation != metadata.representation
            for value, metadata in zip(residuals, expected)):
        return False, None, "producer_residual_metadata"
    return True, options, None


def forward(graph, context):
    """Select GEMM epilogues and activation/codec producer preparation."""
    from .fp8 import FP8Linear

    candidates, patterns = [], []
    for node in list(graph.nodes):
        if not isinstance(node, CallNode) or type(node.specification) not in (BF16Linear, FP8Linear):
            continue
        spec = node.specification
        source = node.args[0]
        producer = source.producer
        activation = _activation(producer)
        consumers = [user for user in node.values[0].users if not isinstance(user, OutputNode)]
        consumer = consumers[0] if len(consumers) == 1 else None
        epilogue = _activation(consumer)
        requested = context.fuse_deepopt_activations and "forward" in context.fusion_phases
        candidate = {**fusion_candidate(graph, node, spec, "forward", "gemm_activation", requested=requested),
            "module_path": spec.module_path, "node_name": spec.node_name,
            "phase": "train" if spec.module_training else "eval", "precision": spec.precision,
            "requested": requested, "effective_enabled": requested,
            "discovered": bool(activation or epilogue or getattr(consumer, "op", None) == "add"), "eligible": None, "selected": False,
            "status": "disabled", "reason": "disabled", "selected_schedule": None,
            "observed_cuda_launches": None}
        candidates.append(candidate)
        if not requested:
            continue
        candidate.update(status="rejected", eligible=False, reason="no_activation")
        if type(spec) is FP8Linear and (spec.delayed or any(block is not None for block in spec.blocks.values())):
            candidate["reason"] = "fp8_recipe_requires_separate_preparation"
            candidate["reason_detail"] = f"scaling_recipe={spec.options['scaling_recipe']}; blockwise/delayed preparation remains explicit"
            continue
        if torch.device(source.meta.device).type != "cuda":
            candidate["reason"] = "device"
            continue
        if type(spec) is BF16Linear and not spec.generated_kernels:
            candidate["reason"] = "generated_bf16_disabled"
            continue
        # The producer result may also escape through the saved-state exports.
        # Only remove it when the combined kernel supplies every escaped value.
        if activation and source.users == [node]:
            from ..scheduler import _is_contiguous

            if type(spec) is BF16Linear and not spec.compressed:
                candidate["reason"] = "no_packing_work"
                continue
            compression_options = (spec.plan["activation_compression"] if type(spec) is BF16Linear else spec.compression)
            if not _is_contiguous(producer.args[0].meta) or spec.compressed and compression_options["group_size"] > 16384:
                candidate["reason"] = "preparation_layout_or_group_size"
                continue
            between = graph.nodes[graph.nodes.index(producer) + 1:graph.nodes.index(node)]
            if any(not isinstance(item, (GetAttrNode, InputNode)) for item in between):
                candidate["reason"] = "intervening_operation"
                continue
            from ..activation_quantize import preparation_backend

            try:
                preparation = preparation_backend(
                    compression=spec.plan["activation_compression"] if type(spec) is BF16Linear and spec.compressed else
                    spec.compression if type(spec) is FP8Linear and spec.compressed else None,
                    policy=context.kernel_policy)
            except ValueError as error:
                candidate.update(reason="preparation_backend_conflict", reason_detail=str(error))
                continue
            derivative = None
            if (context.kernel_policy.pinned("activation", "activation.forward") and
                    context.kernel_policy.route("activation", "activation.forward") != preparation):
                candidate["reason"] = "activation_backend_conflict"
                continue
            if isinstance(producer, CallNode):
                valid, derivative, reason = _producer_residual_contract(producer, spec)
                if not valid:
                    candidate["reason"] = reason
                    continue
            if len(node.values) != 1 + len(spec.saved_components):
                candidate["reason"] = "consumer_residual_cardinality"
                continue
            if type(spec) is BF16Linear and spec.compressed:
                from .semantic import packed_metas

                expected = packed_metas(spec.matrix_shape, spec.plan["activation_compression"], source.meta.device)
                if any(value.meta is None or value.meta.shape != metadata.shape or value.meta.dtype != metadata.dtype
                       or value.meta.representation != metadata.representation or value.meta.requires_grad
                       for value, metadata in zip(node.values[1:], expected)):
                    candidate["reason"] = "consumer_residual_metadata"
                    continue
            if type(spec) is BF16Linear:
                spec.producer_activation = activation
                spec.producer_derivative = derivative
                spec.producer_preparation_backend = preparation
                spec.activation = activation
                spec.activation_state = derivative
                if derivative:
                    node.values += producer.values[1:]
                spec.forward_steps = ("fused_activation_pack", f"{spec.execution}_bf16_gemm")
            else:
                spec.activation = activation
                spec.forward_steps = (f"fused_{activation[0]}_operand_preparation", "cached_weight", "fp8_gemm")
            node.args = (producer.args[0], *node.args[1:])
            graph.nodes.remove(producer)
            pattern = f"{spec.precision}_{activation[0]}_producer_pack"
        elif type(spec) is BF16Linear and not spec.compressed and epilogue and isinstance(consumer, PointwiseNode):
            if (context.kernel_policy.pinned("activation", "activation.forward") and
                    context.kernel_policy.route("activation", "activation.forward") != spec.backend("forward")):
                candidate["reason"] = "activation_backend_conflict"
                continue
            between = graph.nodes[graph.nodes.index(node) + 1:graph.nodes.index(consumer)]
            if any(not isinstance(item, (GetAttrNode, InputNode)) for item in between):
                candidate["reason"] = "intervening_operation"
                continue
            spec.epilogue, spec.retain_preactivation = epilogue, True
            original = node.values[0]
            node.values = (consumer.value, original, *node.values[1:])
            graph.nodes.remove(consumer)
            spec.forward_steps = ("bf16_prepare", f"generated_bf16_gemm_{epilogue[0]}")
            pattern = f"bf16_{epilogue[0]}_gemm_epilogue"
        elif type(spec) is BF16Linear and isinstance(consumer, PointwiseNode) and consumer.op == "add" and node.values[0].users == [consumer]:
            from ..scheduler import _is_contiguous
            from ..ir import TensorValue

            residual = consumer.args[1] if consumer.args[0] is node.values[0] else consumer.args[0]
            if (not isinstance(residual, TensorValue) or residual.producer is node or residual.meta.dtype != torch.bfloat16
                    or residual.meta.shape != node.values[0].meta.shape or not _is_contiguous(residual.meta)
                    or consumer.kwargs.get("alpha", 1) != 1):
                candidate["reason"] = "residual_contract"
                continue
            if graph.nodes.index(residual.producer) >= graph.nodes.index(node):
                candidate["reason"] = "residual_dependency_not_available"
                continue
            spec.residual = True
            node.args += (residual,)
            node.values = (consumer.value, *node.values[1:])
            graph.nodes.remove(consumer)
            spec.forward_steps = ("bf16_prepare", "generated_bf16_gemm_residual")
            spec.backward_steps += ("residual_gradient",)
            pattern = "bf16_residual_gemm_epilogue"
        else:
            candidate["reason"] = "escaping_value_or_no_compatible_activation"
            continue
        candidate.update(eligible=True, selected=True, status="selected", reason="selected", pattern=pattern)
        node.target = recorded(spec, spec.forward, "forward")
        select(node, candidate, roles={"activation_pack" if "producer_pack" in pattern else "gemm_epilogue":
               ["_bf16_prepare" if "producer_pack" in pattern and spec.precision == "bf16" else
                "_bf16_gemm" if spec.precision == "bf16" else "_prepare"]},
               consumed=(getattr(producer, "op", ""), node.op) if "producer_pack" in pattern else
                        (node.op, getattr(consumer, "op", "")),
               source_files=("opt/backends/triton/gemm/bf16.py",) if spec.precision == "bf16" else ())
        if "producer_pack" in pattern:
            candidate["kernel_evidence"]["forbidden_kernel_symbols"] = ["_activation_forward"]
            if isinstance(producer, CallNode):
                activation_spec = producer.specification
                site = activation_spec.module_path or activation_spec.node_name
                candidate["kernel_evidence"]["removed_profile_ranges"] = [f"deepopt.compiler.{site}.forward.forward"]
        if type(spec) is BF16Linear:
            candidate["requested_schedule"] = {"backward_order": spec.backward_order}
            candidate["retained_preactivation_bytes"] = (original.meta.dtype.itemsize *
                prod(original.meta.shape) if spec.retain_preactivation else 0)
        patterns.append(pattern)
        rebuild(graph)
    return candidates, tuple(patterns)


def _bf16_derivative(spec, coefficient_spec, gradient, weight, *saved):
    from deepopt.opt.backends.bf16 import product
    from deepopt.opt.quantization.activation import PackedActivation

    packed = PackedActivation(*saved, coefficient_spec.geometry.matrix_shape,
        coefficient_spec.compression["group_size"], bits=coefficient_spec.compression["bits"])
    return product(gradient, weight.T, backend=spec.backend("dx"), derivative_packed=packed,
        tile=spec.tile, num_stages=spec.num_stages).reshape(coefficient_spec.input_meta.shape).to(coefficient_spec.input_meta.dtype)


def _bf16_native_derivative(spec, activation, gradient, weight, value):
    from deepopt.opt.backends.bf16 import product

    return product(gradient, weight.T, backend=spec.backend("dx"),
        derivative_input=value.reshape(gradient.shape[0], weight.shape[1]), derivative_activation=activation,
        tile=spec.tile, num_stages=spec.num_stages).reshape(value.shape).to(value.dtype)


def backward(graph, context=None, diagnostics=None):
    patterns = []
    enabled = context is None or (context.fuse_deepopt_activations and "backward" in context.fusion_phases)
    diagnostics = diagnostics if diagnostics is not None else []
    for node in list(graph.nodes):
        if not isinstance(node, (PointwiseNode, ExternKernelNode)) or node.op not in ("gelu_backward", "silu_backward"):
            continue
        gradient, value = node.args[:2]
        producer = gradient.producer
        spec = getattr(getattr(producer, "target", None), "specification", None)
        if getattr(producer, "op", None) != "bf16.gemm_dx" or type(spec) is not BF16Linear:
            continue
        pattern = f"bf16_dx_{node.op.removesuffix('_backward')}_epilogue"
        candidate = fusion_candidate(graph, producer, spec, "backward", pattern, requested=enabled)
        diagnostics.append(candidate)
        if not enabled:
            continue
        if (getattr(producer, "op", None) != "bf16.gemm_dx" or type(spec) is not BF16Linear
                or not spec.generated_kernels or gradient.users != [node] or value.meta.dtype != torch.bfloat16
                or value.meta.representation != "dense" or node.value.meta.dtype != torch.bfloat16):
            candidate["reason"] = "derivative_layout_dtype_fanout_or_backend"
            continue
        if spec.kernel_policy.pinned("activation", "activation.backward") and spec.kernel_policy.route(
                "activation", "activation.backward") != spec.backend("dx"):
            candidate["reason"] = "activation_backend_conflict"
            continue
        activation = (node.op.removesuffix("_backward"), node.kwargs.get("approximate", "none"))
        function = partial(_bf16_native_derivative, spec, activation)
        function.__name__ = "native_derivative_gemm"
        replacement = CallNode("bf16.gemm_dx_native_derivative", recorded(spec, function, "backward"),
                               (*producer.args, value), {}, (node.value,), spec)
        replacement.scheduling_constraints = _constraints(graph, producer, node, spec)
        select(replacement, candidate, roles={"gemm_derivative": ["_bf16_gemm"]},
               consumed=(producer.op, node.op), source_files=("opt/backends/triton/gemm/bf16.py",))
        candidate["requested_schedule"] = {"backward_order": spec.backward_order}
        position = graph.nodes.index(node)
        graph.nodes[position] = replacement
        graph.nodes.remove(producer)
        patterns.append(pattern)
        rebuild(graph)
    for node in list(graph.nodes):
        if not isinstance(node, CallNode) or getattr(node, "op", None) != "activation.coefficient_vjp":
            # VJP emission currently uses ExternKernelNode for single results.
            if getattr(node, "op", None) != "activation.coefficient_vjp":
                continue
        coefficient_spec = getattr(node.target, "specification", None)
        if coefficient_spec is None or coefficient_spec.input_meta.dtype != torch.bfloat16:
            continue
        if coefficient_spec.kind not in ("gelu", "silu", "sigmoid", "tanh") or len(node.args) != 4:
            continue
        source = node.args[0]
        producer = source.producer
        spec = getattr(getattr(producer, "target", None), "specification", None)
        if getattr(producer, "op", None) != "bf16.gemm_dx" or type(spec) is not BF16Linear:
            continue
        candidate = fusion_candidate(graph, producer, spec, "backward", "bf16_dx_packed_derivative_epilogue", requested=enabled)
        diagnostics.append(candidate)
        if not enabled:
            continue
        if (getattr(producer, "op", None) != "bf16.gemm_dx" or not isinstance(spec, BF16Linear)
                or not spec.generated_kernels or source.users != [node] or not coefficient_spec.compression["enabled"]):
            candidate["reason"] = "derivative_codec_dtype_fanout_or_backend"
            continue
        policy = spec.kernel_policy
        if any(policy.pinned(family, role) and policy.route(family, role) != spec.backend("dx")
               for family, role in (("activation", "activation.backward"), ("activation_codec", "activation.backward"),
                                    ("activation_codec", "linear.dx"))):
            candidate["reason"] = "activation_codec_backend_conflict"
            continue
        function = partial(_bf16_derivative, spec, coefficient_spec)
        function.__name__ = "packed_derivative_gemm"
        node.target = recorded(spec, function, "backward")
        node.scheduling_constraints = _constraints(graph, producer, node, spec)
        node.args = (*producer.args, *node.args[1:])
        node.op = "bf16.gemm_dx_packed_derivative"
        select(node, candidate, roles={"gemm_derivative": ["_bf16_gemm"]},
               consumed=(producer.op, "activation.coefficient_vjp"), source_files=("opt/backends/triton/gemm/bf16.py",))
        graph.nodes.remove(producer)
        patterns.append("bf16_dx_packed_derivative_epilogue")
        rebuild(graph)
    return tuple(patterns)


def _constraints(graph, producer, consumer, spec):
    between = graph.nodes[graph.nodes.index(producer) + 1:graph.nodes.index(consumer)]
    return {"requested_order": spec.backward_order,
            "prefer_before": tuple(value.name for node in between for value in node.outputs()),
            "dependencies": tuple(value.name for value in (*producer.inputs(), *consumer.inputs())
                                  if value.producer is not producer)}


def backward_layouts(graph):
    """Do not pass a guessed dense layout to a generated pointwise consumer.

    Native GEMM descriptors, transposes, expanded reductions and attention can
    return strided gradients. Unknown layouts remain legal native operands;
    only proven contiguous values may enter the current flat pointwise ABI.
    """
    from ..trace.metadata_utils import contiguous_stride

    dense_results = {"bf16.gemm_dx", "bf16.gemm_dw", "bf16.bias_reduce",
                     "bf16.gemm_dx_packed_derivative", "bf16.gemm_dx_native_derivative"}
    for node in graph.nodes:
        if isinstance(node, (InputNode, GetAttrNode, OutputNode)):
            continue
        dense = getattr(node, "op", "") in dense_results
        if isinstance(node, PointwiseNode):
            dense = all(value.meta is not None and value.meta.shape is not None
                        and value.meta.stride == contiguous_stride(value.meta.shape) for value in node.inputs())
        for value in node.outputs():
            if value.meta is not None and value.meta.representation == "dense":
                allocation = value.fresh_allocation
                proven = allocation is not None and value.meta == allocation.meta
                alias = node.scheduling_constraints.get("owned_output_alias", {})
                donor = next((operand for operand in node.inputs() if operand.name == alias.get("donor")), None)
                donor_proof = getattr(donor, "fresh_allocation", None)
                owned_alias = (getattr(node, "op", None) == "memory.donated_add_out"
                    and node.scheduling_constraints.get("mutation") == "invocation_owned_add_out"
                    and alias.get("result") == value.name and alias.get("scope") == "backward_invocation"
                    and donor_proof is not None and donor.meta == donor_proof.meta
                    and value.meta == donor_proof.meta)
                value.meta = replace(value.meta, stride=(allocation.meta.stride if proven else
                    donor_proof.meta.stride if owned_alias else
                    contiguous_stride(value.meta.shape) if dense else None))
