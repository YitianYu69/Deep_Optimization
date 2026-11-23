"""Specialized SwiGLU fusion over native-derived, unchanged saved leaves."""

from math import prod
import hashlib
import json
import operator
from types import CodeType

import torch

from ..ir import CallNode, ExternKernelNode, GetAttrNode, InputNode, PointwiseNode, TensorValue
from ..trace.metadata_utils import contiguous_stride
from .fusion_contracts import candidate, select
from .semantic import NativeActivation, recorded
from .training_fusion import TrainingFusionABI


ACTIVATION_PACK = "swiglu_activation_pack"
FORWARD = "swiglu_forward_residuals"
BACKWARD = "swiglu_backward_residuals"
_SOURCES = ("compiler/autodiff/swiglu_fusion.py", "opt/backends/triton/activation/swiglu.py",
            "opt/backends/triton/activation/math.py", "opt/backends/triton/quantization/activation/affine.py")
_ROLES = tuple(f"{name}.{component}" for name in ("derivative", "rounded_gate", "raw_up")
               for component in ("payload", "step", "minimum"))
_ACTIVATION_ABI = TrainingFusionABI(ACTIVATION_PACK, ("gate", *_ROLES[:6]), ("gate_bf16",))
_FORWARD_ABI = TrainingFusionABI(FORWARD, ("product", *_ROLES), ("gate_bf16", "product_bf16"))
_BACKWARD_ABI = TrainingFusionABI(BACKWARD, ("preactivation_gradient", "up_gradient"),
    ("decoded_operands_bf16", "product_adjoint_bf16", "activation_adjoint_bf16"), iteration_layout="logical_strided_reads")


def _abi(record, abi, outputs, stage):
    receipt = abi.receipt(outputs)
    record["training_fusion_abi"] = receipt
    identity = json.dumps(dict(stage=stage, abi=receipt, torch=str(torch.__version__), cuda=torch.version.cuda), sort_keys=True)
    record["implementation_fingerprint"] = hashlib.sha256(
        (record["implementation_fingerprint"] + identity).encode()).hexdigest()


def _stage(context):
    stage = getattr(context, "swiglu_fusion_stage", "off")
    if stage not in ("off", "activation_pack", "forward", "forward_backward"):
        raise ValueError("Unknown SwiGLU fusion stage")
    return stage


def _dense(value, shape=None, *, strided=False):
    meta = value.meta
    return (meta is not None and meta.device is not None and torch.device(meta.device).type == "cuda"
            and meta.dtype == torch.bfloat16 and meta.representation == "dense" and meta.storage_dtype is None
            and (strided or meta.stride in (None, contiguous_stride(meta.shape)))
            and (shape is None or meta.shape == shape))


def _pack(graph, value):
    matches = [node for node in graph.nodes if isinstance(node, CallNode)
               and node.op == "codec.save_activation" and node.args == (value,)]
    return matches[0] if len(matches) == 1 else None


def _pack_reason(node, shape):
    if node is None or len(node.values) != 3 or node.kwargs:
        return "three_leaf_operand_save_required"
    if not _codec_function(node.target, "pack", shape):
        return "untrusted_operand_save_executor"
    options = dict(node.specification)
    if (options.get("bits") != 2 or options.get("group_size") != 1024
            or options.get("logical_dtype") != torch.bfloat16
            or tuple(options.get("logical_shape", ())) != shape
            or options.get("traversal") != "logical_row_major"
            or options.get("source_cast") is not None
            or options.get("restored_stride") != contiguous_stride(shape)):
        return "operand_codec_contract"
    return None


def _leaves(values, shape, device=None):
    groups = (prod(shape) + 1023) // 1024
    return len(values) == 3 and all(value.meta is not None and value.meta.shape == dimensions
        and value.meta.dtype == dtype and value.meta.storage_dtype is None and value.meta.device is not None
        and (device is None or torch.device(value.meta.device) == torch.device(device))
        and value.meta.stride in (None, contiguous_stride(dimensions))
        for value, dimensions, dtype in zip(values, ((groups, 64), (groups,), (groups,)),
                                            (torch.int32, torch.float32, torch.float32)))


def _codec_function(function, name, shape):
    from .aot_auto_diff import BackwardTraceContext

    expected = next(code for code in BackwardTraceContext._store_codec.__code__.co_consts
                    if isinstance(code, CodeType) and code.co_name == name)
    if getattr(function, "__code__", None) is not expected:
        return False
    closure = dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))
    valid = (closure.get("bits") == 2 and closure.get("group") == 1024
             and closure.get("shape") == (prod(shape[:-1]), shape[-1]))
    if name == "restore":
        metadata = closure.get("metadata")
        valid = valid and (metadata is not None and metadata.shape == shape and metadata.dtype == torch.bfloat16
                           and closure.get("restore_stride") == contiguous_stride(shape))
    return valid


def _effects(node):
    if node.scheduling_constraints:
        return True
    return any(getattr(owner, name, False) for owner in (node, getattr(node, "target", None))
               for name in ("stateful", "effects", "side_effects", "mutation", "mutates", "alias_writes")) or any(
        getattr(owner, name, None) is not None for owner in (node, getattr(node, "target", None))
        for name in ("stream", "stream_id"))


def _multiply(node):
    from ..trace.register_ops import get_trace_op_node

    return (isinstance(node, PointwiseNode) and node.op == "mul" and not node.kwargs
            and any(node.target is function for function in (operator.mul, torch.mul, get_trace_op_node("mul").target)))


def _commutable(node):
    """Only built-in read-only work may separate the fused source operations."""
    if _effects(node):
        return False
    if isinstance(node, (InputNode, GetAttrNode)):
        return True
    if isinstance(node, PointwiseNode) and not node.kwargs:
        from ..trace.register_ops import get_trace_op_node

        return node.op in ("add", "sub", "mul", "neg") and node.target is get_trace_op_node(node.op).target
    if isinstance(node, (CallNode, ExternKernelNode)):
        from .spatial import Convolution

        spec = getattr(node.target, "specification", None)
        function = getattr(node.target, "numerical_function", None)
        return (type(spec) is Convolution and spec.precision == "native" and "fp8_training" not in spec.plan
            and getattr(function, "__self__", None) is spec
            and any(getattr(function, "__func__", None) is method for method in
                    (Convolution.forward, Convolution.pack_source, Convolution.dx_db, Convolution.dw))
            and getattr(node.target, "__code__", None) is recorded(spec, function, "forward").__code__)
    return False


class _Run:
    activation = ("silu", "none")

    def __init__(self, activation, shape, product):
        self.activation_spec, self.shape, self.product = activation, shape, product

    def __getattr__(self, name):
        return getattr(self.activation_spec, name)

    def forward(self, x, up=None):
        if (type(x) is not torch.Tensor or x.device.type != "cuda" or x.dtype != torch.bfloat16
                or tuple(x.shape) != self.shape or not x.is_contiguous()
                or self.product and (type(up) is not torch.Tensor or up.device != x.device
                    or up.dtype != x.dtype or tuple(up.shape) != self.shape or not up.is_contiguous())):
            raise ValueError("SwiGLU fusion requires its captured contiguous CUDA BF16 operands")
        from deepopt.opt._execution import finite_checks_enabled
        from deepopt.opt.backends.triton.activation.swiglu import forward

        values = forward(x, up, shape=self.shape, product=self.product, check_finite=finite_checks_enabled())
        self.module.last_execution = dict(backend="triton", activation="silu", swiglu_fused=True,
            swiglu_stage="forward" if self.product else "activation_pack", forward_launches=1,
            saved_state_bytes=sum(value.numel() * value.element_size() for value in values[1:]))
        return self.output_abi.check_runtime(values, self.output_metadata)

    def backward(self, gradient, *saved):
        if (type(gradient) is not torch.Tensor or gradient.device.type != "cuda"
                or gradient.dtype != torch.bfloat16 or tuple(gradient.shape) != self.shape
                or any(stride < 0 for stride in gradient.stride())):
            raise ValueError("SwiGLU fusion requires its captured CUDA BF16 cotangent")
        from deepopt.opt.backends.triton.activation.swiglu import backward

        values = backward(gradient, *saved, shape=self.shape)
        return self.output_abi.check_runtime(values, self.output_metadata)


class SwiGLUPackedOffloadPanel:
    """Private, named consumer capability installed only by the proven pass.

    The runtime owns staging/copy/event lifetimes. This adapter owns the same
    numerical ABI and the two final gradient destinations as ordinary VJP.
    """

    bits = 2
    group_size = 1024
    kind = "swiglu_backward_v1"

    def __init__(self, runtime):
        if type(runtime) is not _Run or not runtime.product:
            raise ValueError("Packed panels require the complete built-in SwiGLU VJP")
        self.shape = tuple(runtime.shape)
        self.output_abi = runtime.output_abi
        self.output_metadata = tuple(runtime.output_metadata)

    def allocate_outputs(self, gradient):
        from deepopt.opt.backends.triton.activation.swiglu import allocate_panel_outputs

        return self.output_abi.check_runtime(allocate_panel_outputs(gradient, shape=self.shape), self.output_metadata)

    def run(self, gradient, outputs, triplets, start_element, element_count):
        from deepopt.opt.backends.triton.activation.swiglu import backward_panel

        backward_panel(gradient, outputs, triplets, start_element, element_count, shape=self.shape)


def _forward_reason(graph, activation, multiply, gate, up, gate_pack, up_pack, product, context):
    spec = activation.specification
    shape = gate.meta.shape if gate.meta is not None else ()
    function = getattr(activation.target, "numerical_function", None)
    if (type(spec) is not NativeActivation or getattr(function, "__self__", None) is not spec
            or getattr(function, "__func__", None) is not NativeActivation.forward
            or getattr(activation.target, "__code__", None) is not recorded(spec, spec.forward, "forward").__code__):
        return "untrusted_activation_executor"
    if (type(spec) is not NativeActivation or spec.kind != "silu" or spec.native_torch_activation
            or spec.recompute or not spec.need_dx or len(activation.values) != 4
            or spec.output_dtype != torch.bfloat16 or spec.input_meta.dtype != torch.bfloat16
            or spec.input_meta.shape != shape or spec.geometry.layout != "last"
            or not spec.compression["enabled"] or spec.compression["bits"] != 2
            or spec.compression["group_size"] != 1024):
        return "packed_silu_derivative_abi_required"
    if (len(shape) != 4 or not prod(shape) or len(activation.args) != 3
            or activation.args[1:] != (None, None) or activation.kwargs or multiply.kwargs
            or not _multiply(multiply)
            or any(not _dense(value, shape) for value in (activation.args[0], gate, up, multiply.value))
            or activation.args[0].meta.stride != contiguous_stride(shape)
            or up.meta.stride != contiguous_stride(shape)):
        return "equal_shape_contiguous_cuda_bf16_required"
    policy = getattr(context, "kernel_policy", None)
    if policy is not None and any(policy.route(family, role) != "triton" for family, role in
            (("activation", "activation.forward"), ("activation_codec", "compiler.forward"),
             ("compiler_pointwise", "compiler.forward"))):
        return "kernel_policy_conflict"
    reason = _pack_reason(gate_pack, shape)
    if reason:
        return reason
    device = activation.args[0].meta.device
    if (any(torch.device(value.meta.device) != torch.device(device) for value in (gate, up, multiply.value))
            or not _leaves(activation.values[1:], shape, device) or not _leaves(gate_pack.values, shape, device)):
        return "saved_leaf_metadata"
    if any(_effects(node) for node in (activation, gate_pack, multiply)):
        return "effect_or_stream_override"
    if product:
        reason = _pack_reason(up_pack, shape)
        if reason:
            return reason
        if len(gate.users) != 2 or any(user is not multiply and user is not gate_pack for user in gate.users):
            return "gate_fanout_requires_dense_output"
        if not _leaves(up_pack.values, shape, device) or _effects(up_pack):
            return "saved_leaf_metadata"
        if any(value.users for value in activation.values[1:]):
            return "forward_derivative_consumer"
        removed = (activation, gate_pack, up_pack, multiply)
        between = graph.nodes[graph.nodes.index(activation) + 1:graph.nodes.index(multiply)]
        if any(not any(node is current for current in removed) and not _commutable(node) for node in between):
            return "intervening_effect_or_opaque_executor"
    saved_outputs = (*activation.values[1:], *gate_pack.values, *(up_pack.values if product else ()))
    originals = (activation, gate_pack, *((up_pack, multiply) if product else ()))
    if any(not any(value is output for output in saved_outputs)
           for node in originals for value in node.saved_tensors_for_bwd):
        return "additional_saved_owner_contract"
    return None


def forward(graph, context):
    from .optimization import rebuild

    graph.validate()
    diagnostics, patterns = [], []
    stage = _stage(context)
    enabled = (getattr(context, "fuse_swiglu_residuals", False) and stage != "off"
               and "forward" in getattr(context, "fusion_phases", ("forward", "backward")))
    product = stage != "activation_pack"
    for multiply in tuple(graph.nodes):
        if not isinstance(multiply, PointwiseNode) or multiply.op != "mul" or len(multiply.args) != 2:
            continue
        pairs = [(value, value.producer, multiply.args[1 - index]) for index, value in enumerate(multiply.args)
                 if isinstance(value, TensorValue) and isinstance(value.producer, CallNode)
                 and type(value.producer.specification) is NativeActivation
                 and value.producer.specification.kind == "silu"]
        if len(pairs) != 1 or not isinstance(pairs[0][2], TensorValue):
            continue
        gate, activation, up = pairs[0]
        gate_pack, up_pack = _pack(graph, gate), _pack(graph, up)
        record = candidate(graph, activation, activation.specification, "forward",
                           FORWARD if product else ACTIVATION_PACK, requested=enabled)
        record.update(transformation="kernel_fusion", requested_stage=stage)
        diagnostics.append(record)
        if not enabled:
            continue
        reason = _forward_reason(graph, activation, multiply, gate, up, gate_pack, up_pack, product, context)
        if reason:
            record.update(eligible=False, reason=reason)
            continue
        spec = activation.specification
        runtime = _Run(spec, gate.meta.shape, product)
        outputs = ((multiply.value if product else gate), *activation.values[1:], *gate_pack.values,
                   *(up_pack.values if product else ()))
        replacement = CallNode("swiglu.forward" if product else "swiglu.activation_pack",
            recorded(runtime, runtime.forward, "forward"), (activation.args[0], up) if product else (activation.args[0],),
            {}, outputs, runtime)
        replacement.bf16_policy = activation.bf16_policy
        replacement.saved_tensors_for_bwd = outputs[1:]
        replacement.activation_save_policy = activation.activation_save_policy
        runtime.output_abi = _FORWARD_ABI if product else _ACTIVATION_ABI
        runtime.output_metadata = tuple(value.meta for value in outputs)
        select(replacement, record, roles={"swiglu_training_forward": ["_swiglu_forward"]},
               consumed=(activation.op, gate_pack.op, *((up_pack.op, multiply.op) if product else ())), source_files=_SOURCES)
        _abi(record, runtime.output_abi, outputs, stage)
        record["replacement_contract"].update(saved_roles=("activation.derivative", "mul.rounded_gate", "mul.raw_up")
            if product else ("activation.derivative", "mul.rounded_gate"), output_count=len(outputs),
            rounding="BF16_silu_then_BF16_product", dense_gate_materialized=not product,
            estimated_scratch_bytes=4, preserved_saved_value_names=tuple(value.name for value in outputs[1:]))
        removed = {id(node) for node in (activation, gate_pack, *((up_pack, multiply) if product else ()))}
        anchor = multiply if product else activation
        position = sum(id(node) not in removed for node in graph.nodes[:graph.nodes.index(anchor)])
        graph.nodes = [node for node in graph.nodes if id(node) not in removed]
        graph.nodes.insert(position, replacement)
        if product:
            spec._swiglu_contract = dict(shape=gate.meta.shape,
                derivative=tuple(value.name for value in activation.values[1:]),
                gate=tuple(value.name for value in gate_pack.values), up=tuple(value.name for value in up_pack.values))
        rebuild(graph)
        patterns.append(record["pattern"])
    return diagnostics, tuple(patterns)


def _restore(value, names):
    node = value.producer
    return (node if isinstance(node, CallNode) and node.op == "codec.restore_activation"
            and _codec_function(node.target, "restore", value.meta.shape)
            and tuple(arg.name for arg in node.args) == tuple("saved_" + name for name in names) else None)


def backward(graph, context, diagnostics=None):
    from .optimization import rebuild

    diagnostics = diagnostics if diagnostics is not None else []
    stage = _stage(context)
    enabled = (getattr(context, "fuse_swiglu_residuals", False) and stage == "forward_backward"
               and "backward" in getattr(context, "fusion_phases", ("forward", "backward")))
    patterns = []
    for activation in tuple(graph.nodes):
        if getattr(activation, "op", None) != "activation.coefficient_vjp":
            continue
        spec = getattr(activation.target, "specification", None)
        contract = getattr(spec, "_swiglu_contract", None)
        if not contract:
            continue
        record = candidate(graph, activation, spec, "backward", BACKWARD, requested=enabled)
        record.update(transformation="kernel_fusion", requested_stage=stage)
        diagnostics.append(record)
        if not enabled:
            continue
        function = getattr(activation.target, "numerical_function", None)
        if (getattr(function, "__self__", None) is not spec
                or getattr(function, "__func__", None) is not NativeActivation.dx
                or getattr(activation.target, "__code__", None) is not recorded(spec, spec.dx, "backward").__code__):
            record.update(eligible=False, reason="untrusted_activation_vjp")
            continue
        gate_mul = activation.args[0].producer
        if (not _multiply(gate_mul)
                or gate_mul.value.users != [activation]):
            record.update(eligible=False, reason="activation_adjoint_fanout_or_operation")
            continue
        restored = [(value, _restore(value, contract["up"])) for value in gate_mul.args if isinstance(value, TensorValue)]
        restored = [(value, node) for value, node in restored if node is not None]
        if len(restored) != 1:
            record.update(eligible=False, reason="raw_up_saved_contract")
            continue
        up_value, up_restore = restored[0]
        gradient = next(value for value in gate_mul.args if value is not up_value)
        if not isinstance(gradient, TensorValue):
            record.update(eligible=False, reason="tensor_cotangent_required")
            continue
        others = []
        for node in graph.nodes:
            if _multiply(node) and gradient in node.args:
                for value in node.args:
                    if isinstance(value, TensorValue) and value is not gradient:
                        restored_gate = _restore(value, contract["gate"])
                        if restored_gate is not None:
                            others.append((node, restored_gate))
        if len(others) != 1:
            record.update(eligible=False, reason="rounded_gate_saved_contract")
            continue
        up_mul, gate_restore = others[0]
        saved = (*activation.args[1:], *gate_restore.args, *up_restore.args)
        policy = getattr(context, "kernel_policy", None)
        if (len(saved) != 9 or tuple(value.name for value in activation.args[1:])
                != tuple("saved_" + name for name in contract["derivative"])
                or any(not isinstance(value.producer, InputNode) for value in saved)
                or up_restore.values[0].users != [gate_mul] or gate_restore.values[0].users != [up_mul]
                or not _dense(gradient, contract["shape"], strided=True)
                or any(not _dense(value, contract["shape"]) for value in (gate_mul.value,
                    up_mul.value, activation.outputs()[0]))
                or any(not _leaves(saved[index:index + 3], contract["shape"], gradient.meta.device)
                       for index in (0, 3, 6))
                or any(_effects(node) for node in (activation, gate_mul, up_mul, up_restore, gate_restore))
                or policy is not None and any(policy.route(family, "compiler.backward") != "triton"
                    for family in ("compiler_pointwise", "activation_codec"))
                or policy is not None and policy.route("activation", "activation.backward") != "triton"):
            record.update(eligible=False, reason="backward_saved_layout_fanout_or_backend")
            continue
        removed_nodes = (activation, gate_mul, up_mul, up_restore, gate_restore)
        first = min(graph.nodes.index(node) for node in (gate_mul, up_mul))
        last = max(graph.nodes.index(node) for node in removed_nodes)
        if any(not any(node is current for current in removed_nodes) and not _commutable(node)
               for node in graph.nodes[first:last]):
            record.update(eligible=False, reason="intervening_effect_or_opaque_executor")
            continue
        runtime = _Run(spec, contract["shape"], True)
        replacement = CallNode("swiglu.backward", recorded(runtime, runtime.backward, "backward"),
            (gradient, *saved), {}, (activation.outputs()[0], up_mul.value), runtime)
        replacement.bf16_policy = activation.bf16_policy
        runtime.output_abi = _BACKWARD_ABI
        runtime.output_metadata = tuple(value.meta for value in replacement.values)
        replacement.target.packed_offload_panel = SwiGLUPackedOffloadPanel(runtime)
        select(replacement, record, roles={"swiglu_training_backward": ["_swiglu_backward"]},
               consumed=(up_restore.op, gate_restore.op, gate_mul.op, up_mul.op, activation.op), source_files=_SOURCES)
        _abi(record, runtime.output_abi, replacement.values, stage)
        record["replacement_contract"].update(rounding="decode_operand_BF16_then_product_adjoint_BF16_then_derivative",
            dense_coefficient_bytes=0, dense_product_adjoint_bytes=0, estimated_scratch_bytes=0,
            output_count=2, preserved_saved_value_names=tuple(value.name for value in saved))
        removed = {id(node) for node in (activation, gate_mul, up_mul, up_restore, gate_restore)}
        inputs = []
        for value in saved:
            producer = value.producer
            if graph.nodes.index(producer) > first and not any(producer is node for node in inputs):
                inputs.append(producer)
        late = {id(node) for node in inputs}
        anchor = sum(id(node) not in removed and id(node) not in late for node in graph.nodes[:first])
        graph.nodes = [node for node in graph.nodes if id(node) not in removed and id(node) not in late]
        graph.nodes[anchor:anchor] = (*inputs, replacement)
        rebuild(graph)
        patterns.append(BACKWARD)
    return tuple(patterns)
