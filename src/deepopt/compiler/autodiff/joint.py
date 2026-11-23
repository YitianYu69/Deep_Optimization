"""Differentiate semantic regions before scheduling either executable graph."""

import copy
from dataclasses import dataclass, field, replace
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from ..ir import CallNode, ExternKernelNode, GetAttrNode, InputNode, IRGraph, OutputNode, PointwiseNode, TensorValue
from ..registry import KernelCodegenContext
from ..scheduler import Scheduler, external_values
from .aot_auto_diff import AOTAutoDiff
from . import semantic
from deepopt.opt.backends._shared.gemm_diagnostics import capture_gemm_recording, restore_gemm_recording
from deepopt.opt.backends._shared.convolution import capture_convolution_recording, restore_convolution_recording


@lru_cache(maxsize=1)
def _packed_source_identity():
    """Cold-path code identity for cost profiles; never hash during updates."""
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _packed_execution_identity(specifications, context, device_type, *, device=None):
    """Recipe/geometry/options and hardware, without owning tensor payloads."""
    from dataclasses import fields, is_dataclass, asdict
    from ..offload_plan import _freeze, _digest

    def metadata(value):
        return _freeze(asdict(value) if is_dataclass(value) else value)

    options = {}
    for item in fields(context):
        if item.name in ("cache_dir", "kernel_policy", "packed_activation_offload",
                         "packed_offload_peak_profile", "packed_offload_selection_mode"):
            continue
        options[item.name] = metadata(getattr(context, item.name))
    recipes = []
    for spec in specifications:
        row = {"type": type(spec).__module__ + "." + type(spec).__qualname__}
        for name in ("plan", "options", "geometry", "compression", "compute_dtype", "input_format",
                     "weight_format", "explicit_padding", "synthetic_right", "transposed", "unbatched",
                     "needs", "kind", "approximate"):
            value = getattr(spec, name, None)
            if value is not None:
                if isinstance(value, torch.memory_format):
                    row[name] = str(value)
                else:
                    row[name] = metadata(value)
        recipes.append(row)
    hardware = dict(device_type=device_type, uuid=None, capability=None)
    if device_type == "cuda" and torch.cuda.is_available():
        device = device or next((meta.device for spec in specifications for meta in getattr(spec, "metas", ())
                                if meta is not None and str(meta.device).startswith("cuda")), "cuda:0")
        props = torch.cuda.get_device_properties(device)
        hardware.update(uuid=str(getattr(props, "uuid", "unavailable")), capability=(props.major, props.minor),
                        name=props.name)
    return dict(hardware=hardware, numerical_recipe_sha256=_digest(recipes), compiler_options_sha256=_digest(options))


def _native_profile_identity(region):
    """Cold metadata shared by resident diagnostics and transport plans."""
    from deepopt._bf16_policy import resolve_bf16_policy
    from deepopt.kernel_policy import resolve_kernel_policy

    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device_type": region.device_type, "bf16_policy": resolve_bf16_policy(region.context.bf16_policy).as_dict(),
        "kernel_policy": resolve_kernel_policy(region.kernel_policy).fingerprint,
        "algorithm_controls": {"deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "compiler_source_sha256": _packed_source_identity(),
        **_packed_execution_identity(region.specifications, region.context, region.device_type,
            device=next((v.meta.device for v in region.inputs if v.meta is not None
                         and str(v.meta.device).startswith("cuda")), None))}


def linear_backward_orders(nodes):
    """Read role order from emitted operations, including fused dX nodes."""
    orders = {}
    for node in nodes:
        op = getattr(node, "op", "")
        role = "dx" if op.startswith("bf16.gemm_dx") else "dw" if op == "bf16.gemm_dw" else None
        if role is None:
            continue
        spec = getattr(getattr(node, "target", None), "specification", None)
        if spec is not None:
            orders.setdefault(id(spec), (spec, []))[1].append(role)
    return orders


def map_tree(value, mapping):
    if isinstance(value, TensorValue):
        return mapping[value]
    if isinstance(value, tuple):
        return tuple(map_tree(item, mapping) for item in value)
    if isinstance(value, list):
        return [map_tree(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: map_tree(item, mapping) for key, item in value.items()}
    return value


def _layer_norm(node):
    if isinstance(node, ExternKernelNode):
        from .native_layer_backward_rules import _native_layer_kind

        return _native_layer_kind(node) == "layer_norm"
    return False


def _normalize_layer_norm(node):
    from types import SimpleNamespace
    from .native_layer_backward_rules import _arg, _module, _parameter

    module = _module(node)
    shape = tuple(module.normalized_shape) if module is not None else tuple(_arg(node, 1, "normalized_shape"))
    eps = module.eps if module is not None else _arg(node, 4, "eps", 1e-5)
    facade = SimpleNamespace(normalized_shape=shape, eps=eps, training=getattr(module, "training", True),
        plan={"fp8": False, "bf16_policy": node.bf16_policy.as_dict(),
              "activation_compression": {"enabled": False, "bits": 2, "group_size": 1024}},
        last_execution={})
    if node.bf16_policy.storage_enabled:
        from deepopt.opt._autocast import _get_activation_compression

        device = torch.device(node.args[0].meta.device).type
        facade.plan["activation_compression"].update(_get_activation_compression(device) or {})
    return SimpleNamespace(module=facade, module_path=getattr(node, "module_path", None),
        bf16_policy=node.bf16_policy,
        node_name=node.value.name, value=node.value, output=node.output,
        op="native_layer_norm", args=(node.args[0], _parameter(node, "weight", 2), _parameter(node, "bias", 3)), kwargs={})


def supported(node, registry, unsafe_values):
    from ..native_linear import NativeLinearNode
    from ..native_stateful import NativeStatefulNode
    from ..linear import DeepOptLinearNode
    from ..native_attention import AttentionNode
    from ..native_operations import OperationNode
    from ..autodiff_runtime import is_native_layer_node
    from .primitives import kind
    from ..effects import stochastic_depth_snapshot

    if (getattr(node, "target", None) is stochastic_depth_snapshot
            and getattr(node, "residual_recipe", None)
            and all(not value.meta.requires_grad for value in node.outputs())):
        return True

    values = (*node.inputs(), *node.outputs())
    if any(value in unsafe_values or value.meta is None for value in values):
        return False
    if any(value.meta.dtype.is_complex for value in values):
        return False
    if isinstance(node, (NativeLinearNode, NativeStatefulNode, DeepOptLinearNode, AttentionNode, OperationNode)):
        return True
    if any(value.meta.representation != "dense" and getattr(value.meta, "storage_dtype", None) not in
           (torch.float8_e4m3fn, torch.float8_e5m2) for value in values):
        return False
    if isinstance(node, ExternKernelNode) and node.autograd_owner == "deepopt":
        return False
    if isinstance(node, ExternKernelNode) and node.stateful:
        return False
    if is_native_layer_node(node):
        return True
    if isinstance(node, ExternKernelNode):
        from .native_layer_backward_rules import _native_layer_kind

        if _native_layer_kind(node) == "reshape_view":
            return True
    if kind(node) is not None:
        return True
    extension = registry.for_node(node)
    return registry.has_backward_rule(getattr(node, "op", extension.name if extension else "")) and (
        isinstance(node, (PointwiseNode, CallNode, ExternKernelNode)) or registry.for_node(node) is not None)


def _clone_region(nodes, context, registry):
    from ..native_linear import NativeLinearNode
    from ..native_stateful import NativeStatefulNode
    from ..linear import DeepOptLinearNode
    from ..native_attention import AttentionNode
    from ..native_operations import OperationNode

    inputs = tuple(external_values(nodes))
    graph, mapping, specs = IRGraph(), {}, []
    for index, value in enumerate(inputs):
        local = TensorValue(value.name, value.meta, save_codec=value.save_codec,
                            provenance=value.provenance, source_name=value.source_name,
                            storage_source=value.storage_source, model_state_copy=value.model_state_copy,
                            model_state_origin=value.model_state_origin, model_state_transforms=value.model_state_transforms)
        graph.add(InputNode(index, local))
        mapping[value] = local
    occurrences = {}
    for node in nodes:
        clone = copy.copy(node)
        extension = registry.for_node(node)
        if not getattr(clone, "op", None) and extension is not None:
            clone.op = extension.name
        for name in ("args", "kwargs", "autograd_inputs", "autograd_buffers"):
            if hasattr(node, name):
                setattr(clone, name, map_tree(getattr(node, name), mapping))
        outputs = tuple(TensorValue(value.name, value.meta, save_codec=value.save_codec,
                        provenance=value.provenance, source_name=value.source_name,
                        storage_source=value.storage_source, model_state_copy=value.model_state_copy,
                        model_state_origin=value.model_state_origin, model_state_transforms=value.model_state_transforms) for value in node.outputs())
        mapping.update(zip(node.outputs(), outputs))
        if hasattr(clone, "values"):
            clone.values = outputs
        elif hasattr(clone, "value"):
            clone.value = outputs[0] if outputs else None
        else:
            raise NotImplementedError(f"No native AutoDiff cloning contract for {type(node).__name__}")
        clone.backward_rule, clone.saved_tensors_for_bwd, clone.activation_save_policy = None, (), None
        spec = None
        if isinstance(clone, NativeLinearNode):
            spec = (semantic.BF16Linear if clone.value.meta.dtype == torch.bfloat16 else semantic.DenseLinear)(clone, context)
        elif isinstance(clone, DeepOptLinearNode):
            from .fp8 import FP8Linear
            from math import prod

            if any(prod(value.meta.shape) == 0 for value in clone.args[:2]):
                raise ValueError("FP8 linear currently requires non-empty input and weight")
            else:
                key = id(getattr(node.target, "module", node.target))
                occurrence = occurrences.get(key, 0)
                occurrences[key] = occurrence + 1
                spec = FP8Linear(clone, replace(context, callsite_occurrence=occurrence))
        elif isinstance(clone, AttentionNode):
            from .attention import SDPA

            spec = SDPA(clone, context)
        elif isinstance(clone, OperationNode):
            from .spatial import specification

            spec = specification(clone, context)
        elif isinstance(clone, NativeStatefulNode):
            if "fp8_training" in clone.module.plan:
                from .stateful_fp8 import StatefulFP8

                spec = StatefulFP8(clone, context)
            else:
                spec = (semantic.NativeLayerNorm if clone.op == "native_layer_norm" else semantic.NativeActivation)(clone, context)
        elif _layer_norm(clone):
            clone = _normalize_layer_norm(clone)
            spec = (semantic.NativeLayerNorm if clone.bf16_policy.storage_enabled else semantic.LibraryLayerNorm)(clone, context)
        else:
            from .primitives import kind, lower

            if kind(clone) is not None:
                clone = lower(clone)
        if spec is not None:
            clone = spec.lower(clone)
            specs.append(spec)
        clone.bf16_policy = node.bf16_policy
        clone.activation_memory = getattr(node, "activation_memory", context.activation_memory)
        for name in ("source_node", "module_path"):
            if getattr(clone, name, None) is None and hasattr(node, name):
                setattr(clone, name, getattr(node, name))
        graph.add(clone)
    node_ids = {id(node) for node in nodes}
    outputs = tuple(value for node in nodes for value in node.outputs()
                    if any(id(user) not in node_ids for user in value.users))
    graph.add(OutputNode(tuple(mapping[value] for value in outputs)))
    graph.validate()
    return graph, inputs, outputs, specs


class _JointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, region, *inputs):
        ctx.region = region
        capture_gemm_recording(ctx)
        capture_convolution_recording(ctx)
        ctx.set_materialize_grads(False)
        from deepopt.opt.scaling.delayed import current_scaling_batch
        from ..runtime.boundary_observation import current_native_boundary_observer

        ctx.scaling_batch = current_scaling_batch()
        observer = current_native_boundary_observer()
        ctx.boundary_observer = observer
        outputs, saved = (region.forward_artifact.fn(*inputs) if observer is None else
                          observer.run_artifact(region.forward_artifact, *inputs, phase="forward"))
        if region.context.collect_saved_state_metadata:
            region.record_saved_metadata(saved)
        ctx.save_for_backward(*saved)
        non_differentiable = [value for value, source in zip(outputs, region.outputs)
                              if not source.meta.requires_grad]
        if non_differentiable:
            ctx.mark_non_differentiable(*non_differentiable)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    @restore_gemm_recording
    @restore_convolution_recording
    def backward(ctx, *gradients):
        region = ctx.region
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt._autocast import _suspend_fp8_interception
        from deepopt.opt.scaling.delayed import autograd_scaling_scope
        from ..runtime.boundary_observation import current_native_boundary_observer

        # Only graph outputs receive seeds. Missing upstream gradients are
        # materialized here, without retaining the forward output storage.
        seeds = []
        for index in region.seed_indices:
            gradient = gradients[index]
            metadata = region.outputs[index].meta
            seeds.append(torch.zeros(metadata.shape, device=metadata.device, dtype=metadata.dtype)
                         if gradient is None else gradient)
        with kernel_policy_scope(region.kernel_policy), _suspend_fp8_interception(), \
                torch.autocast(region.device_type, enabled=False), autograd_scaling_scope(ctx.scaling_batch):
            owned_inputs = [*seeds, *ctx.saved_tensors]
            seeds.clear()
            # This engine hook respects the current GraphTask's keep_graph
            # flag. Once released, the consuming executable owns the only
            # transient references to saved activations and codec payloads.
            if hasattr(ctx, "maybe_clear_saved_tensors"):
                ctx.maybe_clear_saved_tensors()
            observer = current_native_boundary_observer()
            if observer is None:
                observer = ctx.boundary_observer
            values = (region.backward_artifact.fn(owned_inputs) if observer is None else
                      observer.run_artifact(region.backward_artifact, owned_inputs, phase="backward"))
        result = [None] * len(region.inputs)
        for target, value in zip(region.trace.grad_targets, values):
            result[int(target.key)] = value
        return (None, *result)


def _packed_observer(observer):
    """Only explicit code/metadata observers may borrow transport execution."""
    if observer is not None and (getattr(observer, "metadata_only", False) is not True
                                 or getattr(observer, "supports_packed_transport", False) is not True):
        raise RuntimeError("Packed activation offload requires a transport-aware metadata-only observer")
    return observer


class _OffloadedJointFunction(torch.autograd.Function):
    """Private bridge for scheduler-managed host saves and lazy CUDA bindings."""

    @staticmethod
    def forward(ctx, region, frame, *inputs):
        ctx.region, ctx.offload_frame = region, frame
        capture_gemm_recording(ctx)
        capture_convolution_recording(ctx)
        ctx.set_materialize_grads(False)
        from deepopt.opt.scaling.delayed import current_scaling_batch
        from ..runtime.boundary_observation import current_native_boundary_observer

        ctx.scaling_batch = current_scaling_batch()
        observer = ctx.boundary_observer = _packed_observer(current_native_boundary_observer())
        outputs, saved = (region.forward_artifact.fn(*inputs, _offload=frame) if observer is None else
                          observer.run_artifact(region.forward_artifact, *inputs, phase="forward", _offload=frame))
        # The transport report is a cold diagnostic snapshot. Rebuilding it
        # on every update would add Python work proportional to all saves.
        if not region.saved_state_metadata:
            region.record_saved_metadata(saved, offload_frame=frame)
        ctx.save_for_backward(*saved)
        frame.finish_forward()
        non_differentiable = [value for value, source in zip(outputs, region.outputs)
                              if not source.meta.requires_grad]
        if non_differentiable:
            ctx.mark_non_differentiable(*non_differentiable)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    @restore_gemm_recording
    @restore_convolution_recording
    def backward(ctx, *gradients):
        region, frame = ctx.region, ctx.offload_frame
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt._autocast import _suspend_fp8_interception
        from deepopt.opt.scaling.delayed import autograd_scaling_scope
        from ..runtime.boundary_observation import current_native_boundary_observer

        observer = _packed_observer(current_native_boundary_observer() or ctx.boundary_observer)
        seeds = []
        for index in region.seed_indices:
            gradient, metadata = gradients[index], region.outputs[index].meta
            seeds.append(torch.zeros(metadata.shape, device=metadata.device, dtype=metadata.dtype)
                         if gradient is None else gradient)
        task, owned_inputs, error = None, [], None
        with frame.backward_scope(), kernel_policy_scope(region.kernel_policy), _suspend_fp8_interception(), \
                torch.autocast(region.device_type, enabled=False), autograd_scaling_scope(ctx.scaling_batch):
            try:
                saved = ctx.saved_tensors
                task = frame.begin_backward(saved, seeds=seeds)
                owned_inputs = [*seeds, *saved]
                seeds.clear()
                del saved
                # Task/scope leases exist before the graph releases tokens.
                if hasattr(ctx, "maybe_clear_saved_tensors"):
                    ctx.maybe_clear_saved_tensors()
                values = (region.backward_artifact.fn(owned_inputs, _offload=task) if observer is None else
                          observer.run_artifact(region.backward_artifact, owned_inputs, phase="backward", _offload=task))
            except BaseException as failure:
                error = failure
                raise
            finally:
                owned_inputs.clear()
                seeds.clear()
                if task is not None:
                    try:
                        task.finish(error=error)
                    except BaseException as cleanup_error:
                        if error is None:
                            raise
                        error.add_note(f"Packed backward cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
        result = [None] * len(region.inputs)
        for target, value in zip(region.trace.grad_targets, values):
            result[int(target.key)] = value
        return (None, None, *result)


@dataclass
class JointProgram:
    graph: IRGraph
    inputs: tuple[TensorValue, ...]
    outputs: tuple[TensorValue, ...]
    specifications: list
    registry: object
    context: KernelCodegenContext
    forward_artifact: object = field(init=False)
    backward_artifact: object = field(init=False)
    saved_state_metadata: dict = field(default_factory=dict, init=False)

    def record_saved_metadata(self, saved, *, offload_frame=None):
        """One bounded diagnostic snapshot; never retain runtime tensors."""
        selectors = {description["name"] for binding in self.trace.saved_value_specs
                     if binding["representation"] == "attention_state"
                     for description in binding["physical_components"] if description["role"] == "implementation"}
        rows = []
        for source, value in zip(self.trace.saved_tensors, saved):
            if value is None:
                if getattr(source.meta, "presence", "required") != "optional":
                    raise ValueError(f"Required saved tensor is absent: {source.name}")
                row = dict(name=source.name, presence="absent", shape=None, dtype=None, device=None, bytes=0)
            else:
                row = dict(name=source.name, presence="present", shape=tuple(value.shape), dtype=str(value.dtype),
                           device=str(value.device), stride=tuple(value.stride()), bytes=value.numel() * value.element_size())
                if source.name in selectors:
                    row["scalar"] = int(value)
                if offload_frame is not None and value.device.type == "cpu" and source.name in self.offload_components:
                    component, bundle_id = self.offload_components[source.name]
                    abi = dict(shape=source.meta.shape, stride=component.stride,
                               dtype=str(source.meta.dtype), device=str(source.meta.device))
                    row["transport"] = dict(version=1, bundle_id=bundle_id, saved_index=component.saved_index,
                        generation=offload_frame.invocation,
                        residency="host_pinned" if value.is_pinned() else "host_pageable",
                        producer_abi=abi, consumer_abi=dict(abi))
            rows.append(row)
        self.saved_state_metadata.update(bindings=tuple(rows), metadata_only=True)

    def __post_init__(self):
        self.kernel_policy = self.context.kernel_policy
        from .activation_saving import activation_device_type

        self.device_type = activation_device_type(self.inputs)
        self.trace = AOTAutoDiff(self.graph, registry=self.registry,
                                saved_activation_policy=self.context.saved_activation_policy).diff()
        # Coverage describes the differentiated source operations, before
        # composite replacements change their forward execution boundaries.
        self.rules = tuple({"operation": node.op, "rule": node.backward_rule,
                            "results": tuple(value.name for value in node.outputs())}
                           for node in self.graph.nodes if node.backward_rule is not None)
        from .norm_activation_residuals import forward as norm_residual_forward, backward as norm_residual_backward

        early_candidates, early_patterns = norm_residual_forward(self.graph, self.context)
        if self.context.fuse_swiglu_residuals:
            from .swiglu_fusion import forward as swiglu_forward

            candidates, patterns = swiglu_forward(self.graph, self.context)
            early_candidates.extend(candidates)
            early_patterns += patterns
        from .pack_fusion import share_packs

        self.trace, self.pack_sharing_decisions = share_packs(
            self.graph, self.trace, enabled=self.context.share_compatible_packs)
        for node in self.trace.graph.nodes:
            if isinstance(node, InputNode) and node.index < len(self.trace.grad_output_source):
                node.value.meta = replace(node.value.meta, stride=None)
                node.value.provenance = "upstream_gradient"
            elif isinstance(node, InputNode):
                # Backward bindings are aliases of forward saved components,
                # including lifted model weights. They are not new callers'
                # allocations simply because they use the InputNode ABI.
                source = self.trace.saved_tensors[node.index - len(self.trace.grad_output_source)]
                node.value.provenance = source.provenance or "saved_state"
                node.value.source_name = source.source_name or source.name
                from ..ir import model_storage_source
                node.value.storage_source = model_storage_source(source)
                node.value.model_state_copy = source.model_state_copy
                node.value.model_state_origin = source.model_state_origin
                node.value.model_state_transforms = source.model_state_transforms
        self.seed_indices = tuple(self.outputs.index(value) for value in self.trace.grad_output_source)
        output_node = self.graph.nodes[-1]
        output_node.result = (output_node.result, self.trace.saved_tensors)
        from ..ir import register_uses

        register_uses(output_node)
        from .optimization import forward, backward as optimize_backward, backward_layouts

        self.fusion_candidates, self.fusion_patterns = forward(self.graph, self.context)
        self.fusion_candidates = early_candidates + self.fusion_candidates
        self.fusion_patterns = early_patterns + self.fusion_patterns
        from .normalization_fusion import forward as norm_forward, backward as norm_backward

        norm_candidates, norm_patterns = norm_forward(self.graph, self.context)
        self.fusion_candidates.extend(norm_candidates)
        self.fusion_patterns += norm_patterns
        from .norm_cat import forward as norm_cat_forward

        cat_candidates, cat_patterns = norm_cat_forward(self.graph, self.context)
        self.fusion_candidates.extend(cat_candidates)
        self.fusion_patterns += cat_patterns
        from .convolution_order import backward as convolution_order_backward
        from .convolution_geometry import backward as convolution_geometry_backward
        from .convolution_reuse import backward as convolution_reuse_backward

        self.fusion_patterns += convolution_reuse_backward(self.trace.graph, self.context,
                                                           diagnostics=self.fusion_candidates)
        self.fusion_patterns += convolution_order_backward(self.trace.graph, self.context,
                                                           diagnostics=self.fusion_candidates)
        self.fusion_patterns += convolution_geometry_backward(self.trace.graph, self.context,
                                                              diagnostics=self.fusion_candidates)
        self.fusion_patterns += norm_residual_backward(self.trace.graph, self.context, self.fusion_candidates)
        if self.context.fuse_swiglu_residuals:
            from .swiglu_fusion import backward as swiglu_backward

            self.fusion_patterns += swiglu_backward(self.trace.graph, self.context, self.fusion_candidates)
        selected_orders = linear_backward_orders(self.trace.graph.nodes)
        self.fusion_patterns += optimize_backward(self.trace.graph, self.context, self.fusion_candidates)
        self.fusion_patterns += norm_backward(self.trace.graph, self.context, self.fusion_candidates)
        from .packed_multiply import backward as packed_multiply_backward

        self.fusion_patterns += packed_multiply_backward(self.trace.graph, self.context, self.fusion_candidates)
        from .donation import backward as donate_backward

        self.fusion_patterns += donate_backward(self.trace.graph, self.context,
                                               diagnostics=self.fusion_candidates)
        rewritten_orders = linear_backward_orders(self.trace.graph.nodes)
        backward_layouts(self.trace.graph)
        options = {name: getattr(self.context, name) for name in (
            "bf16_gemm_backend", "bf16_decode", "bf16_tile", "bf16_num_stages", "bf16_backward_order")}
        schedules = []
        for graph, backward in ((self.graph, False), (self.trace.graph, True)):
            schedule = Scheduler(
                graph, registry=self.registry, kernel_policy=self.kernel_policy, backward=backward,
                max_pointwise_fusion_length=self.context.max_pointwise_fusion_length).run()
            schedules.append(schedule)
        self._schedules = tuple(schedules)
        self._wrapper_options = options
        self._offload_candidates = None
        self._offload_finalized = False
        self._has_executed = False
        self.offload_plan = self.offload_runtime = None
        self.offload_components = {}
        from deepopt._packed_offload import resolve_packed_offload

        packed_policy = resolve_packed_offload(self.context.packed_activation_offload)
        # Metadata generation happens once at compilation. It adds no runtime
        # hooks to the resident path and lets an offline converter prove that
        # a resident diagnostic matches a later transport scope.
        from ..offload_plan import _digest, _freeze, _schedule_descriptor

        self.native_profile_unavailable_reason = None
        try:
            self.native_profile_identity = _freeze(_native_profile_identity(self))
            self.native_forward_signature, self.native_backward_signature = (
                _digest(_schedule_descriptor(schedule)) for schedule in schedules)
        except ValueError as error:
            if packed_policy.enabled:
                raise  # An enabled transport descriptor still requires exact identities.
            # Custom resident policies may implement a numerical protocol
            # without a canonical metadata schema. Diagnostic provenance must
            # not narrow their existing execution contract or retain owners.
            self.native_profile_identity = None
            self.native_forward_signature = self.native_backward_signature = None
            self.native_profile_unavailable_reason = str(error)[:1024]
        if packed_policy.enabled:
            if self.device_type != "cuda":
                raise ValueError("Packed activation offload requires native CUDA execution")
            from ..offload_plan import build_packed_offload_plan
            from ..runtime.manager import CompiledGraphRuntimeManager

            self.offload_plan = build_packed_offload_plan(self.graph, self.trace.graph, *schedules,
                self.trace.saved_value_specs, policy=packed_policy,
                saved_tensors=self.trace.saved_tensors, backward_argument_sources=self.trace.backward_argument_sources,
                defer_host_budget=packed_policy.physical_host_budget,
                profile_identity=self.native_profile_identity)
            self.offload_plan.validate_current(*schedules)
            self.offload_components = {component.name: (component, bundle.bundle_id)
                for bundle in self.offload_plan.bundles for component in bundle.components}
            if packed_policy.physical_host_budget:
                # Candidate discovery cannot spend the entire host budget
                # independently in each region. The outer wrapper freezes one
                # selection before generating transport-aware region wrappers.
                self._offload_candidates = self.offload_plan
                self.offload_plan = None
                self.offload_components = {}
            elif self.offload_plan.bundles:
                self.offload_runtime = CompiledGraphRuntimeManager(packed_policy)
        if self._offload_candidates is None:
            self._compile_wrappers()
        else:
            self.forward_artifact = self.backward_artifact = None
        from .fusion_contracts import ordered_operations

        self.execution_order = {"forward": ordered_operations(schedules[0]),
                                "backward": ordered_operations(schedules[1])}
        for candidate in self.fusion_candidates:
            if candidate.get("selected"):
                candidate["actual_order"] = self.execution_order[candidate.get("execution_phase", "forward")]
        final_orders = linear_backward_orders(schedules[1].nodes())
        self.backward_orders = tuple({
            "module_path": spec.module_path, "node_name": spec.node_name,
            "requested_order": spec.backward_order,
            "selected_order": tuple(selected),
            "post_fusion_order": tuple(rewritten_orders.get(key, (None, []))[1]),
            "final_order": tuple(final_orders.get(key, (None, []))[1]),
            "reason": ("fusion_dependency_or_effect_limited" if selected != final_orders.get(key, (None, []))[1]
                       else "fusion_order_restored" if selected != rewritten_orders.get(key, (None, []))[1]
                       else "final_source_order"),
        } for key, (spec, selected) in selected_orders.items())

    def _compile_wrappers(self):
        from ..codegen.wrapper import WrapperCodegen

        artifacts = []
        for schedule, backward in zip(self._schedules, (False, True)):
            artifacts.append(WrapperCodegen(torch.nn.Module(), schedule,
                registry=self.registry, cache_dir=self.context.cache_dir, block_size=self.context.block_size,
                enable_autodiff=False, kernel_policy=self.kernel_policy, bf16_options=self._wrapper_options,
                diagnostic_phase="backward" if backward else "forward", consume_inputs=backward,
                gradient_input_count=len(self.trace.grad_output_source) if backward else 0,
                packed_offload_plan=self.offload_plan).run())
        self.forward_artifact, self.backward_artifact = artifacts
        self.set_native_scope_metadata(0, None)

    def set_native_scope_metadata(self, index, identity):
        """Scalar diagnostic provenance, with no invocation or storage owner."""
        from ..offload_plan import _digest

        for artifact in (self.forward_artifact, self.backward_artifact):
            artifact.native_region_index = index
            artifact.native_scope_identity = identity
            artifact.native_forward_signature = self.native_forward_signature
            artifact.native_backward_signature = self.native_backward_signature
            artifact.native_recipe_identity_sha256 = (
                _digest(self.native_profile_identity) if self.native_profile_identity is not None else None)
            artifact.native_profile_unavailable_reason = self.native_profile_unavailable_reason

    def finalize_offload(self, plan, manager, *, model_plan):
        if self._offload_finalized or self._has_executed:
            raise RuntimeError("Cannot change a native offload plan after its first execution")
        plan.validate_current(*self._schedules)
        self.offload_plan = plan
        self.offload_runtime = manager if plan.bundles else None
        self.model_offload_plan = model_plan
        self.offload_components = {component.name: (component, bundle.bundle_id)
            for bundle in plan.bundles for component in bundle.components}
        self._compile_wrappers()
        self._offload_finalized = True

    def __call__(self, *inputs):
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt._autocast import _suspend_fp8_interception
        if self.forward_artifact is None:
            # Direct native-region calls still receive a complete, immutable
            # local scope; applications never manage transfer lifecycles.
            from ..model_offload_plan import finalize_model_offload_regions

            finalize_model_offload_regions((self,))
        self._has_executed = True
        with kernel_policy_scope(self.kernel_policy), _suspend_fp8_interception():
            if self.offload_runtime is None:
                outputs = _JointFunction.apply(self, *inputs)
            else:
                from ..runtime.boundary_observation import current_native_boundary_observer
                from ..runtime.manager import current_runtime_manager

                _packed_observer(current_native_boundary_observer())
                manager = current_runtime_manager(self.context.packed_activation_offload) or self.offload_runtime
                frame = manager.begin_forward(self.offload_plan)
                try:
                    with frame.saved_token_scope():
                        outputs = _OffloadedJointFunction.apply(self, frame, *inputs)
                except BaseException as error:
                    try:
                        frame.abort(error=error)
                    except BaseException as cleanup_error:
                        error.add_note(f"Packed forward cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
                    raise
        return outputs[0] if len(self.outputs) == 1 else outputs


@dataclass(eq=False)
class JointRegionNode(CallNode):
    """One bridge over a graph differentiated by native AOTAutoDiff."""


def partition(graph, registry, config, kernel_policy, unsafe_values):
    """Build maximal ordered semantic regions; leave explicit external boundaries."""
    semantic.register_rules(registry)
    from .fp8 import register_rules

    register_rules(registry)
    from .stateful_fp8 import register_rules

    register_rules(registry)
    from .convolution import register_rules

    register_rules(registry)
    from ..native_mha import register_rules

    register_rules(registry)
    from .spatial import register_rules

    register_rules(registry)
    from .attention import register_rules

    register_rules(registry)
    from .primitives import register_rules

    register_rules(registry)
    from .activation_saving import activation_device_type, capture_saved_activation_policy

    device_type = activation_device_type(value for node in graph.nodes for value in node.outputs())
    from deepopt._bf16_policy import current_bf16_policy

    policies = {node.bf16_policy for node in graph.nodes
                if not isinstance(node, (InputNode, GetAttrNode, OutputNode))}
    captured_policy = next(iter(policies)) if len(policies) == 1 else current_bf16_policy(device_type)
    from deepopt._activation_memory import current_activation_memory, activation_memory_compiler_options

    memory = config.activation_memory if not config.activation_memory.is_default else current_activation_memory(device_type)
    memory_options = activation_memory_compiler_options(memory)
    context = KernelCodegenContext(backend=kernel_policy.backend, kernel_policy=kernel_policy,
        bf16_policy=captured_policy,
        activation_memory=memory,
        packed_activation_offload=config.packed_activation_offload,
        packed_offload_selection_mode=config.packed_offload_selection_mode,
        packed_offload_peak_profile=config.packed_offload_peak_profile,
        saved_activation_policy=capture_saved_activation_policy(device_type),
        cache_dir=str(config.cache_dir) if config.cache_dir else None, block_size=config.block_size,
        fuse_deepopt_activations=config.fuse_deepopt_activations,
        fusion_phases=config.fusion_phases,
        share_compatible_packs=memory_options.get("share_compatible_packs", config.share_compatible_packs),
        fuse_packed_multiply=memory_options.get("fuse_packed_multiply", config.fuse_packed_multiply),
        qualify_native_group_fusion=config.qualify_native_group_fusion,
        fuse_norm_cat_body=config.fuse_norm_cat_body,
        donate_fresh_backward_add=config.donate_fresh_backward_add,
        native_conv_backward_order=config.native_conv_backward_order,
        cuda_conv_dx_geometry=config.cuda_conv_dx_geometry,
        native_conv_workspace_reuse=config.native_conv_workspace_reuse,
        native_conv_dx_batch_size=config.native_conv_dx_batch_size,
        fuse_norm_activation_residuals=config.fuse_norm_activation_residuals,
        norm_activation_fusion_stage=config.norm_activation_fusion_stage,
        fuse_swiglu_residuals=config.fuse_swiglu_residuals,
        swiglu_fusion_stage=config.swiglu_fusion_stage,
        max_pointwise_fusion_length=config.max_pointwise_fusion_length,
        collect_saved_state_metadata=config.collect_saved_state_metadata,
        **{name: getattr(config, name) for name in ("bf16_gemm_backend", "bf16_decode", "bf16_tile",
                                                   "bf16_num_stages", "bf16_backward_order")})
    groups, current = [], []
    for node in graph.nodes:
        if isinstance(node, (InputNode, GetAttrNode)):
            continue
        # Running-statistic writes form ordered regions of their own. A save
        # in a preceding region must observe the pre-write buffer/version.
        module = getattr(node, "module", None)
        statistic_write = (isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                           and module.training and module.track_running_stats)
        if statistic_write and current:
            groups.append(current)
            current = []
        if not isinstance(node, OutputNode) and supported(node, registry, unsafe_values):
            current.append(node)
            if statistic_write:
                groups.append(current)
                current = []
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    replacements, removed, programs = {}, set(), []
    for group in groups:
        if not any(value.meta.requires_grad for node in group for value in node.outputs()):
            continue
        local, inputs, outputs, specs = _clone_region(group, context, registry)
        if not outputs:
            continue
        program = JointProgram(local, inputs, tuple(local.nodes[-1].result), specs, registry, context)
        # Preserve original graph values at the execution boundary. A separate
        # graph owns all cloned semantic values and saved-state exports.
        region = JointRegionNode("native_autodiff_region", program, inputs, {}, outputs, program)
        replacements[id(group[0])] = region
        removed.update(id(node) for node in group)
        programs.append(program)
    result = IRGraph()
    emitted = set()
    for node in graph.nodes:
        item = replacements.get(id(node))
        if item is not None:
            # Parameters first encountered later inside the pure region become
            # explicit bridge operands. Resolve their GetAttr bindings before
            # the bridge, without moving them across an external effect.
            for value in item.inputs():
                producer = value.producer
                if isinstance(producer, GetAttrNode) and id(producer) not in emitted:
                    result.nodes.append(producer)
                    emitted.add(id(producer))
            for output in item.outputs():
                output.producer = None
            result.add(item)
        elif id(node) not in removed and id(node) not in emitted:
            result.nodes.append(node)
        emitted.add(id(node))
    # Old users belonging to removed semantic nodes must not extend execution
    # lifetimes or appear in scheduler legality checks.
    for node in result.nodes:
        for value in node.outputs():
            value.users = []
    from ..ir import register_uses

    for node in result.nodes:
        register_uses(node)
    result.validate()
    return result, programs
