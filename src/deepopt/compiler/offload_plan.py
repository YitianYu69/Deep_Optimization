"""Cold-path packed-save inventory over the final native schedules.

Descriptors deliberately contain no IR objects, tensors, pointers or events.
Logical bytes here are admission sizes, not an observed physical GPU peak.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from math import prod

from deepopt._packed_offload import PackedActivationOffloadPolicy, resolve_packed_offload
from .ir import CallNode, GetAttrNode, InputNode, OutputNode, find_tensor_values, model_storage_source
from .scheduler import scheduled_inputs, scheduled_nodes, scheduled_outputs
from .runtime.offload_cost import PrefetchRequest, plan_prefetches


def _freeze(value):
    if isinstance(value, str):
        return str(value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("Offload metadata keys must be strings")
        return tuple(sorted((key, _freeze(child)) for key, child in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(child) for child in value)
    # Metadata's torch dtype/device have no invocation ownership.
    import torch

    if isinstance(value, (torch.dtype, torch.device, torch.memory_format)):
        return str(value)
    raise ValueError(f"Offload descriptors accept metadata only, got {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _schedule_descriptor(schedule):
    schedule.validate()
    return tuple(tuple((type(node).__name__, getattr(node, "op", None),
                        tuple(value.name for value in node.inputs()),
                        tuple((value.name, _freeze(asdict(value.meta)) if value.meta is not None else None)
                              for value in node.outputs()), _freeze(node.scheduling_constraints))
                       for node in scheduled_nodes(item)) for item in schedule.items)


@dataclass(frozen=True)
class PackedOffloadComponent:
    name: str
    saved_index: int
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    device: str
    nbytes: int
    role: str = "value"

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or type(self.saved_index) is not int or self.saved_index < 0
                or type(self.shape) is not tuple or any(type(n) is not int or n < 0 for n in self.shape)
                or type(self.stride) is not tuple or len(self.stride) != len(self.shape)
                or any(type(n) is not int or n < 0 for n in self.stride)
                or type(self.dtype) is not str or type(self.device) is not str
                or type(self.nbytes) is not int or self.nbytes < 0 or type(self.role) is not str):
            raise ValueError("Invalid immutable packed offload component")

    @property
    def bytes(self):
        return self.nbytes


@dataclass(frozen=True)
class PackedOffloadBundle:
    bundle_id: str
    components: tuple[PackedOffloadComponent, ...]
    producer_boundary: int
    last_forward_use: int
    consumers: tuple[str, ...] = ()
    first_backward_use: int = -1
    last_backward_use: int = -1

    def __post_init__(self):
        if (type(self.bundle_id) is not str or not self.bundle_id or type(self.components) is not tuple
                or len(self.components) != 3 or any(type(item) is not PackedOffloadComponent for item in self.components)
                or type(self.consumers) is not tuple or any(type(item) is not str for item in self.consumers)
                or type(self.producer_boundary) is not int or self.producer_boundary < 0
                or type(self.last_forward_use) is not int or self.last_forward_use < self.producer_boundary
                or type(self.first_backward_use) is not int or self.first_backward_use < -1
                or type(self.last_backward_use) is not int or self.last_backward_use < self.first_backward_use):
            raise ValueError("Invalid immutable packed offload bundle")

    @property
    def bytes(self):
        return sum(component.nbytes for component in self.components)

    @property
    def consumer_ids(self):
        return self.consumers


@dataclass(frozen=True)
class PackedOffloadPanel:
    kind: str
    shape: tuple[int, ...]
    component_names: tuple[str, ...]
    groups_per_panel: int
    buffer_count: int
    staging_bytes: int
    group_size: int = 1024
    bits: int = 2

    def __post_init__(self):
        if (type(self.kind) is not str or self.kind != "swiglu_backward_v1" or type(self.shape) is not tuple or len(self.shape) != 4
                or any(type(n) is not int or n < 1 for n in self.shape)
                or type(self.component_names) is not tuple or len(self.component_names) != 9
                or any(type(name) is not str for name in self.component_names)
                or type(self.groups_per_panel) is not int or self.groups_per_panel < 1
                or type(self.buffer_count) is not int or self.buffer_count not in (1, 2)
                or type(self.group_size) is not int or self.group_size != 1024 or type(self.bits) is not int or self.bits != 2
                or type(self.staging_bytes) is not int
                or self.staging_bytes != self.groups_per_panel * self.buffer_count * 3 * 264):
            raise ValueError("Invalid named packed-panel consumer contract")


@dataclass(frozen=True)
class PackedOffloadConsumer:
    consumer_id: str
    schedule_index: int
    bundle_ids: tuple[str, ...]
    bytes: int
    panel: PackedOffloadPanel | None = None

    def __post_init__(self):
        if (type(self.consumer_id) is not str or not self.consumer_id or type(self.schedule_index) is not int
                or self.schedule_index < 0 or type(self.bundle_ids) is not tuple
                or any(type(item) is not str for item in self.bundle_ids)
                or type(self.bytes) is not int or self.bytes < 0):
            raise ValueError("Invalid immutable packed offload consumer")
        if self.panel is not None and type(self.panel) is not PackedOffloadPanel:
            raise ValueError("Only a proven named consumer may use packed panels")

    @property
    def boundary(self):
        return self.schedule_index

    @property
    def staging_bytes(self):
        return self.panel.staging_bytes if self.panel is not None else self.bytes


@dataclass(frozen=True)
class PackedOffloadRejection:
    components: tuple[str, ...]
    reason: str

    def __post_init__(self):
        if type(self.components) is not tuple or any(type(name) is not str for name in self.components) or type(self.reason) is not str:
            raise ValueError("Offload rejection evidence must be immutable metadata")


@dataclass(frozen=True)
class PackedOffloadPlan:
    policy: PackedActivationOffloadPolicy
    bundles: tuple[PackedOffloadBundle, ...] = ()
    consumers: tuple[PackedOffloadConsumer, ...] = ()
    signature: str = ""
    forward_signature: str = ""
    backward_signature: str = ""
    rejections: tuple[PackedOffloadRejection, ...] = ()
    profile_identity: tuple = ()
    progress_boundaries: tuple[int, ...] = ()
    prefetches: tuple[PrefetchRequest, ...] = ()
    model_scope_signature: str = ""
    selection_mode: str = "peak"
    _forward_token: str = field(default="", repr=False, compare=False)
    _backward_token: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.policy, PackedActivationOffloadPolicy):
            raise ValueError("Offload plan requires a resolved immutable policy")
        if self.selection_mode not in ("peak", "all_eligible", "regional_order"):
            raise ValueError("Invalid immutable offload selection mode")
        if any(type(getattr(self, name)) is not str for name in (
                "signature", "forward_signature", "backward_signature", "_forward_token", "_backward_token")):
            raise ValueError("Offload plan signatures must be metadata strings")
        for name, expected in (("bundles", PackedOffloadBundle), ("consumers", PackedOffloadConsumer),
                               ("rejections", PackedOffloadRejection)):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not expected for item in value):
                raise ValueError(f"Offload plan {name} must be immutable descriptors")
        object.__setattr__(self, "profile_identity", _freeze(self.profile_identity))
        if type(self.prefetches) is not tuple or any(type(item) is not PrefetchRequest for item in self.prefetches):
            raise ValueError("Offload prefetches must be immutable scheduling hints")
        if type(self.progress_boundaries) is not tuple or any(type(index) is not int or index < 0 for index in self.progress_boundaries):
            raise ValueError("Offload progress boundaries must be immutable schedule indices")

    @property
    def selected_bytes(self):
        return sum(bundle.bytes for bundle in self.bundles)

    @property
    def required_staging_bytes(self):
        return max((consumer.staging_bytes for consumer in self.consumers), default=0)

    @property
    def required_d2h_inflight_bytes(self):
        return max((bundle.bytes for bundle in self.bundles), default=0)

    @property
    def selected_component_names(self):
        return tuple(component.name for bundle in self.bundles for component in bundle.components)

    @property
    def boundary_ids(self):
        return tuple(sorted(set((*self.progress_boundaries, *(consumer.schedule_index for consumer in self.consumers),
                                 *(request.submit_boundary for request in self.prefetches)))))

    @property
    def forward_publish_at(self):
        return tuple((bundle.producer_boundary, bundle.bundle_id) for bundle in self.bundles)

    @property
    def forward_release_at(self):
        return tuple((bundle.last_forward_use, bundle.bundle_id) for bundle in self.bundles)

    def as_dict(self):
        return {"schema_version": 1, "policy": self.policy.as_dict(), "signature": self.signature,
                "forward_signature": self.forward_signature, "backward_signature": self.backward_signature,
                "selected_bytes": self.selected_bytes, "selected_component_names": self.selected_component_names,
                "bundles": tuple({**asdict(bundle), "bytes": bundle.bytes} for bundle in self.bundles),
                "consumers": tuple(asdict(consumer) for consumer in self.consumers),
                "rejections": tuple(asdict(row) for row in self.rejections),
                "profile_identity": self.profile_identity, "boundary_ids": self.boundary_ids,
                "prefetches": tuple(asdict(request) for request in self.prefetches),
                "physical_storage_observed": False,
                **({"model_scope_signature": self.model_scope_signature,
                    "selection_mode": self.selection_mode} if self.model_scope_signature else {})}

    def with_prefetches(self, requests):
        requests = tuple(requests)
        bundles = {bundle.bundle_id for bundle in self.bundles}
        consumers = {consumer.consumer_id: consumer for consumer in self.consumers}
        if any(type(request) is not PrefetchRequest or request.bundle_id not in bundles
               or request.consumer_id not in consumers
               or request.bundle_id not in consumers[request.consumer_id].bundle_ids
               or request.consumer_boundary != consumers[request.consumer_id].schedule_index for request in requests):
            raise ValueError("Prefetch hints contradict the final packed consumer plan")
        plan = replace(self, prefetches=requests, signature="")
        return replace(plan, signature=_digest(plan.as_dict()))

    def validate_current(self, forward_schedule, backward_schedule, *, profile_identity=None):
        if (self.forward_signature != _digest(_schedule_descriptor(forward_schedule))
                or self.backward_signature != _digest(_schedule_descriptor(backward_schedule))
                or self._forward_token != _digest(forward_schedule.signature())
                or self._backward_token != _digest(backward_schedule.signature())):
            raise ValueError("Offload plan is stale after a final schedule or metadata rewrite")
        if profile_identity is not None and _freeze(profile_identity) != self.profile_identity:
            raise ValueError("Offload profile GPU/software/recipe identity changed")


def _contiguous(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.insert(0, stride)
        stride *= max(size, 1)
    return tuple(result)


def _unwrap_native(function):
    """Only repository-owned profiling wrappers preserve executor authority."""
    from types import CodeType
    from .autodiff.fusion_contracts import select
    from .autodiff.pack_fusion import _profile_pack
    from .autodiff.primitives import emit as primitive_emit

    bodies = {item for owner in (select, _profile_pack) for item in owner.__code__.co_consts
              if isinstance(item, CodeType) and item.co_name == "execute"}
    typed = {item for item in primitive_emit.__code__.co_consts
             if isinstance(item, CodeType) and item.co_name == "typed_contribution"}
    for _ in range(4):
        if getattr(function, "__code__", None) in typed:
            closure = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
            function = closure["function"].cell_contents if "function" in closure else None
            if function is None:
                return None
            continue
        if getattr(function, "__code__", None) not in bodies:
            return function
        function = getattr(function, "__wrapped__", None)
        if function is None:
            return None
    return None


def _recorded_method(function, spec, methods):
    from .autodiff.semantic import recorded

    function = _unwrap_native(function)
    numerical = getattr(function, "numerical_function", None)
    return (getattr(function, "specification", None) is spec and getattr(numerical, "__self__", None) is spec
            and getattr(numerical, "__func__", None) in methods
            and getattr(function, "__code__", None) is recorded(spec, numerical, "backward").__code__)


def _unsafe_annotations(node, *, private_workspace=False):
    for owner in (node, getattr(node, "target", None)):
        if any(getattr(owner, name, False) for name in ("stateful", "side_effects", "mutates", "mutation", "alias_writes")):
            return True
        if any(getattr(owner, name, None) is not None for name in ("stream", "stream_id")):
            return True
    for name in ("stateful", "side_effects", "mutation", "mutates", "alias_writes", "owned_output_alias", "output_alias", "stream", "stream_id"):
        if node.scheduling_constraints.get(name):
            return True
    return node.scheduling_constraints.get("effects") not in (
        None, False, "pure", *(('private_decoded_workspace_overwrite',) if private_workspace else ()))


def _private_constructor(value):
    node = value.producer
    if not isinstance(node, CallNode):
        return False
    operations = {"codec.save_activation", "codec.save_convolution_source", "autodiff_spatial",
                  "autodiff_native_layer_norm", "autodiff_native_activation", "autodiff_bf16_linear",
                  "autodiff_dense_linear", "autodiff_storage_exp", "normalization.cat_body_forward",
                  "normalization.silu_residual_forward", "swiglu.forward", "swiglu.activation_pack"}
    if node.op not in operations or _unsafe_annotations(node):
        return False
    if node.op == "codec.save_activation":
        from types import CodeType
        from .autodiff.aot_auto_diff import BackwardTraceContext

        bodies = tuple(item for item in BackwardTraceContext._store_codec.__code__.co_consts
                       if isinstance(item, CodeType) and item.co_name == "pack")
        return isinstance(node.specification, Mapping) and getattr(_unwrap_native(node.target), "__code__", None) in bodies
    if node.op == "codec.save_convolution_source":
        from .autodiff.spatial import Convolution

        target = _unwrap_native(node.target)
        return (isinstance(node.specification, Mapping) and type(getattr(target, "specification", None)) is Convolution
                and _recorded_method(target, target.specification, (Convolution.pack_source,)))
    if node.op == "autodiff_storage_exp":
        from deepopt.opt.nn.primitive_state import StorageExpState

        return (type(node.specification) is StorageExpState and node.specification.packed
                and getattr(node.target, "__self__", None) is node.specification
                and getattr(node.target, "__func__", None) is StorageExpState.forward)
    spec = getattr(node, "specification", None)
    from .autodiff.semantic import BF16Linear, DenseLinear, NativeActivation, NativeLayerNorm, LibraryLayerNorm
    from .autodiff.spatial import Normalization
    from .autodiff.norm_cat import _NormCatBody
    from .autodiff.normalization_fusion import _NormActivation
    from .autodiff.norm_activation_residuals import NormActivationResidual
    from .autodiff.swiglu_fusion import _Run

    return (type(spec) in (BF16Linear, DenseLinear, NativeActivation, NativeLayerNorm, LibraryLayerNorm,
                           Normalization, _NormCatBody, _NormActivation, NormActivationResidual, _Run)
            and _recorded_method(node.target, spec, (type(spec).forward,)))


def _private_backward_consumer(item):
    """Selected packed leaves cannot escape through views/opaque extensions."""
    from functools import partial
    from types import CodeType
    from .autodiff.aot_auto_diff import BackwardTraceContext
    from .autodiff.semantic import BF16Linear, DenseLinear, NativeActivation, NativeLayerNorm, LibraryLayerNorm
    from .autodiff.spatial import Convolution, Normalization
    from .autodiff.normalization_fusion import _NormActivation
    from .autodiff.norm_activation_residuals import NormActivationResidual
    from .autodiff.swiglu_fusion import _Run
    from .autodiff.packed_multiply import _multiply

    nodes = scheduled_nodes(item)
    if len(nodes) != 1:
        return False
    node = nodes[0]
    private_workspace = False
    if getattr(node, "op", None) == "convolution.native_private_workspace_tiled_dx":
        from .autodiff.convolution_reuse import is_private_workspace_consumer

        private_workspace = is_private_workspace_consumer(node)
        if not private_workspace:
            return False
    if _unsafe_annotations(node, private_workspace=private_workspace):
        return False
    target = _unwrap_native(getattr(node, "target", None))
    if getattr(node, "op", None) == "codec.restore_activation":
        bodies = tuple(body for body in BackwardTraceContext._store_codec.__code__.co_consts
                       if isinstance(body, CodeType) and body.co_name == "restore")
        return getattr(target, "__code__", None) in bodies
    if getattr(node, "op", None) in ("codec.packed_multiply", "codec.packed_multiply_reduce"):
        return isinstance(target, partial) and target.func is _multiply
    if getattr(node, "op", None) == "exp.exact_input_vjp":
        from deepopt.opt.nn.primitive_state import StorageExpState

        return (type(getattr(target, "__self__", None)) is StorageExpState and target.__self__.packed
                and getattr(target, "__func__", None) is StorageExpState.backward)
    spec = getattr(target, "specification", None)
    methods = {
        Convolution: (Convolution.dw, *( (Convolution.reuse_gradients,) if private_workspace else ())),
        Normalization: (Normalization.gradients,),
        NativeActivation: (NativeActivation.dx,), NativeLayerNorm: (NativeLayerNorm.gradients,),
        LibraryLayerNorm: (LibraryLayerNorm.gradients,), BF16Linear: (BF16Linear.dw,), DenseLinear: (DenseLinear.dw,),
        _NormActivation: (_NormActivation.backward,), NormActivationResidual: (NormActivationResidual.backward,),
        _Run: (_Run.backward,),
    }.get(type(spec), ())
    return bool(methods) and _recorded_method(target, spec, methods)


def _panel_contract(item, backward_saved, saved_tensors, policy):
    """No generic extension/name-based panel permission."""
    if not policy.device_staging_bytes:
        return None
    from .autodiff.swiglu_fusion import _Run, SwiGLUPackedOffloadPanel

    if type(item) is not CallNode or item.op != "swiglu.backward" or item.kwargs or len(item.args) != 10:
        return None
    adapter = getattr(item.target, "packed_offload_panel", None)
    spec = item.specification
    if (type(spec) is not _Run or type(adapter) is not SwiGLUPackedOffloadPanel or not spec.product
            or not _recorded_method(item.target, spec, (_Run.backward,))
            or len(item.values) != 2 or tuple(adapter.shape) != tuple(spec.shape)
            or any(value not in backward_saved for value in item.args[1:])):
        return None
    names = tuple(saved_tensors[backward_saved[value]].name for value in item.args[1:])
    groups = (prod(adapter.shape) + 1023) // 1024
    if groups * 3 * 264 <= policy.device_staging_bytes:
        return None  # Preserve the lower-overhead ordinary path when it fits.
    buffers = 2 if policy.device_staging_bytes >= 2 * 3 * 264 else 1
    window = min(groups, policy.device_staging_bytes // (buffers * 3 * 264),
                 max(1, policy.copy_chunk_bytes // (3 * 264)))
    if policy.host_pageable_bytes:
        # The pageable path packs one complete nine-leaf panel into pinned
        # staging. Its forward D2H triplet-fit check alone is insufficient.
        window = min(window, policy.host_pinned_bytes // (3 * 264))
        if policy.physical_host_budget:
            from types import SimpleNamespace
            from .runtime.pinned_arena import plan_pinned_placement

            low, high, fits = 1, window, 0
            while low <= high:
                middle = (low + high) // 2
                descriptions = tuple(SimpleNamespace(shape=(middle, 64) if index % 3 == 0 else (middle,),
                    dtype="torch.int32" if index % 3 == 0 else "torch.float32") for index in range(9))
                if plan_pinned_placement(descriptions, budget_bytes=policy.host_pinned_bytes) is None:
                    high = middle - 1
                else:
                    fits, low = middle, middle + 1
            window = fits
    if window < 1:
        return None
    return PackedOffloadPanel("swiglu_backward_v1", tuple(adapter.shape), names, window, buffers,
                              window * buffers * 3 * 264)


def build_packed_offload_plan(forward_graph, backward_graph, forward_schedule, backward_schedule,
                              residual_recipes=(), *, policy=None, saved_tensors=(),
                              backward_argument_sources=(), profile_identity=None, defer_host_budget=False):
    """Select immutable affine triplets, after all native rewrites/scheduling.

    Saved indices are relative to ``ctx.save_for_backward`` (not seed inputs).
    OutputNode saved escapes are intentionally excluded from numerical last use.
    Physical identities/generations remain mandatory runtime admission checks.
    """
    policy = resolve_packed_offload(policy)
    from .runtime.pinned_arena import plan_pinned_placement
    residual_recipes = tuple(residual_recipes)
    forward_graph.validate()
    backward_graph.validate()
    fw_descriptor = _schedule_descriptor(forward_schedule)
    bw_descriptor = _schedule_descriptor(backward_schedule)
    if (sorted(map(id, forward_schedule.nodes())) != sorted(map(id, forward_graph.nodes))
            or sorted(map(id, backward_schedule.nodes())) != sorted(map(id, backward_graph.nodes))):
        raise ValueError("Offload planning requires schedules of the final forward/backward graphs")
    base = dict(policy=policy, forward_signature=_digest(fw_descriptor), backward_signature=_digest(bw_descriptor),
                _forward_token=_digest(forward_schedule.signature()), _backward_token=_digest(backward_schedule.signature()),
                profile_identity=_freeze({} if profile_identity is None else profile_identity))
    if not policy.enabled:
        plan = PackedOffloadPlan(**base)
        return replace(plan, signature=_digest(plan.as_dict()))

    if not saved_tensors:
        result = forward_graph.nodes[-1].result if isinstance(forward_graph.nodes[-1], OutputNode) else None
        if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], tuple):
            raise ValueError("Offload plan requires the native saved-tensor output ABI")
        saved_tensors = result[1]
    saved_tensors = tuple(saved_tensors)
    names = [value.name for value in saved_tensors]
    if len(names) != len(set(names)):
        raise ValueError("Offload saved slots require unique canonical component names")
    values = {value.name: value for value in saved_tensors}
    saved_indices = {value.name: index for index, value in enumerate(saved_tensors)}
    actual_outputs = set(find_tensor_values(forward_graph.nodes[-1].result[0]))
    births, fw_uses, unsafe_forward_uses = {}, defaultdict(list), set()
    for boundary, item in enumerate(forward_schedule.items):
        for value in scheduled_outputs(item):
            births[value.name] = boundary
        if not isinstance(item, OutputNode):
            for value in scheduled_inputs(item):
                fw_uses[value.name].append(boundary)
                if not _private_backward_consumer(item):
                    unsafe_forward_uses.add(value.name)

    input_nodes = [node for node in backward_graph.nodes if isinstance(node, InputNode)]
    seed_count = (len(backward_argument_sources) - len(saved_tensors) if backward_argument_sources
                  else len(input_nodes) - len(saved_tensors))
    if seed_count < 0:
        raise ValueError("Backward saved argument map is incomplete")
    backward_saved = {node.value: node.index - seed_count for node in input_nodes if node.index >= seed_count}
    if set(backward_saved.values()) != set(range(len(saved_tensors))):
        raise ValueError("Backward saved input slots do not match the forward saved ABI")
    if backward_argument_sources and tuple(backward_argument_sources[seed_count:]) != saved_tensors:
        raise ValueError("Backward saved argument sources changed after pack sharing")
    bw_uses, panel_contracts, unsafe_uses = defaultdict(list), {}, set()
    escaped_backward = set(find_tensor_values(backward_graph.nodes[-1].result))
    for boundary, item in enumerate(backward_schedule.items):
        if isinstance(item, (InputNode, OutputNode, GetAttrNode)):
            continue
        for value in scheduled_inputs(item):
            if value in backward_saved:
                source_name = saved_tensors[backward_saved[value]].name
                bw_uses[source_name].append(boundary)
                if not _private_backward_consumer(item):
                    unsafe_uses.add(source_name)
        panel = _panel_contract(item, backward_saved, saved_tensors, policy)
        if panel is not None:
            panel_contracts[f"backward_{boundary}"] = panel

    from .saved_state import _CODECS, _codec, _components

    descriptions = {name: dict(name=name, shape=value.meta.shape, dtype=str(value.meta.dtype), device=str(value.meta.device))
                    for name, value in values.items() if value.meta is not None}
    memberships = defaultdict(list)
    for record in residual_recipes:
        if not isinstance(record, Mapping):
            raise ValueError("Offload residual recipes must be metadata records")
        _freeze(record)
        for name in record.get("components", ()):
            memberships[name].append(record)
    rejections, candidates, seen = [], [], set()
    for record in residual_recipes:
        if record.get("representation") not in _CODECS:
            continue
        components = tuple(record.get("components", ()))
        if components in seen:
            continue
        seen.add(components)
        try:
            recipe = dict(record.get("recipe", ()))
            if recipe.get("bits") != 2 or type(recipe.get("bits")) is not int or recipe.get("group_size") != 1024:
                raise ValueError("only_bits2_group1024_supported")
            if len(components) != 3 or any(name not in values for name in components):
                raise ValueError("incomplete_saved_triplet")
            proof = _codec(record, descriptions, False)
            for name in components:
                for owner in memberships[name]:
                    _components(owner, descriptions)
                    if owner.get("representation") in _CODECS:
                        if _codec(owner, descriptions, False).get(name) != proof[name]:
                            raise ValueError("conflicting_component_ownership")
                    elif (tuple(owner.get("components", ())) != (name,)
                          or owner.get("physical_source", owner.get("source")) != name
                          or owner.get("storage_source") or owner.get("provenance") in ("parameter", "persistent_buffer", "constant")):
                        raise ValueError("unsupported_component_membership")
            leaves = tuple(values[name] for name in components)
            if any(value in actual_outputs or model_storage_source(value) is not None for value in leaves):
                raise ValueError("caller_model_or_output_alias")
            if any(name in unsafe_forward_uses for name in components):
                raise ValueError("unproven_non_aliasing_forward_consumer")
            if any(name in unsafe_uses for name in components):
                raise ValueError("unproven_non_aliasing_backward_consumer")
            if any(value in escaped_backward and saved_tensors[index].name in components
                   for value, index in backward_saved.items()):
                raise ValueError("backward_output_escape")
            if any(not _private_constructor(value) for value in leaves):
                raise ValueError("unproven_private_codec_constructor")
            descriptors = []
            for role, value in zip(("payload", "codec_step", "codec_minimum"), leaves):
                meta = value.meta
                if (meta is None or meta.requires_grad or getattr(meta, "presence", "required") != "required"
                        or not prod(meta.shape)
                        or str(meta.device).split(":", 1)[0] != "cuda"
                        or meta.stride is not None and meta.stride != _contiguous(meta.shape)):
                    raise ValueError("unsupported_component_device_layout_or_presence")
                descriptors.append(PackedOffloadComponent(value.name, saved_indices[value.name], tuple(meta.shape),
                    str(meta.dtype), tuple(meta.stride) if meta.stride is not None else _contiguous(meta.shape),
                    str(meta.device), prod(meta.shape) * meta.dtype.itemsize, role))
            if len({births[name] for name in components}) != 1:
                raise ValueError("non_atomic_triplet_producer")
            uses = sorted({boundary for name in components for boundary in bw_uses[name]})
            if not uses:
                raise ValueError("no_backward_consumer")
            bundle_id = "pack_" + _digest(components)[:16]
            candidates.append(PackedOffloadBundle(bundle_id, tuple(descriptors), births[components[0]],
                max((max(fw_uses[name], default=births[name]) for name in components)),
                tuple(f"backward_{boundary}" for boundary in uses), uses[0], uses[-1]))
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
            rejections.append(PackedOffloadRejection(components, str(error)))

    # Select by bytes times saved lifetime, then bytes, with deterministic ties.
    # This is a conservative initial policy, not proof of a globally minimal peak.
    by_components = {tuple(component.name for component in bundle.components): bundle for bundle in candidates}
    by_id = {bundle.bundle_id: bundle for bundle in candidates}
    panel_bundles = {}
    units = []
    for name, panel in tuple(panel_contracts.items()):
        triplets = tuple(panel.component_names[index:index + 3] for index in (0, 3, 6))
        if any(triplet not in by_components for triplet in triplets):
            del panel_contracts[name]
            continue
        identifiers = tuple(by_components[triplet].bundle_id for triplet in triplets)
        if len(set(identifiers)) != 3:
            del panel_contracts[name]
            continue
        panel_bundles[name] = identifiers
        # A panel's three operands are an atomic selection. Merge sharing
        # memberships rather than selecting a misleading partial panel.
        members = set(identifiers)
        overlapping = [unit for unit in units if unit.intersection(members)]
        for unit in overlapping:
            members.update(unit)
            units.remove(unit)
        units.append(members)
    grouped_ids = set().union(*units) if units else set()
    units.extend({bundle.bundle_id} for bundle in candidates if bundle.bundle_id not in grouped_ids)
    selected, occupied, consumer_bundles = [], set(), defaultdict(set)
    used_host = 0
    d2h_limit = min(policy.host_pinned_bytes, policy.d2h_inflight_bytes or policy.host_pinned_bytes)
    def priority(unit):
        bundles = tuple(by_id[name] for name in sorted(unit))
        return (-sum(item.bytes * (len(forward_schedule.items) - item.last_forward_use + item.first_backward_use) for item in bundles),
                -sum(item.bytes for item in bundles), tuple(sorted(unit)))

    for unit in sorted(units, key=priority):
        bundles = tuple(by_id[name] for name in sorted(unit))
        names = tuple(component.name for bundle in bundles for component in bundle.components)
        size = sum(bundle.bytes for bundle in bundles)
        affected = {name for bundle in bundles for name in bundle.consumers}
        reason = None
        if occupied.intersection(names):
            reason = "partially_overlapping_triplets"
        elif any(bundle.bytes > d2h_limit for bundle in bundles):
            reason = "whole_bundle_exceeds_pinned_or_D2H_budget"
        elif policy.physical_host_budget and any(plan_pinned_placement(
                bundle.components, budget_bytes=policy.host_pinned_bytes) is None for bundle in bundles):
            reason = "whole_bundle_exceeds_physical_pinned_capacity"
        elif not defer_host_budget and used_host + size > policy.canonical_host_bytes:
            reason = "canonical_history_budget_exceeded"
        elif policy.device_staging_bytes:
            for name in affected:
                prospective = consumer_bundles[name] | {bundle.bundle_id for bundle in bundles if name in bundle.consumers}
                panel = panel_contracts.get(name)
                panel_selected = panel is not None and prospective == set(panel_bundles[name])
                required = panel.staging_bytes if panel_selected else sum(by_id[identifier].bytes for identifier in prospective)
                if required > policy.device_staging_bytes:
                    reason = "atomic_consumer_staging_budget_exceeded"
                    break
        if reason:
            rejections.append(PackedOffloadRejection(names, reason))
            continue
        selected.extend(bundles)
        occupied.update(names)
        used_host += size
        for bundle in bundles:
            for consumer in bundle.consumers:
                consumer_bundles[consumer].add(bundle.bundle_id)
    if policy.admission == "error" and rejections:
        raise ValueError("Packed offload admission rejected requested inventory: " + "; ".join(row.reason for row in rejections))
    selected.sort(key=lambda item: (item.producer_boundary, item.bundle_id))
    grouped = defaultdict(list)
    for bundle in selected:
        for consumer in bundle.consumers:
            grouped[consumer].append(bundle)
    consumers = []
    for name, bundles in sorted(grouped.items(), key=lambda item: int(item[0].removeprefix("backward_"))):
        panel = panel_contracts.get(name)
        if panel is not None and set(bundle.bundle_id for bundle in bundles) == set(panel_bundles[name]):
            identifiers = panel_bundles[name]
        else:
            panel = None
            identifiers = tuple(bundle.bundle_id for bundle in bundles)
        consumers.append(PackedOffloadConsumer(name, int(name.removeprefix("backward_")), identifiers,
                                               sum(bundle.bytes for bundle in bundles), panel))
    consumers = tuple(consumers)
    # Coarse preceding numerical boundaries support calibrated late prefetch.
    numerical = tuple(index for index, item in enumerate(backward_schedule.items)
                      if not isinstance(item, (InputNode, OutputNode, GetAttrNode)))
    progress = tuple(sorted({numerical[0] for _ in consumers} | {
        max((index for index in numerical if index < consumer.schedule_index), default=consumer.schedule_index)
        for consumer in consumers}))
    plan = PackedOffloadPlan(**base, bundles=tuple(selected), consumers=consumers,
        rejections=tuple(rejections), progress_boundaries=progress)
    return plan.with_prefetches(plan_prefetches(plan))
