"""Default-off GroupNorm/SiLU composites preserving native saved-state slots."""

from math import prod

import torch

from ..ir import CallNode, ExternKernelNode, GetAttrNode, InputNode
from ..trace.metadata_utils import contiguous_stride
from .fusion_contracts import candidate, select
from .semantic import NativeActivation, recorded, result
from .spatial import Normalization
from .training_fusion import TrainingFusionABI


FORWARD = "group_norm_silu_residual_forward"
BACKWARD = "group_norm_silu_residual_backward"
ABI = TrainingFusionABI("group_norm_silu_saved_v1",
    ("activated", *Normalization.state_names, "activation_derivative", "activation_step", "activation_minimum"),
    ("normalized_affine_to_bf16_before_silu", "coefficient_product_to_bf16_before_norm_reductions"))


def _enabled(context, phase):
    flag = getattr(context, "fuse_norm_activation_residuals", False)
    stage = getattr(context, "norm_activation_fusion_stage", "off")
    if type(flag) is not bool or stage not in ("off", "forward", "forward_backward") or flag != (stage != "off"):
        raise ValueError("Invalid norm activation residual fusion flag/stage")
    return flag and phase in getattr(context, "fusion_phases", ("forward", "backward")) and (
        phase == "forward" or stage == "forward_backward")


def _trusted(node, spec, method):
    numerical = getattr(node.target, "numerical_function", None)
    node_type = ExternKernelNode if method == "dx" else CallNode
    return (type(node) is node_type and getattr(node.target, "specification", None) is spec
            and getattr(numerical, "__self__", None) is spec
            and getattr(numerical, "__func__", None) is getattr(type(spec), method)
            and getattr(node.target, "__code__", None) is recorded(spec, getattr(spec, method), "forward").__code__)


def _annotations(nodes):
    return any(node.scheduling_constraints
        or any(getattr(owner, key, False) for owner in (node, node.target)
               for key in ("stateful", "effects", "side_effects", "mutates", "mutation", "alias_writes"))
        or any(getattr(owner, key, None) is not None for owner in (node, node.target) for key in ("stream", "stream_id"))
        for node in nodes)


def _saved_metadata(spec, normalization, activation):
    from .semantic import packed_metas, meta

    device, shape = spec.metas[0].device, spec.geometry.shape
    groups = packed_metas(spec.geometry.matrix_shape, spec.compression, device, (1, 1024))
    empty = meta((0,), torch.bfloat16, device)
    stats = meta((spec.geometry.domains,), torch.float32, device)
    affine = meta((shape[1],), torch.float32, device) if spec.needs[0] and spec.metas[1] is not None else empty
    expected = (*groups, stats, empty, affine,
                stats if spec.compression.get("norm_xhat_recenter", False) else empty, meta((), torch.int64, "cpu"))
    actual = tuple(value.meta for value in normalization.values[1:])
    activation_expected = packed_metas((prod(shape[:-1]), shape[-1]), spec.compression, device)
    activation_actual = tuple(value.meta for value in activation.values[1:])
    return actual == expected and activation_actual == activation_expected


def _reason(spec, activation):
    if type(spec) is not Normalization or type(activation) is not NativeActivation:
        return "builtin_group_norm_and_activation_required"
    if (spec.kind != "group" or spec.empty or spec.native_group is not None or spec.output_layout is not None
            or spec.policy.enabled or "fp8_training" in spec.plan or not spec.training):
        return "ordinary_training_group_norm_required"
    if spec.normalization_variant not in ("auto", "baseline"):
        return "baseline_normalization_statistics_required"
    if (activation.kind != "silu" or activation.native_torch_activation or activation.recompute
            or not activation.need_dx or activation.geometry.layout != "last"):
        return "packed_silu_coefficient_required"
    if (not spec.compression["enabled"] or not activation.compression["enabled"]
            or any(options.get("bits") != 2 or options.get("group_size") != 1024
                   for options in (spec.compression, activation.compression))):
        return "two_bit_group1024_required"
    shape = tuple(spec.geometry.shape)
    if len(shape) != 4 or any(size <= 0 for size in shape) or spec.geometry.elements <= 4096:
        return "non_small_positive_nchw_geometry_required"
    if spec.geometry.elements % 1024:
        return "normalization_and_flat_activation_groups_misaligned"
    source = spec.metas[0]
    if (source.dtype != torch.bfloat16 or spec.output_dtype != torch.bfloat16
            or activation.input_meta.dtype != torch.bfloat16 or activation.output_dtype != torch.bfloat16
            or activation.input_meta.shape != shape or source.storage_dtype is not None
            or source.representation != "dense"):
        return "dense_bf16_rounding_contract_required"
    if source.device is None or torch.device(source.device).type != "cuda":
        return "cuda_only_residual_fusion"
    if spec.backend != "triton" or source.stride != contiguous_stride(shape):
        return "contiguous_nchw_source_required"
    if any(meta is not None and (meta.dtype != torch.float32 or meta.shape != (shape[1],)
               or meta.stride != (1,) or torch.device(meta.device) != torch.device(source.device))
           for meta in spec.metas[1:3]):
        return "contiguous_fp32_affine_contract_required"
    if any(spec.kernel_policy.route(family, role) != "triton" for family, role in (
            ("normalization", "normalization.forward"), ("normalization", "normalization.backward"),
            ("activation_codec", "normalization.forward"), ("activation", "activation.forward"),
            ("activation_codec", "normalization.backward"), ("activation_codec", "activation.forward"),
            ("activation_codec", "activation.backward"), ("activation", "activation.backward"))):
        return "triton_role_contract_required"
    if any(route != "triton" for route in activation.kernel_routes.values()):
        return "activation_backend_conflict"
    return None


class NormActivationResidual:
    """Invocation-independent numerical specification with twelve output slots."""

    def __init__(self, normalization, activation, metadata):
        self.normalization, self.activation_spec = normalization, activation
        self.activation = ("silu", "none")
        self.output_metadata = tuple(metadata)
        self.forward_steps = ("unchanged_group_statistics", "affine_silu_and_two_distinct_packs")
        self.backward_steps = ("saved_coefficient_norm_reductions", "saved_coefficient_affine_reductions")

    def __getattr__(self, name):
        return getattr(self.normalization, name)

    def forward(self, x, weight, bias, running_mean, running_var, counter):
        if (type(x) is not torch.Tensor or x.device.type != "cuda" or x.dtype != torch.bfloat16
                or tuple(x.shape) != self.normalization.geometry.shape or not x.is_contiguous()):
            raise ValueError("Fused GroupNorm/SiLU requires its declared CUDA BF16 contiguous input")
        from deepopt.opt.backends.triton.normalization import forward
        from deepopt.opt.nn.normalization.geometry import output_strides
        from deepopt.opt.nn.normalization.policy import normalization_policy_receipt
        from deepopt.opt._execution import finite_checks_enabled

        spec = self.normalization
        saved_activation = tuple(torch.empty(meta.shape, dtype=meta.dtype, device=x.device)
                                 for meta in self.output_metadata[-3:])
        center = (torch.empty(spec.geometry.domains, dtype=torch.float32, device=x.device)
                  if spec.compression.get("norm_xhat_recenter", False) else None)
        spec.module.last_execution = info = {}
        output, mean, inverse, packed = forward(x, weight, bias, spec.geometry, spec.compression,
            out_dtype=torch.bfloat16, center_shift=center, execution_info=info,
            activation=("silu", "none"), activation_saved=saved_activation)
        if center is not None and finite_checks_enabled() and not (
                torch.isfinite(inverse) & (inverse > 0) & torch.isfinite(center)).all():
            raise ValueError("Normalization xhat correction requires positive finite rstd and finite codec error")
        if tuple(output.stride()) != output_strides(x):
            raise ValueError("Fused GroupNorm/SiLU output stride differs from its declared output")
        selector = spec.policy.select(x)
        empty = x.new_empty((0,))
        saved = (packed.data, packed.step, packed.minimum, inverse if spec.need_h or spec.needs[0] else empty,
                 empty, weight if spec.needs[0] and weight is not None else empty,
                 center if center is not None else empty, selector if selector is not None else spec.empty_selector)
        info.update(activation_residual_fused=True, compression=True, compression_target="x_hat",
                    norm_xhat_recenter=center is not None, materialized_normalization_output_bytes=0,
                    saved_state_bytes=sum(t.numel() * t.element_size() for t in (*saved, *saved_activation)))
        info.update(normalization_policy_receipt(spec.bf16_policy, x, torch.bfloat16, "triton"),
                    recenter_state_bytes=center.numel() * center.element_size() if center is not None else 0,
                    recenter_rstd_bytes=inverse.numel() * inverse.element_size() if center is not None else 0,
                    recenter_extra_rstd_bytes=0)
        self.activation_spec.module.last_execution = {"backend": "triton_group_norm_silu", "compression": True,
            "saved_state_bytes": sum(t.numel() * t.element_size() for t in saved_activation)}
        return result(ABI.check_runtime((output, *saved, *saved_activation), self.output_metadata))

    def backward(self, gradient, a_data, a_step, a_minimum, *saved):
        if (type(gradient) is not torch.Tensor or gradient.device.type != "cuda" or gradient.dtype != torch.bfloat16
                or tuple(gradient.shape) != self.normalization.geometry.shape or gradient.layout != torch.strided):
            raise ValueError("Fused GroupNorm/SiLU requires its declared CUDA BF16 cotangent")
        from deepopt.opt.backends.triton.normalization import backward
        from deepopt.opt.quantization.activation import PackedActivation
        from deepopt.opt.backends._shared.normalization import saved_normalization_variant

        spec = self.normalization
        data, step, minimum, inverse, mean, weight, center, selector = saved
        packed = PackedActivation(data, step, minimum, spec.geometry.matrix_shape, 1024, (1, 1024), 2)
        if spec.policy.slot(selector) is not None:
            raise ValueError("Residual fusion has no prepared gradient transport contract")
        # Keep the same saved-normalization variant scope as the original VJP.
        @saved_normalization_variant
        def run(owner):
            return backward(gradient, weight if weight.numel() else None, spec.geometry, packed=packed,
                inverse=inverse if inverse.numel() else None, center_shift=center if center.numel() else None,
                input_dtype=spec.metas[0].dtype, need_dx=spec.needs[0], need_dw=spec.needs[1], need_db=spec.needs[2],
                derivative_saved=(a_data, a_step, a_minimum), execution_info=spec.module.last_execution)
        values = run(self)
        return result([value.to(meta.dtype) for value, meta, need in zip(values, spec.metas[:3], spec.needs) if need])


def _select(node, record, phase):
    roles = ({"normalization_silu_and_derivative_pack": ["_apply"]} if phase == "forward" else
             {"coefficient_norm_gradient": ["_partial_backward", "_input_gradient", "_partial_affine"]})
    select(node, record, roles=roles, consumed=("normalization", "activation.coefficient"), source_files=(
        "compiler/autodiff/norm_activation_residuals.py", "compiler/autodiff/training_fusion.py",
        "opt/backends/_shared/normalization_activation.py", "opt/backends/triton/normalization/__init__.py",
        "opt/backends/triton/normalization/forward.py", "opt/backends/triton/normalization/backward.py",
        "opt/backends/triton/normalization/activation.py", "opt/backends/triton/normalization/common.py",
        "opt/backends/triton/normalization/recenter.py", "opt/backends/triton/normalization/statistics.py",
        "opt/backends/triton/activation/math.py", "opt/backends/triton/quantization/activation/affine.py"))
    record["kernel_evidence"]["forbidden_kernel_symbols"] = [
        "_activation_forward" if phase == "forward" else "_activation_backward"]
    record.update(qualification="experimental_CUDA_not_yet_target_qualified", native_autodiff_owner=True,
                  statistics_contract="unchanged_baseline", global_peak_saving_bytes=None)


def forward(graph, context):
    from .optimization import rebuild

    enabled = _enabled(context, "forward")
    diagnostics, patterns = [], []
    for node in tuple(graph.nodes):
        if type(node) is not CallNode or type(node.specification) is not Normalization:
            continue
        spec = node.specification
        for consumer in tuple(node.values[0].users):
            activation = getattr(consumer, "specification", None)
            if type(activation) is not NativeActivation or activation.kind != "silu":
                continue
            record = candidate(graph, node, spec, "forward", FORWARD, requested=enabled)
            diagnostics.append(record)
            if not enabled:
                continue
            reason = _reason(spec, activation)
            if reason is None and (not _trusted(node, spec, "forward") or not _trusted(consumer, activation, "forward")):
                reason = "trusted_native_numerical_executors_required"
            if reason is None and (node.kwargs or consumer.kwargs or len(node.args) != 6
                    or node.args[3:] != (None, None, None) or consumer.args != (node.values[0], None, None)):
                reason = "ordinary_normalization_activation_arguments_required"
            if reason is None and node.values[0].users != [consumer]:
                reason = "normalization_output_fanout"
            between = graph.nodes[graph.nodes.index(node) + 1:graph.nodes.index(consumer)]
            if reason is None and any(type(other) not in (GetAttrNode, InputNode) for other in between):
                reason = "intervening_operation"
            if reason is None and (len(node.values) != 9 or len(consumer.values) != 4):
                reason = "declared_saved_state_arity_required"
            if reason is None and (_annotations((node, consumer)) or not _saved_metadata(spec, node, consumer)):
                reason = "saved_metadata_or_effect_contract"
            if reason is None and (node.saved_tensors_for_bwd != node.values[1:]
                                   or consumer.saved_tensors_for_bwd != consumer.values[1:]):
                reason = "unexpected_saved_owner_annotation"
            if reason:
                record.update(eligible=False, reason=reason)
                continue
            values = (consumer.values[0], *node.values[1:], *consumer.values[1:])
            wrapper = NormActivationResidual(spec, activation, tuple(value.meta for value in values))
            spec._norm_activation_residual_fusion = wrapper
            replacement = CallNode("normalization.silu_residual_forward", recorded(wrapper, wrapper.forward, "forward"),
                                   node.args, {}, values, wrapper)
            replacement.bf16_policy = node.bf16_policy
            replacement.saved_tensors_for_bwd = (*node.saved_tensors_for_bwd, *consumer.saved_tensors_for_bwd)
            replacement.activation_save_policy = node.activation_save_policy
            for value in replacement.values:
                value.fresh_allocation = None
            graph.nodes[graph.nodes.index(node)] = replacement
            graph.nodes.remove(consumer)
            _select(replacement, record, "forward")
            record.update(saved_output_abi=ABI.receipt(values), removed_normalization_output_bytes=prod(spec.geometry.shape) * 2)
            patterns.append(FORWARD)
            rebuild(graph)
            break
    return diagnostics, tuple(patterns)


def backward(graph, context, diagnostics=None):
    from .optimization import rebuild

    enabled = _enabled(context, "backward")
    patterns = []
    for node in tuple(graph.nodes):
        spec = getattr(node.target, "specification", None) if type(node) is CallNode else None
        if type(spec) is not Normalization or node.op != "normalization.vjp":
            continue
        gradient = node.args[0]
        derivative = gradient.producer
        activation = getattr(getattr(derivative, "target", None), "specification", None)
        if type(activation) is not NativeActivation or getattr(derivative, "op", None) != "activation.coefficient_vjp":
            continue
        record = candidate(graph, node, spec, "backward", BACKWARD, requested=enabled)
        if diagnostics is not None:
            diagnostics.append(record)
        if not enabled:
            continue
        wrapper = getattr(spec, "_norm_activation_residual_fusion", None)
        reason = _reason(spec, activation)
        if reason is None and (wrapper is None or wrapper.activation_spec is not activation):
            reason = "matching_forward_residual_composite_required"
        if reason is None and (not _trusted(node, spec, "gradients") or not _trusted(derivative, activation, "dx")):
            reason = "trusted_native_vjp_executors_required"
        if reason is None and (gradient.users != [node] or len(derivative.args) != 4 or len(node.args) != 9):
            reason = "single_gradient_consumer_and_declared_saved_state_required"
        between = graph.nodes[graph.nodes.index(derivative) + 1:graph.nodes.index(node)]
        if reason is None and any(type(other) not in (GetAttrNode, InputNode) for other in between):
            reason = "intervening_backward_operation"
        if reason is None and (_annotations((node, derivative)) or node.saved_tensors_for_bwd
                               or derivative.saved_tensors_for_bwd):
            reason = "effect_or_stream_contract"
        if reason is None and (derivative.args[0].meta.dtype != torch.bfloat16 or gradient.meta.dtype != torch.bfloat16):
            reason = "bf16_activation_adjoint_boundary_required"
        if reason:
            record.update(eligible=False, reason=reason)
            continue
        replacement = CallNode("normalization.silu_residual_backward", recorded(wrapper, wrapper.backward, "backward"),
                               (*derivative.args, *node.args[1:]), {}, node.values, wrapper)
        replacement.bf16_policy = node.bf16_policy
        for value in replacement.values:
            value.fresh_allocation = None
        graph.nodes[graph.nodes.index(node)] = replacement
        graph.nodes.remove(derivative)
        _select(replacement, record, "backward")
        record.update(removed_activation_gradient_bytes=prod(spec.geometry.shape) * 2,
                      coefficient_decode="tile_local_flat1024", materialized_activation_gradient_bytes=0,
                      cotangent_layout_policy="contiguous_before_normalization_reductions",
                      raw_cotangent_copy="only_for_noncontiguous_input_recorded_in_execution_info",
                      raw_cotangent_copy_max_bytes=prod(spec.geometry.shape) * 2)
        patterns.append(BACKWARD)
        rebuild(graph)
    return tuple(patterns)
