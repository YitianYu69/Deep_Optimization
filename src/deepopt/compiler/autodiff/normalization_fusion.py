"""Post-differentiation normalization/activation kernel replacements.

These passes consume already-derived native VJPs. Forward residual identities
stay public IR results; xhat compression never substitutes for an activation's
preactivation or derivative codec. Unsupported state/transport contracts remain
explicit numerical operations with a visible rejection.
"""

from math import prod

import torch

from ..ir import CallNode, ExternKernelNode, GetAttrNode, InputNode, PointwiseNode
from .semantic import LibraryLayerNorm, NativeActivation, NativeLayerNorm, recorded, result
from .spatial import Normalization
from deepopt.opt.backends._shared.normalization import saved_normalization_variant


def _normalization(spec):
    return type(spec) in (NativeLayerNorm, LibraryLayerNorm, Normalization)


def _activation(node):
    if isinstance(node, PointwiseNode) and node.op in ("gelu", "silu"):
        return node.op, node.kwargs.get("approximate", "none")
    if isinstance(node, CallNode) and type(node.specification) is NativeActivation:
        return node.specification.kind, node.specification.approximate
    return None


def _enabled(context, phase):
    return bool(getattr(context, "fuse_deepopt_activations", True)
                and phase in getattr(context, "fusion_phases", ("forward", "backward")))


def _reason(spec, source, activation, phase, context=None):
    if type(spec) is LibraryLayerNorm:
        return "library_normalization_backend"
    if getattr(spec, "native_group", None) is not None:
        if not getattr(context, "qualify_native_group_fusion", False):
            return "native_group_norm_statistics_contract"
        native = spec.native_group
        if (native.compute_dtype not in (torch.bfloat16, torch.float32)
                or native.stats_dtype not in (torch.bfloat16, torch.float32)
                or any(size <= 0 for size in spec.geometry.shape)):
            return "native_group_norm_staged_contract"
    if isinstance(spec, Normalization) and spec.kind != "group":
        return "stateful_batch_normalization"
    if activation[0] not in ("silu", "gelu") or activation[1] not in ("none", "tanh"):
        return "activation_contract"
    if torch.device(source.meta.device).type != "cuda":
        return "device"
    if (source.meta.representation != "dense" or source.meta.storage_dtype is not None
            or source.meta.dtype not in (torch.bfloat16, torch.float32)
            or spec.output_dtype not in (torch.bfloat16, torch.float32)):
        return "native_dense_dtype_required"
    if isinstance(spec, Normalization) and spec.output_layout is not None:
        return "fp8_output_transport"
    if phase == "backward" and getattr(getattr(spec, "policy", None), "enabled", False):
        return "prepared_fp8_gradient_edge"
    policy = spec.kernel_policy
    roles = [("normalization", f"normalization.{phase}"), ("activation", f"activation.{phase}"),
             ("compiler_pointwise", f"compiler.{activation[0]}" if phase == "forward" else "compiler.backward")]
    if spec.compression["enabled"]:
        roles.append(("activation_codec", f"normalization.{phase}"))
    if policy.route("normalization", f"normalization.{phase}") != "triton":
        return "normalization_backend_has_no_activation_epilogue"
    if any(policy.pinned(family, role) and policy.route(family, role) != "triton" for family, role in roles):
        return "consumed_role_backend_conflict"
    return None


def _candidate(graph, node, spec, phase, activation, context):
    kind = "layer_norm" if isinstance(spec, NativeLayerNorm) else "group_norm" if spec.kind == "group" else "batch_norm"
    pattern = f"{kind}_{activation[0]}_{'epilogue' if phase == 'forward' else 'vjp'}"
    from .fusion_contracts import candidate

    record = candidate(graph, node, spec, phase, pattern, requested=_enabled(context, phase))
    record.update(precision="native", qualification="unqualified", requested_schedule=None,
                  selected_schedule=None, phase="train" if spec.module.training else "eval")
    return record


class _NormActivation:
    """A numerical kernel specification, with no invocation-owned tensors."""

    def __init__(self, normalization, activation, retain_preactivation=False):
        self.normalization = normalization
        self.activation = activation
        self.retain_preactivation = retain_preactivation
        self.forward_steps = ("normalization_statistics", "normalization_affine_activation_apply")
        self.backward_steps = ("activation_derivative_norm_reductions", "bounded_affine_finish")
        self.source = "DeepOpt native-derived normalization/activation Triton epilogues (CUDA unqualified)"

    def __getattr__(self, name):
        return getattr(self.normalization, name)

    def forward(self, x, weight, bias, *extra):
        spec = self.normalization
        if getattr(spec, "native_group", None) is not None:
            return self.native_forward(x, weight, bias)
        if isinstance(spec, NativeLayerNorm):
            from deepopt.opt.nn.autograd.normalization.layer import executor
            from deepopt.opt.nn.autograd._output import restore_fresh_output
            from deepopt.opt._execution import finite_checks_enabled

            preactivation = (torch.empty(spec.input_meta.shape, device=x.device, dtype=spec.output_dtype)
                             if self.retain_preactivation else None)
            center = torch.empty(spec.geometry.rows, device=x.device, dtype=torch.float32) if spec.compression["enabled"] else None
            gain = torch.empty_like(center) if center is not None else None
            output, _, state, _, _, info = executor(x, spec.kernel_policy.backend).forward(
                x, spec.geometry, weight=weight, bias=bias, dtype=spec.output_dtype, block_shape=None,
                compression=spec.compression, check_finite=finite_checks_enabled() and spec.compression["enabled"],
                center_shift=center, moment_gain=gain, activation=self.activation, preactivation=preactivation)
            saved = ([state.packed.data, state.packed.step, state.packed.minimum, state.inverse, center, gain]
                     if state.packed is not None else [state.mean, state.inverse] if spec.need_h else [])
            spec.module.last_execution = {**info, "compression": spec.compression["enabled"],
                "compression_target": "x_hat", "norm_xhat_recenter": center is not None,
                "layer_norm_xhat_moment_correction": gain is not None,
                "saved_state_bytes": sum(t.numel() * t.element_size() for t in saved)}
            output = restore_fresh_output(output, spec.geometry.transport.restore)
        else:
            from deepopt.opt.backends.normalization import executor
            from deepopt.opt.nn.normalization.geometry import output_strides
            from deepopt.opt._execution import finite_checks_enabled

            preactivation = (torch.empty_strided(spec.geometry.shape, output_strides(x), device=x.device,
                                                dtype=spec.output_dtype) if self.retain_preactivation else None)
            center = (torch.empty(spec.geometry.domains, device=x.device, dtype=torch.float32)
                      if spec.compression["enabled"] and spec.compression.get("norm_xhat_recenter", False) else None)
            spec.module.last_execution = info = {}
            # output_dtype was captured at trace time: backward and later
            # invocations do not re-decide CUDA autocast GroupNorm precision.
            output, mean, inverse, packed = executor(x, spec.backend).forward(
                x, weight, bias, spec.geometry, spec.compression, out_dtype=spec.output_dtype,
                center_shift=center, execution_info=info, activation=self.activation, preactivation=preactivation)
            if center is not None and finite_checks_enabled() and not (
                    torch.isfinite(inverse) & (inverse > 0) & torch.isfinite(center)).all():
                raise ValueError("Normalization xhat correction requires positive finite rstd and finite codec error")
            selector = spec.policy.select(x)
            saved = [packed.data if packed else x if spec.need_h else None,
                     packed.step if packed else None, packed.minimum if packed else None,
                     inverse if spec.need_h or spec.needs[0] else None,
                     mean if spec.need_h and packed is None else None,
                     weight if spec.needs[0] else None, center]
            info.update(compression=spec.compression["enabled"], compression_target="x_hat",
                        norm_xhat_recenter=center is not None,
                        saved_state_bytes=sum(t.numel() * t.element_size() for t in saved if t is not None))
            empty = x.new_empty((0,))
            saved = [value if value is not None else empty for value in saved] if spec.training else [empty] * 7
            saved.append(selector if selector is not None else spec.empty_selector)
        return result([output, *([preactivation] if self.retain_preactivation else []), *saved])

    def native_contract(self):
        from deepopt.opt.nn.normalization.native_group import NativeGroupContract

        spec = self.normalization
        native = spec.native_group
        return NativeGroupContract(tuple(spec.geometry.shape), native.groups, native.eps,
            native.compute_dtype, native.stats_dtype, spec.metas[0].dtype,
            native.parameter_dtype, native.cast_affine)

    @staticmethod
    def native_executor(tensor):
        if tensor.device.type == "cuda":
            from deepopt.opt.backends.triton.normalization import native_group
        else:
            from deepopt.opt.backends.reference.normalization import native_group
        return native_group

    def native_forward(self, x, weight, bias):
        from deepopt.opt.nn.normalization.geometry import output_strides

        spec = self.normalization
        contract = self.native_contract()
        preactivation = (torch.empty_strided(contract.shape, output_strides(x), device=x.device,
                            dtype=contract.compute_dtype) if self.retain_preactivation else None)
        spec.module.last_execution = info = {}
        output, mean, inverse, source, gamma = self.native_executor(x).forward(
            x, weight, bias, contract, activation=self.activation,
            preactivation=preactivation, execution_info=info)
        empty = x.new_empty((0,))
        selector = spec.policy.select(x)
        saved = [source if spec.need_h else empty, empty, empty,
                 inverse if spec.need_h else empty, mean if spec.need_h else empty,
                 gamma if spec.needs[0] and gamma is not None else empty, empty,
                 selector if selector is not None else spec.empty_selector]
        info.update(compression=False, saved_state_bytes=sum(t.numel() * t.element_size() for t in saved))
        return result([output, *([preactivation] if self.retain_preactivation else []), *saved])

    def native_backward(self, gradient, preactivation, *args):
        spec = self.normalization
        source, _, _, inverse, mean, weight, _, _ = args
        values = self.native_executor(gradient).backward(gradient,
            source if source.numel() else None, mean if mean.numel() else None,
            inverse if inverse.numel() else None, weight if weight.numel() else None,
            self.native_contract(), needs=spec.needs[:3], derivative_input=preactivation,
            derivative_activation=self.activation, execution_info=spec.module.last_execution)
        return result([value.to(meta.dtype) for value, meta, need in zip(values, spec.metas[:3], spec.needs) if need])

    @saved_normalization_variant
    def backward(self, gradient, preactivation, *args):
        spec = self.normalization
        if getattr(spec, "native_group", None) is not None:
            return self.native_backward(gradient, preactivation, *args)
        from deepopt.opt.quantization.activation import PackedActivation

        packed = center = gain = mean = inverse = None
        if isinstance(spec, NativeLayerNorm):
            from deepopt.opt.nn.autograd.normalization.layer import executor

            x, weight, bias, *state = args
            if spec.compression["enabled"]:
                data, step, minimum, inverse, center, gain = state
                group = spec.compression["group_size"]
                packed = PackedActivation(data, step, minimum, spec.geometry.matrix_shape, group, (1, group), spec.compression["bits"])
            elif spec.need_h:
                mean, inverse = state
            values = executor(gradient, spec.kernel_policy.backend).backward(
                gradient, weight, spec.geometry, packed=packed, x=x, mean=mean, inverse=inverse,
                center_shift=center, moment_gain=gain, need_dx=spec.needs[0], need_dw=spec.needs[1],
                need_db=spec.needs[2], input_dtype=spec.input_meta.dtype,
                execution_info=spec.module.last_execution, derivative_input=preactivation,
                derivative_activation=self.activation)
            metadata = (spec.input_meta, *spec.affine_metas)
        else:
            from deepopt.opt.backends.normalization import executor

            data, step, minimum, inverse, mean, weight, center, selector = args
            x = data if data.numel() and not spec.compression["enabled"] else None
            if spec.compression["enabled"]:
                group = spec.compression["group_size"]
                packed = PackedActivation(data, step, minimum, spec.geometry.matrix_shape, group, (1, group), spec.compression["bits"])
            values = executor(gradient, spec.backend).backward(
                gradient, weight if weight.numel() else None, spec.geometry, packed=packed, x=x,
                mean=mean if mean.numel() else None, inverse=inverse if inverse.numel() else None,
                center_shift=center if center.numel() else None, input_dtype=spec.metas[0].dtype,
                need_dx=spec.needs[0], need_dw=spec.needs[1], need_db=spec.needs[2],
                execution_info=spec.module.last_execution, derivative_input=preactivation,
                derivative_activation=self.activation)
            metadata = spec.metas[:3]
        return result([value.to(meta.dtype) for value, meta, needed in zip(values, metadata, spec.needs) if needed])


def _evidence(candidate, spec, phase, wrapper):
    site = spec.module_path or spec.node_name
    candidate["profile_range"] = f"deepopt.compiler.{site}.{phase}.{'forward' if phase == 'forward' else 'backward'}"
    if phase == "forward":
        roles = {"normalization_activation_apply": ["_row_forward", "_epilogue"]
                 if isinstance(spec, NativeLayerNorm) else ["_apply", "_small_forward"]}
    elif isinstance(spec, NativeLayerNorm):
        roles = {}
        if spec.needs[0]:
            roles["derivative_input_gradient"] = ["_row_backward", "_input_gradient", "_fused_backward", "_tiled_backward"]
        if spec.needs[1] or spec.needs[2]:
            roles["derivative_affine_gradient"] = ["_affine_partial", "_fused_backward", "_tiled_backward"]
    else:
        roles = {}
        if spec.needs[0]:
            roles["derivative_input_gradient"] = ["_input_gradient"]
        if spec.needs[1] or spec.needs[2]:
            roles["derivative_affine_gradient"] = ["_partial_affine"]
    if getattr(spec, "native_group", None) is not None:
        roles = ({"normalization_activation_apply": ["_native_group_apply"]} if phase == "forward" else
                 {"derivative_reductions": ["_native_ds_db"],
                  **({"derivative_input_gradient": ["_native_group_dx"]} if spec.needs[0] else {}),
                  **({"derivative_affine_gradient": ["_native_affine_finish"]} if any(spec.needs[1:3]) else {})})
        candidate.update(qualification="experimental_cuda_unqualified",
                         native_statistics_contract=wrapper.native_contract().receipt())
    candidate["kernel_evidence"] = {"required_kernel_roles": roles,
        "forbidden_kernel_symbols": ["_activation_forward"] if phase == "forward" else ["_activation_backward"]}
    candidate["implementation"] = "triton_normalization_activation"
    candidate["original_operations"] = ("normalization", wrapper.activation[0]) if phase == "forward" else (f"{wrapper.activation[0]}_backward", "normalization_vjp")
    candidate["dtype_boundaries"] = ("affine_output_cast_before_activation", "derivative_product_cast_before_normalization")
    candidate["residual_contract"] = "xhat_codec_and_exact_preactivation_are_distinct"


def _select(node, candidate, spec, phase):
    from .fusion_contracts import select

    suffix = "layer/" if isinstance(spec, NativeLayerNorm) else ""
    sources = (f"opt/backends/triton/normalization/{suffix}{phase}.py",
               f"opt/backends/triton/normalization/{suffix}{'launch' if phase == 'forward' else 'backward_launch'}.py"
               if suffix else "opt/backends/triton/normalization/__init__.py",
               "opt/backends/triton/normalization/activation.py")
    if not suffix and phase == "forward":
        sources += ("opt/backends/triton/normalization/small.py",)
    if getattr(spec, "native_group", None) is not None:
        sources = ("opt/backends/triton/normalization/native_group.py",
                   "opt/backends/triton/normalization/statistics.py",
                   "opt/backends/triton/normalization/activation.py",
                   "opt/nn/normalization/native_group.py")
    sources += ("compiler/autodiff/normalization_fusion.py",)
    evidence = candidate["kernel_evidence"]
    select(node, candidate, roles=evidence["required_kernel_roles"],
           consumed=candidate["original_operations"], source_files=sources)
    candidate["kernel_evidence"]["forbidden_kernel_symbols"] = evidence["forbidden_kernel_symbols"]


def forward(graph, context):
    from .optimization import rebuild

    candidates, patterns = [], []
    for node in list(graph.nodes):
        if not isinstance(node, CallNode) or not _normalization(node.specification):
            continue
        spec = node.specification
        activations = [(user, _activation(user)) for user in node.values[0].users]
        activations = [(user, activation) for user, activation in activations if activation]
        for consumer, activation in activations:
            candidate = _candidate(graph, node, spec, "forward", activation, context)
            candidates.append(candidate)
            if not _enabled(context, "forward"):
                continue
            reason = _reason(spec, node.args[0], activation, "forward", context)
            if reason is None and not isinstance(consumer, PointwiseNode):
                reason = "activation_saved_coefficient_codec"
            if reason is None and consumer.args != (node.values[0],):
                reason = "activation_arguments"
            between = graph.nodes[graph.nodes.index(node) + 1:graph.nodes.index(consumer)]
            if reason is None and any(not isinstance(item, (GetAttrNode, InputNode)) for item in between):
                reason = "intervening_operation"
            if reason:
                candidate["reason"] = reason
                continue
            original = node.values[0]
            retained = any(user is not consumer for user in original.users)
            wrapper = _NormActivation(spec, activation, retained)
            node.values = (consumer.value, *([original] if retained else []), *node.values[1:])
            node.specification = wrapper
            node.op = "normalization.activation_forward"
            node.target = recorded(wrapper, wrapper.forward, "forward")
            graph.nodes.remove(consumer)
            _evidence(candidate, spec, "forward", wrapper)
            candidate.update(eligible=True, selected=True, status="selected", reason="selected",
                retained_preactivation_bytes=prod(original.meta.shape) * torch.empty((), dtype=original.meta.dtype).element_size() if retained else 0,
                public_outputs=tuple(value.name for value in node.values))
            _select(node, candidate, spec, "forward")
            patterns.append(candidate["pattern"])
            rebuild(graph)
            break
    return candidates, tuple(patterns)


def backward(graph, context=None, diagnostics=None):
    from .optimization import rebuild

    patterns = []
    for node in list(graph.nodes):
        if not isinstance(node, CallNode) or node.op not in ("normalization.layer_norm_vjp", "normalization.vjp"):
            continue
        spec = getattr(node.target, "specification", None)
        if not _normalization(spec):
            continue
        gradient = node.args[0]
        derivative = gradient.producer
        if getattr(derivative, "op", None) == "activation.coefficient_vjp":
            coefficient = getattr(derivative.target, "specification", None)
            if type(coefficient) is NativeActivation:
                activation = (coefficient.kind, coefficient.approximate)
                candidate = _candidate(graph, node, spec, "backward", activation, context)
                if _enabled(context, "backward"):
                    candidate["reason"] = "activation_saved_coefficient_codec"
                if diagnostics is not None:
                    diagnostics.append(candidate)
            continue
        if not isinstance(derivative, (PointwiseNode, ExternKernelNode)) or derivative.op not in ("gelu_backward", "silu_backward"):
            continue
        activation = (derivative.op.removesuffix("_backward"), derivative.kwargs.get("approximate", "none"))
        candidate = _candidate(graph, node, spec, "backward", activation, context)
        if diagnostics is not None:
            diagnostics.append(candidate)
        if not _enabled(context, "backward"):
            continue
        upstream, preactivation = derivative.args[:2]
        reason = _reason(spec, preactivation, activation, "backward", context)
        if reason is None and gradient.users != [node]:
            reason = "activation_gradient_fanout"
        if reason is None and (upstream.meta.representation != "dense" or upstream.meta.storage_dtype is not None):
            reason = "prepared_fp8_upstream_gradient"
        if reason is None and (gradient.meta.dtype != preactivation.meta.dtype or gradient.meta.dtype != spec.output_dtype):
            reason = "derivative_product_cast_contract"
        if reason:
            candidate["reason"] = reason
            continue
        wrapper = _NormActivation(spec, activation)
        node.args = (upstream, preactivation, *node.args[1:])
        node.target = recorded(wrapper, wrapper.backward, "backward")
        node.specification = wrapper
        node.op = "normalization.activation_backward"
        graph.nodes.remove(derivative)
        _evidence(candidate, spec, "backward", wrapper)
        candidate.update(eligible=True, selected=True, status="selected", reason="selected",
                         removed_activation_gradient_bytes=prod(gradient.meta.shape) * torch.empty((), dtype=gradient.meta.dtype).element_size(),
                         public_outputs=tuple(value.name for value in node.values))
        _select(node, candidate, spec, "backward")
        patterns.append(candidate["pattern"])
        rebuild(graph)
    return tuple(patterns)
