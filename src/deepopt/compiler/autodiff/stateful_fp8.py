"""Native VJPs for prepared FP8 activation/LayerNorm representation edges."""

from dataclasses import replace

import torch
import os

from ..ir import AuxiliaryTensorMeta, CallNode, InputNode, TensorValue
from .backward_rules import GradWrapper
from .gradient import GradientPolicy
from .semantic import result, recorded
from deepopt.opt.backends._shared.activation_codec import saved_activation_layout
from deepopt.opt.backends._shared.normalization import saved_normalization_variant


class ForwardResiduals:
    """Transient adapter for shared numerical forward routines, never cached."""

    def __init__(self, needs):
        self.needs_input_grad = needs
        self.tensors = ()

    def save_for_backward(self, *tensors):
        self.tensors = tensors


class StatefulFP8:
    op = "autodiff_fp8_stateful"

    def __init__(self, node, context):
        from deepopt.kernel_policy import resolve_kernel_policy
        from deepopt.opt.nn.activation.geometry import ReLUGeometry
        from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
        from deepopt.opt._execution import backward_finite_check_policy
        from deepopt.opt.backends._shared.activation_codec import activation_pointwise_tile
        from deepopt.opt.backends._shared.activation_codec import activation_layout_policy
        from deepopt.opt.backends._shared.normalization import normalization_variant, normalization_tuning

        self.module = node.module
        self.module_path, self.node_name = node.module_path, node.node_name
        self.precision = "fp8_stateful"
        self.profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"
        self.plan = node.plan if node.plan is not None else self.module.plan
        self.recipe = self.plan.get("fp8_training", self.plan)
        self.input_meta = node.args[0].meta
        self.metas = tuple(value.meta if value is not None else None for value in node.args)
        self.needs = tuple(value is not None and value.meta.requires_grad for value in node.args)
        self.training = node.value.meta.requires_grad
        self.needs = tuple(self.training and needed for needed in self.needs)
        self.module_training = self.module.training
        self.state_training = self.module_training and torch.is_grad_enabled()
        self.gradient_check = backward_finite_check_policy()
        self.pointwise_tile = activation_pointwise_tile()
        self.activation_kernel_policy = activation_layout_policy()
        self.normalization_variant, self.normalization_tuning = normalization_variant(), normalization_tuning()
        self.norm = node.op == "native_layer_norm"
        self.family = "normalization" if self.norm else "activation"
        self.kind = getattr(self.module, "kind", "relu")
        self.output_dtype = node.value.meta.storage_dtype or node.value.meta.dtype
        self.logical_dtype = node.value.meta.dtype
        self.layout = node.value.meta.layout or "last"
        self.blocks = self.plan.get("block_shapes", {}).get("activation")
        self.compression = self.recipe["activation_compression"]
        norm_options = node.options or ({"normalized_shape": self.module.normalized_shape, "eps": self.module.eps} if self.norm else {})
        self.geometry = (LayerNormGeometry(self.input_meta.shape, tuple(norm_options["normalized_shape"]), norm_options["eps"], self.layout)
                         if self.norm else ReLUGeometry(self.input_meta.shape, self.layout))
        self.kernel_policy = resolve_kernel_policy(context.kernel_policy)
        self.backend = self.recipe.get("backend", self.kernel_policy.backend if torch.device(self.input_meta.device).type == "cuda" else "reference")
        self.gradient_policy = GradientPolicy(self.module, node.root or self.module,
            not isinstance(node.args[0].producer, InputNode), context.callsite_occurrence)
        self.scaling_state = (getattr(self.module, "fp8_scaling", None)
                              if self.output_dtype in
                              (torch.float8_e4m3fn, torch.float8_e5m2) else None)
        self.delayed = self.scaling_state is not None
        self.state_names = (("original_or_xhat", "source_scale_or_step", "minimum", "inverse", "mean", "weight", "center", "gain")
                            if self.norm else ("coefficient", "step", "minimum", "coefficient_metadata")) + ("gradient_selector",)
        self.empty_selector = torch.tensor(-1, device="cpu", dtype=torch.int64)

    def lower(self, node):
        states = tuple(TensorValue(f"{node.value.name}__{name}", AuxiliaryTensorMeta()) for name in self.state_names)
        return CallNode(self.op, recorded(self, self.forward, "forward"), node.args, {}, (node.value, *states), self)

    def forward(self, x, weight, bias):
        from deepopt.opt._execution import finite_checks_enabled
        from deepopt.opt.nn.activation.state import SavedActivation

        selector = self.gradient_policy.select(x)
        selector = selector if selector is not None else self.empty_selector
        ctx = ForwardResiduals(self.needs)
        self.module.last_execution = info = {}
        empty = x.new_empty((0,))
        if self.norm:
            from deepopt.opt.nn.autograd.normalization.layer import layer_norm_forward

            options = dict(dtype=self.output_dtype, logical_dtype=self.logical_dtype, block_shape=self.blocks,
                backward_enabled=self.training, compression=self.compression, scale=None,
                backend=self.backend,
                check_finite=finite_checks_enabled() and (self.compression["enabled"] or self.output_dtype in
                                                         (torch.float8_e4m3fn, torch.float8_e5m2)))
            output = layer_norm_forward(ctx, x, weight, bias, self.geometry, options, self.scaling_state,
                                        self.state_training, info)
            return (output, *(value if value is not None else empty for value in ctx.tensors), selector)
        from deepopt.opt.quantization.activation_tensor import FP8Activation

        wrapped = isinstance(x, FP8Activation)
        fp8 = self.output_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        reuse = (self.kind == "relu" and wrapped and fp8 and x._data.dtype == self.output_dtype
                 and x._block_shape == self.blocks and (self.layout == x._activation_layout or self.blocks is None))
        options = dict(dtype=self.output_dtype, block_shape=self.blocks, packed=self.compression["enabled"],
            inplace=False, reuse=reuse, scale=None, check_finite=finite_checks_enabled() and (fp8 or self.compression["enabled"]))
        if self.kind == "relu":
            from deepopt.opt.nn.autograd.activation.relu import ReLUFunction

            output = ReLUFunction.forward(ctx, x, self.geometry, options, self.scaling_state,
                                          self.state_training, self.needs[0], info)
            saved = SavedActivation(ctx.tensors[0], mask=True, packed=self.compression["enabled"]) if self.needs[0] else None
        else:
            from deepopt.opt.nn.autograd.activation.elementwise import activation_forward

            options.update(kind=self.kind, approximate=self.module.approximate,
                           bits=self.compression["bits"], group_size=self.compression["group_size"])
            output = activation_forward(ctx, x, self.geometry, options, self.scaling_state,
                                        self.state_training, self.needs[0], info)
            saved = SavedActivation.restore(ctx.tensors, ctx.saved_metadata) if self.needs[0] else None
        metadata = (saved.metadata if saved is not None else (512, None, 2, False, False, None))
        group, block, bits, mask, packed, recompute = metadata
        header = torch.tensor((group, *(block or (-1, -1)), bits, int(mask), int(packed),
                               int(recompute is not None), int(recompute is not None and recompute[1] == "tanh")),
                              device="cpu", dtype=torch.int64)
        tensors = list(saved.tensors) if saved is not None else []
        tensors += [empty] * (3 - len(tensors))
        return output, *tensors, header, selector

    @saved_activation_layout
    def activation_gradient(self, gradient, data, step, minimum, metadata, selector):
        from deepopt.opt.nn.activation.state import SavedActivation
        from deepopt.opt.backends.activation import executor
        from deepopt.opt.quantization.activation_tensor import FP8Activation

        group, br, bc, bits, mask, packed, recompute, tanh = metadata.tolist()
        saved = SavedActivation(data, step if step.numel() else None, minimum if minimum.numel() else None,
            group, None if br < 0 else (br, bc), bits, bool(mask), bool(packed),
            (self.kind, "tanh" if tanh else "none") if recompute else None)
        slot = self.gradient_policy.slot(selector)
        compute = executor(gradient) if self.kind == "relu" else executor(gradient, self.kind)
        if slot is not None:
            with slot.output(self.input_meta.shape, self.input_meta.dtype, training=self.state_training,
                             check_finite=self.gradient_check) as output:
                return (compute.backward(gradient, data, self.geometry, packed=bool(packed),
                        dtype=self.input_meta.dtype, gradient_output=output) if self.kind == "relu" else
                        compute.backward(gradient, saved, self.geometry, dtype=self.input_meta.dtype, gradient_output=output,
                                         pointwise_tile=self.pointwise_tile))
        reuse = (self.kind == "relu" and not self.gradient_policy.paths and self.input_meta.storage_dtype is not None
                 and isinstance(gradient, FP8Activation)
                 and (gradient._activation_layout == self.layout or gradient._block_shape is None))
        if self.kind == "relu":
            values = compute.backward(gradient, data, self.geometry, packed=bool(packed),
                dtype=gradient._data.dtype if reuse else self.input_meta.dtype, reuse=reuse)
            return (FP8Activation(values, gradient._scale_inv, self.input_meta.shape, gradient._block_shape,
                                  self.layout, self.input_meta.dtype) if reuse else self.geometry.restore(values))
        return self.geometry.restore(compute.backward(gradient, saved, self.geometry, dtype=self.input_meta.dtype,
                                                       pointwise_tile=self.pointwise_tile))

    @saved_normalization_variant
    def norm_gradients(self, gradient, *state):
        from deepopt.opt.nn.autograd.normalization.layer import executor
        from deepopt.opt.quantization.activation import PackedActivation
        from deepopt.opt.quantization.activation_tensor import FP8Activation

        data, step, minimum, inverse, mean, weight, center, gain, selector = state
        packed = original = None
        if minimum.numel():
            group = min(self.compression["group_size"], self.geometry.elements)
            packed = PackedActivation(data, step, minimum, self.geometry.matrix_shape, group, (1, group), self.compression["bits"])
        elif data.numel():
            original = (FP8Activation(data, step, self.input_meta.shape, self.input_meta.block_shape,
                self.input_meta.layout, self.input_meta.dtype, self.input_meta.stride)
                if self.input_meta.storage_dtype is not None else data)
        arguments = dict(packed=packed, x=original, mean=mean if mean.numel() else None,
            inverse=inverse if inverse.numel() else None, center_shift=center if center.numel() else None,
            moment_gain=gain if gain.numel() else None, need_dx=self.needs[0], need_dw=self.needs[1], need_db=self.needs[2],
            input_dtype=self.input_meta.dtype, execution_info=self.module.last_execution)
        compute = executor(gradient, self.backend)
        slot = self.gradient_policy.slot(selector)
        if self.needs[0] and slot is not None:
            with slot.output(self.input_meta.shape, self.input_meta.dtype, training=self.state_training,
                             check_finite=self.gradient_check) as output:
                values = compute.backward(gradient, weight if weight.numel() else None, self.geometry, gradient_output=output, **arguments)
        else:
            values = compute.backward(gradient, weight if weight.numel() else None, self.geometry, **arguments)
        return result([value if isinstance(value, FP8Activation) else value.to(metadata.dtype)
                       for value, metadata, needed in zip(values, self.metas, self.needs) if needed])


def vjp(ctx, node, gradients):
    spec = node.specification
    targets = tuple(value for value, needed in zip(node.args, spec.needs) if needed)
    if not targets:
        return ()
    values = tuple(TensorValue(f"{node.values[0].name}_grad_{index}", replace(value.meta, requires_grad=False,
        representation="opaque" if index == 0 and spec.gradient_policy.enabled else value.meta.representation))
        for index, value in enumerate(targets))
    operands = (gradients[0].value, *(ctx.saved(value).value for value in node.values[1:]))
    ctx.builder.graph.add(CallNode("normalization.fp8_vjp" if spec.norm else "activation.fp8_vjp",
        recorded(spec, spec.norm_gradients if spec.norm else spec.activation_gradient, "backward"), operands, {}, values))
    return tuple(GradWrapper(target, value) for target, value in zip(targets, values))


def register_rules(registry):
    if not registry.has_backward_rule(StatefulFP8.op):
        registry.register_backward_rule(StatefulFP8.op, vjp, multi_output=True)
