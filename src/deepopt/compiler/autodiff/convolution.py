"""FP8 convolution adjoints expressed as operand preparation, dX and dW IR."""

from dataclasses import replace
from contextlib import nullcontext

import torch

from ..ir import AuxiliaryTensorMeta, CallNode, InputNode, TensorValue
from .spatial import Convolution
from .gradient import GradientPolicy
from .semantic import emit, recorded
from .backward_rules import GradWrapper


class FP8Convolution(Convolution):
    op = "autodiff_fp8_convolution"
    state_names = ("x_payload", "x_scale", "weight_payload", "weight_scale", "codec_step", "codec_minimum", "gradient_selector")

    def __init__(self, node, context):
        super().__init__(node, context)
        from deepopt.opt.backends import convolution, convolution_transpose
        from deepopt.opt.backends._shared.activation_codec import spatial_row_alignment

        self.executor = convolution_transpose if self.transposed else convolution
        self.options = self.plan["fp8_training"]
        self.blocks, self.dtypes = dict(self.plan["block_shapes"]), dict(self.options["fp8_dtypes"])
        self.pointwise = (not self.transposed and self.geometry.pointwise and all(block is None for block in self.blocks.values())
                          and (self.metas[0].storage_dtype is None or self.options.get("backward_precision") == "auto"))
        if self.metas[0].storage_dtype is not None and not self.explicit_padding and not any(self.synthetic_right):
            self.blocks["activation"] = self.metas[0].block_shape
            self.dtypes["activation"] = self.metas[0].storage_dtype
        self.compute_dtype = node.values[0].meta.dtype
        self.state_training = self.module.training and torch.is_grad_enabled()
        self.scaling_state = getattr(self.module, "fp8_scaling", None) if self.options["scaling_recipe"] == "delayed" else None
        self.row_alignment = (1 if self.transposed else spatial_row_alignment(self.geometry, self.dtypes["activation"],
            self.blocks["activation"], {**self.compression, "enabled": self.compressed},
            torch.device(self.metas[0].device), delayed=self.scaling_state is not None))
        self.policy = GradientPolicy(self.module, node.root or self.module,
            not isinstance(node.args[0].producer, InputNode), context.callsite_occurrence)
        self.empty_selector = torch.tensor(-1, device="cpu", dtype=torch.int64)
        self.output_fp8 = node.values[0].meta.storage_dtype is not None
        self.output_slot = getattr(self.module, "output_scaling", None) if self.output_fp8 else None

    def lower(self, node):
        states = tuple(TensorValue(f"{node.node_name}__{name}", AuxiliaryTensorMeta()) for name in self.state_names)
        return CallNode(self.op, recorded(self, self.forward, "forward"), node.args, {}, (*node.values, *states), self)

    def route(self, backward=False):
        return (self.options["backward_backend"] if backward and self.options["backward_backend"] != "inherit"
                else self.options["backend"])

    def arguments(self, backward=False):
        options = dict(backend=self.route(backward), num_stages=self.options["num_stages"], autotune=self.options["autotune"])
        if not self.transposed:
            options["kernel_variant"] = self.options.get("convolution_policy", "strict")
        return options

    def forward(self, x, weight, bias):
        from deepopt.opt.scaling.delayed import fp8_scaling_scope

        with fp8_scaling_scope() if self.scaling_state is not None else nullcontext():
            return self._forward(x, weight, bias)

    def _forward(self, x, weight, bias):
        from deepopt.opt.nn.convolution import source_matrix, restore_matrix
        from deepopt.opt.quantization.activation_tensor import FP8Activation
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from deepopt.opt.quantization.activation import quantize_pack_activation
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
        from deepopt.opt.scaling.delayed import checkpoint_recomputing

        selector = self.policy.select(x)
        source = self.source(x)
        supplied = source.quantized("channels") if isinstance(source, FP8Activation) else None
        matrix = None if supplied is not None else source_matrix(source)
        packed = None
        if supplied is not None:
            qx = supplied
            if self.compressed:
                packed = quantize_pack_activation(qx.data, self.compression["group_size"], qx.block_shape, bits=self.compression["bits"])
        elif self.scaling_state is not None:
            if self.compressed:
                qx, packed = self.scaling_state.quantize_pack_activation(matrix, group_size=self.compression["group_size"],
                    bits=self.compression["bits"], training=self.state_training, row_alignment=self.row_alignment,
                    autotune=self.options["autotune"])
            else:
                qx = self.scaling_state.quantize(matrix, "activation", training=self.state_training,
                                                autotune=self.options["autotune"])[0]
        elif self.compressed:
            qx, packed = quantize_pack_fp8_activation(matrix, fp8_dtype=self.dtypes["activation"],
                block_shape=self.blocks["activation"], group_size=self.compression["group_size"], bits=self.compression["bits"],
                row_alignment=self.row_alignment)
        else:
            qx = quantize_fp8(matrix, fp8_dtype=self.dtypes["activation"], block_shape=self.blocks["activation"])
        def quantizer(value, *, fp8_dtype, block_shape):
            values = value.reshape(self.geometry.weight_matrix_shape)
            return (self.scaling_state.quantize(values, "weight", training=self.state_training,
                    autotune=self.options["autotune"])[0] if self.scaling_state is not None else
                    quantize_fp8(values, fp8_dtype=fp8_dtype, block_shape=block_shape))
        cache = getattr(self.module, "_fp8_weight_cache", None)
        recipe = None
        if self.scaling_state is not None:
            recipe = (self.scaling_state.cache_token, self.state_training)
            if checkpoint_recomputing():
                cache = None
        qw = (cache.quantize(weight, fp8_dtype=self.dtypes["weight"], block_shape=self.blocks["weight"],
            quantizer=quantizer, recipe=recipe) if cache is not None else
            quantizer(weight, fp8_dtype=self.dtypes["weight"], block_shape=self.blocks["weight"]))
        scope = (self.output_slot.output(self.geometry.output_shape, self.compute_dtype, training=self.state_training)
                 if self.output_slot is not None else nullcontext(None))
        with scope as planned_output:
            if self.pointwise:
                from deepopt.opt.backends import gemm_nt

                output = gemm_nt(qx, qw, backend=self.route(), bias=bias, out_dtype=self.compute_dtype,
                    num_stages=self.options["num_stages"], autotune=self.options["autotune"], diagnostic_role="forward")
            else:
                output = self.executor.forward(qx, qw, self.geometry, bias=bias,
                out_dtype=torch.float32 if planned_output is not None else self.compute_dtype,
                **({"output": planned_output} if self.transposed else {}), **self.arguments())
        if not isinstance(output, FP8Activation):
            output = restore_matrix(output, self.geometry.output_shape, self.input_format)
        empty = x.new_empty((0,))
        saved = ((packed.data if packed is not None else qx.data) if self.needs[1] else empty,
                 qx.scale_inv if self.needs[1] else empty, qw.data if self.needs[0] else empty,
                 qw.scale_inv if self.needs[0] else empty, packed.step if packed is not None else empty,
                 packed.minimum if packed is not None else empty, selector if selector is not None else self.empty_selector)
        self.module.last_execution = {"forward": "fp8", "backward": "native_autodiff", "compression": self.compressed,
            "saved_state_bytes": sum(value.numel() * value.element_size() for value in saved)}
        return output.squeeze(0) if self.unbatched else output, *saved

    def prepare_gradient(self, gradient, selector):
        from deepopt.opt.quantization.gradient import gradient_operand, gradient_matrix, gradient_bias
        from deepopt.opt.quantization.quantizer import quantize_fp8

        gradient = gradient.unsqueeze(0) if self.unbatched else gradient
        supplied = gradient_operand(gradient, "channels")
        matrix = None if supplied is not None else gradient_matrix(gradient, "channels")
        self.policy.record(selector, reused=supplied is not None)
        qg = None
        if self.needs[0] or self.needs[1]:
            qg = (supplied if supplied is not None else self.scaling_state.quantize(matrix, "grad", bias=False,
                training=self.state_training, check_finite=self.gradient_check, autotune=self.options["autotune"])[0]
                if self.scaling_state is not None else quantize_fp8(matrix, fp8_dtype=self.dtypes["grad"],
                    block_shape=self.blocks["grad"], check_finite=self.gradient_check))
        empty = gradient.new_empty((0,))
        reduction = (matrix.sum(0, dtype=torch.float32) if self.pointwise else
                     gradient.sum((0, *range(2, gradient.ndim)), dtype=torch.float32)) if self.needs[2] and supplied is None else None
        db = ((gradient_bias(supplied) if supplied is not None else reduction).to(self.metas[2].dtype)
              if self.needs[2] else empty)
        descriptor = torch.tensor(qg.block_shape or (-1, -1), device="cpu", dtype=torch.int64) if qg is not None else self.empty_selector
        return qg.data if qg is not None else empty, qg.scale_inv if qg is not None else empty, descriptor, db

    @staticmethod
    def gradient_operand(data, inverse, descriptor):
        from deepopt.opt.quantization.tensor import QuantizedTensor

        block = tuple(descriptor.tolist())
        return QuantizedTensor(data, inverse, None if block[0] < 0 else block)

    def dx(self, data, inverse, descriptor, wdata, winverse, selector):
        from deepopt.opt.quantization.tensor import QuantizedTensor
        from deepopt.opt.nn.convolution import restore_matrix

        qg = self.gradient_operand(data, inverse, descriptor)
        qw = QuantizedTensor(wdata, winverse, self.blocks["weight"])
        slot = self.policy.slot(selector)
        if self.pointwise:
            from deepopt.opt.backends import gemm_nt

            if slot is not None:
                with slot.output(self.geometry.input_shape, self.metas[0].dtype, training=self.state_training,
                                 check_finite=self.gradient_check) as output:
                    value = gemm_nt(qg, qw.transpose(contiguous=False), backend=self.route(True), out_dtype=torch.float32,
                        num_stages=self.options["num_stages"], autotune=self.options["autotune"], gradient_output=output, diagnostic_role="dx")
                    return self.restore_input_gradient(value)
            values = gemm_nt(qg, qw.transpose(contiguous=False), backend=self.route(True), out_dtype=self.metas[0].dtype,
                num_stages=self.options["num_stages"], autotune=self.options["autotune"], diagnostic_role="dx")
            dx = restore_matrix(values, self.geometry.input_shape, self.input_format)
            return self.restore_input_gradient(dx)
        if slot is not None:
            with slot.output(self.geometry.input_shape, self.metas[0].dtype, training=self.state_training,
                             check_finite=self.gradient_check) as output:
                value = self.executor.input_gradient(qg, qw, self.geometry, out_dtype=torch.float32,
                    gradient_output=output, **self.arguments(True))
                return self.restore_input_gradient(value)
        values = self.executor.input_gradient(qg, qw, self.geometry, out_dtype=self.metas[0].dtype, **self.arguments(True))
        dx = restore_matrix(values, self.geometry.input_shape, self.input_format)
        return self.restore_input_gradient(dx)

    def dw(self, data, inverse, descriptor, xdata, xinverse, step, minimum):
        from deepopt.opt.quantization.tensor import QuantizedTensor
        from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

        qg = self.gradient_operand(data, inverse, descriptor)
        if self.compressed:
            packed = PackedActivation(xdata, step, minimum, self.geometry.input_matrix_shape,
                self.compression["group_size"], self.blocks["activation"], self.compression["bits"])
            xdata = dequantize_unpack_activation(packed, dtype=self.dtypes["activation"], row_alignment=self.row_alignment)
        qx = QuantizedTensor(xdata, xinverse, self.blocks["activation"])
        if self.pointwise:
            from deepopt.opt.backends import gemm_nt

            value = gemm_nt(qg.transpose(contiguous=False), qx.transpose(contiguous=False), backend=self.route(True),
                out_dtype=self.compute_dtype, num_stages=self.options["num_stages"], autotune=self.options["autotune"], diagnostic_role="dw")
        else:
            value = self.executor.weight_gradient(qx, qg, self.geometry, out_dtype=self.compute_dtype, **self.arguments(True))
        return value.reshape(self.geometry.weight_shape).to(self.metas[1].dtype).contiguous(memory_format=self.weight_format)


def vjp(ctx, node, gradients):
    spec = node.specification
    saved = [ctx.saved(value) for value in node.values[spec.public_count:]]
    xdata, xscale, wdata, wscale, step, minimum, selector = saved
    prepared = tuple(TensorValue(f"{node.values[0].name}_gradient_{index}", AuxiliaryTensorMeta()) for index in range(4))
    ctx.builder.graph.add(CallNode("convolution.prepare_gradient", recorded(spec, spec.prepare_gradient, "backward"),
        (gradients[0].value, selector.value), {}, prepared))
    data, scale, descriptor, bias = prepared
    contributions = []
    if spec.needs[0]:
        metadata = replace(spec.metas[0], representation="opaque") if spec.policy.enabled else spec.metas[0]
        contributions.append(GradWrapper(node.args[0], emit(ctx, "convolution.fp8_dx", spec.dx,
            (data, scale, descriptor, wdata, wscale, selector), metadata)))
    if spec.needs[1]:
        contributions.append(GradWrapper(node.args[1], emit(ctx, "convolution.fp8_dw", spec.dw,
            (data, scale, descriptor, xdata, xscale, step, minimum), spec.metas[1])))
    if spec.needs[2]:
        bias.meta = replace(spec.metas[2], requires_grad=False)
        contributions.append(GradWrapper(node.args[2], bias))
    return contributions


def register_rules(registry):
    if not registry.has_backward_rule(FP8Convolution.op):
        registry.register_backward_rule(FP8Convolution.op, vjp, multi_output=True)
