"""Native FP8 Linear VJP: explicit operand preparation and GEMM dependencies."""

from dataclasses import replace
import os

import torch

from ..ir import AuxiliaryTensorMeta, CallNode, InputNode, TensorValue
from .backward_rules import GradWrapper
from .semantic import meta, packed_metas, call_node, result, emit, recorded


def physical(shape, dtype, device):
    # Backend quantizers may pad row strides. Numerical GEMM descriptors own
    # those layouts; no pointwise emitter may assume a contiguous FP8 payload.
    return replace(meta(shape, dtype, device, representation="fp8_storage"), stride=None)


def scale_meta(shape, block, device):
    grid = (1, 1) if block is None else tuple((size + tile - 1) // tile for size, tile in zip(shape, block))
    return physical(grid, torch.float32, device)


class FP8Linear:
    family = "linear"
    op = "autodiff_fp8_linear"
    precision = "fp8"
    execution = "existing_fp8_kernels"
    generated_kernels = False
    reuses_existing_kernels = True
    activation = activation_state = epilogue = decode = None
    residual = False
    profile_ranges = True

    def __init__(self, node, context):
        from math import prod
        from deepopt.compiler.linear import linear_module
        from deepopt.kernel_policy import plan_kernel_policy, resolve_kernel_policy
        from deepopt.opt.backends._shared.gemm_policy import linear_backward_policy, gradient_reduction_policy
        from deepopt.opt._execution import backward_finite_check_policy

        self.module, self.plan = node.module or linear_module(node.target), node.plan
        self.profile_ranges = os.environ.get("DEEPOPT_PROFILE_RANGES") == "1"
        self.module_path, self.node_name = node.module_path, node.value.name
        self.module_training, self.training = self.module.training, node.value.meta.requires_grad
        self.state_training = self.module_training and torch.is_grad_enabled()
        self.kernel_policy = resolve_kernel_policy(context.kernel_policy)
        if self.plan["fp8_training"]["backend"] != "reference" and plan_kernel_policy(self.plan) != self.kernel_policy:
            raise ValueError("Compiler kernel policy conflicts with the prepared Linear policy")
        self.kernel_policy_fingerprint = self.kernel_policy.fingerprint
        self.kernel_routes = {role: self.kernel_policy.route("gemm", f"linear.{role}") for role in ("forward", "dx", "dw", "db")}
        x, weight, bias = node.args
        self.input_meta, self.weight_meta, self.output_meta = x.meta, weight.meta, node.value.meta
        from .gradient import GradientPolicy

        root = getattr(node.target, "root", self.module)
        self.gradient_policy = GradientPolicy(self.module, root, not isinstance(x.producer, InputNode), context.callsite_occurrence)
        self.bias_meta = bias.meta if bias is not None else None
        self.matrix_shape = (prod(x.meta.shape[:-1]), x.meta.shape[-1])
        self.need_dx, self.need_dw = self.training and x.meta.requires_grad, self.training and weight.meta.requires_grad
        self.need_db = self.training and bias is not None and bias.meta.requires_grad
        self.options, self.blocks = self.plan["fp8_training"], dict(self.plan["block_shapes"])
        self.dtypes = dict(self.options["fp8_dtypes"])
        self.supplied_input = self.input_meta.storage_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        if self.supplied_input:
            self.blocks["activation"] = self.input_meta.block_shape
            self.dtypes["activation"] = self.input_meta.storage_dtype
        self.compression = self.options["activation_compression"]
        self.compressed = self.need_dw and self.compression["enabled"]
        self.delayed = self.options["scaling_recipe"] == "delayed"
        self.scaling_state = self.module.fp8_scaling if self.delayed else None
        self.paired = self.delayed or self.options["backend"] != "reference" and torch.device(x.meta.device).type == "cuda"
        self.pair_x = self.paired and not self.supplied_input and self.need_dw and not self.compressed and self.blocks["activation"] is None and (self.delayed or prod(x.meta.shape) > 4096)
        self.pair_w = self.paired and self.need_dx and self.blocks["weight"] is None and (self.delayed or prod(weight.meta.shape) > 4096)
        self.pair_g = self.paired and self.need_dw and self.blocks["grad"] is None and (self.delayed or prod(node.value.meta.shape) > 4096)
        block = self.blocks["activation"]
        self.saved_transpose = not self.supplied_input and self.need_dw and block is not None and block[0] == 1 and block[1] > 1
        self.saved_shape = tuple(reversed(self.matrix_shape)) if self.saved_transpose else self.matrix_shape
        block = self.blocks["grad"]
        self.grad_transpose = self.need_dw and block is not None and block[0] == 1 and block[1] > 1
        self.dw_backend, self.native_layout = linear_backward_policy()
        self.gradient_reduction = gradient_reduction_policy()
        self.check_finite = backward_finite_check_policy()
        self.saved_components = tuple((['x_payload', 'x_scale_inv'] if self.need_dw else [])
            + (['weight_payload', 'weight_scale_inv'] if self.need_dx else [])
            + (['codec_step', 'codec_minimum'] if self.compressed else [])
            + (['gradient_selector'] if self.gradient_policy.paths and (self.need_dx or self.need_dw) else []))
        self.forward_steps = ("quantize_pack" if self.compressed else "quantize", "cached_weight", "fp8_gemm")
        self.backward_steps = tuple((["quantize_gradient"] if self.need_dx or self.need_dw else [])
            + (["fp8_gemm_dx"] if self.need_dx else []) + (["fp8_gemm_dw"] if self.need_dw else [])
            + (["bias_reduce"] if self.need_db else []))
        self.source = "DeepOpt native AutoDiff FP8 VJP"

    def inference(self, x, weight, bias):
        from deepopt.opt.scaling.delayed import fp8_scaling_scope
        from contextlib import nullcontext

        with fp8_scaling_scope() if self.delayed else nullcontext():
            values = self.forward(x, weight, bias)
        return values[0] if self.saved_components else values

    __call__ = inference

    def lower(self, node):
        m, k = self.matrix_shape
        n = self.weight_meta.shape[0]
        device, dtypes = self.input_meta.device, self.dtypes
        state = []
        if self.need_dw:
            payload = (packed_metas(self.saved_shape, self.compression, device, self.blocks["activation"])[0] if self.compressed else
                       physical((k, m) if self.pair_x or self.saved_transpose else (m, k), dtypes["activation"], device))
            state += [payload, scale_meta(self.saved_shape, self.blocks["activation"], device)]
        if self.need_dx:
            state += [physical((k, n) if self.pair_w else (n, k), dtypes["weight"], device),
                      scale_meta((n, k), self.blocks["weight"], device)]
        if self.compressed:
            state += list(packed_metas(self.saved_shape, self.compression, device, self.blocks["activation"])[1:])
        if self.gradient_policy.paths and (self.need_dx or self.need_dw):
            state += [meta((), torch.int64, "cpu")]
        return call_node(node, self, state)

    def route(self, role):
        if self.options["backend"] == "reference":
            return "reference"
        backward = self.options["backward_backend"]
        if role != "forward" and backward in ("native_fast", "native_precise"):
            return backward
        return self.kernel_policy.route("gemm", f"linear.{role}")

    def call(self, role, function, *args, **kwargs):
        from deepopt.kernel_policy import kernel_policy_scope

        with kernel_policy_scope(self.kernel_policy, role=role):
            return function(*args, **kwargs)

    def forward(self, x, weight, bias):
        from contextlib import nullcontext
        from deepopt.opt.scaling.delayed import fp8_scaling_scope

        with fp8_scaling_scope() if self.delayed else nullcontext():
            return self._forward(x, weight, bias)

    def _forward(self, x, weight, bias):
        from deepopt.opt.quantization.quantizer import quantize_fp8, _quantize_tensorwise_pair
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
        from deepopt.opt.backends import gemm_nt
        from deepopt.opt.backends._shared.gemm_policy import linear_forward_backend
        from deepopt.opt.nn.autograd._output import restore_fresh_output

        selector = self.gradient_policy.select(x)
        matrix = None if self.supplied_input else x.reshape(self.matrix_shape)
        dtypes = self.dtypes
        packed = qxt = None
        if self.supplied_input:
            qx = x.quantized("last")
            if self.compressed:
                from deepopt.opt.quantization.activation import quantize_pack_activation

                packed = x.linear_packed(group_size=self.compression["group_size"], block_shape=qx.block_shape,
                    bits=self.compression["bits"])
                if packed is None:
                    packed = self.call("linear.prepare_input", quantize_pack_activation, qx.data,
                        self.compression["group_size"], qx.block_shape, bits=self.compression["bits"])
        elif self.activation is not None:
            from ..activation_quantize import prepare

            qx, qxt, packed = self.call("compiler.preparation", prepare, matrix, activation=self.activation,
                fp8_dtype=dtypes["activation"], paired=self.pair_x,
                compression=self.compression if self.compressed else None)
        elif self.compressed and self.delayed:
            qx, packed = self.call("linear.prepare_input", self.scaling_state.quantize_pack_activation,
                matrix, group_size=self.compression["group_size"], bits=self.compression["bits"], training=self.state_training,
                autotune=self.options["autotune"])
        elif self.compressed and not self.saved_transpose:
            qx, packed = self.call("linear.prepare_input", quantize_pack_fp8_activation, matrix,
                fp8_dtype=dtypes["activation"], block_shape=self.blocks["activation"],
                group_size=self.compression["group_size"], bits=self.compression["bits"])
        elif self.delayed:
            qx, qxt, _ = self.call("linear.prepare_input", self.scaling_state.quantize, matrix, "activation",
                transposed=self.pair_x, training=self.state_training, autotune=self.options["autotune"])
        elif self.pair_x:
            qx, qxt = self.call("linear.prepare_input", _quantize_tensorwise_pair, matrix, fp8_dtype=dtypes["activation"])
        else:
            qx = self.call("linear.prepare_input", quantize_fp8, matrix, fp8_dtype=dtypes["activation"], block_shape=self.blocks["activation"])
        quantizer = _quantize_tensorwise_pair if self.pair_w else quantize_fp8
        recipe = (None if self.kernel_policy.backend == "triton" else
                  (("compiler-linear", self.kernel_policy_fingerprint), "current"))
        cache = self.module._fp8_weight_cache
        if self.delayed:
            from deepopt.opt.scaling.delayed import checkpoint_recomputing

            def quantizer(value, **kwargs):
                normal, transposed, _ = self.scaling_state.quantize(value, "weight", transposed=self.pair_w,
                    training=self.state_training, autotune=self.options["autotune"])
                return (normal, transposed) if self.pair_w else normal
            recipe = (self.scaling_state.cache_token, self.state_training)
            if checkpoint_recomputing():
                cache = None
        quantize_weight = lambda *args, **kwargs: self.call("linear.prepare_weight", quantizer, *args, **kwargs)
        cached = (cache.quantize(weight, fp8_dtype=dtypes["weight"], block_shape=self.blocks["weight"],
            paired=self.pair_w, quantizer=quantize_weight, recipe=recipe) if cache is not None else
            quantize_weight(weight, fp8_dtype=dtypes["weight"], block_shape=self.blocks["weight"]))
        qw, qwt = cached if self.pair_w else (cached, None)
        override = linear_forward_backend()
        selected = override if override != "inherit" and self.options["backend"] != "reference" else self.route("forward")
        output = self.call("linear.forward", gemm_nt, qx, qw, backend=selected, bias=bias,
            out_dtype=self.output_meta.dtype, num_stages=self.options["num_stages"],
            autotune=self.options["autotune"], diagnostic_role="forward")
        if self.saved_transpose:
            if self.compressed:
                qx, packed = self.call("linear.prepare_input", quantize_pack_fp8_activation, matrix.T,
                    fp8_dtype=dtypes["activation"], block_shape=self.blocks["activation"],
                    group_size=self.compression["group_size"], bits=self.compression["bits"])
            else:
                qx = self.call("linear.prepare_input", quantize_fp8, matrix.T,
                    fp8_dtype=dtypes["activation"], block_shape=self.blocks["activation"])
        saved = []
        if self.need_dw:
            saved += [packed.data if packed is not None else qxt.data if qxt is not None else qx.data, qx.scale_inv]
        if self.need_dx:
            saved += [qwt.data if qwt is not None else qw.data, qwt.scale_inv if qwt is not None else qw.scale_inv]
        if packed is not None:
            saved += [packed.step, packed.minimum]
        if self.gradient_policy.paths and (self.need_dx or self.need_dw):
            saved += [selector]
        return result([restore_fresh_output(output, lambda t: t.reshape(*x.shape[:-1], weight.shape[0])), *saved])

    def gradient_schema(self):
        m, _ = self.matrix_shape
        n = self.weight_meta.shape[0]
        values = []
        if self.need_dx:
            values += [("qg", AuxiliaryTensorMeta(shape=(m, n), representation="fp8_storage")),
                       ("qg_scale", AuxiliaryTensorMeta(dtype=torch.float32, representation="fp8_storage"))]
        if self.need_dw:
            values += [("qgt", AuxiliaryTensorMeta(shape=(n, m), representation="fp8_storage")),
                       ("qgt_scale", AuxiliaryTensorMeta(dtype=torch.float32, representation="fp8_storage"))]
        if self.need_dx or self.need_dw:
            values += [("gradient_blocks", meta((4,), torch.int64, "cpu"))]
        if self.need_db:
            values += [("db", meta((n,), self.bias_meta.dtype, self.input_meta.device))]
        return values

    def prepare_gradient(self, gradient, selector=None):
        from deepopt.opt.quantization.quantizer import (quantize_fp8, _quantize_tensorwise_pair,
            _quantize_tensorwise_transpose, _quantize_gradient_pair_and_bias)
        from deepopt.opt.quantization.gradient import gradient_operand, gradient_bias, gradient_transpose

        supplied = gradient_operand(gradient, "last")
        matrix = None if supplied is not None else gradient.reshape(-1, self.weight_meta.shape[0])
        qg = qgt = db = None
        dtype = self.dtypes["grad"]
        kwargs = dict(fp8_dtype=dtype, check_finite=self.check_finite)
        self.gradient_policy.record(selector, reused=supplied is not None,
            transpose=bool(self.need_dw and supplied is not None and supplied.block_shape is not None and
                           supplied.block_shape[0] == 1 and supplied.block_shape[1] > 1))
        if self.need_dx or self.need_dw:
            if supplied is not None:
                qg = supplied
                if self.need_dw:
                    blocks = supplied.block_shape
                    qgt = (gradient_transpose(gradient, dtype, blocks, self.check_finite)
                           if blocks is not None and blocks[0] == 1 and blocks[1] > 1 else supplied.transpose(contiguous=False))
            elif self.delayed:
                qg, qgt, db = self.call("linear.prepare_gradient", self.scaling_state.quantize,
                    matrix, "grad", normal=self.need_dx, transposed=self.need_dw, bias=self.need_db,
                    training=self.state_training, check_finite=self.check_finite, autotune=self.options["autotune"])
            elif self.grad_transpose:
                if self.need_dx:
                    qg = self.call("linear.prepare_gradient", quantize_fp8, matrix, block_shape=self.blocks["grad"], **kwargs)
                qgt = self.call("linear.prepare_gradient", quantize_fp8, matrix.T, block_shape=self.blocks["grad"], **kwargs)
            elif self.pair_g and self.need_db and self.gradient_reduction == "shared":
                qg, qgt, db = self.call("linear.prepare_gradient", _quantize_gradient_pair_and_bias,
                    matrix, transpose_only=not self.need_dx, **kwargs)
            elif self.pair_g and self.need_dx:
                qg, qgt = self.call("linear.prepare_gradient", _quantize_tensorwise_pair, matrix, **kwargs)
            elif self.pair_g:
                qgt = self.call("linear.prepare_gradient", _quantize_tensorwise_transpose, matrix, **kwargs)
            else:
                qg = self.call("linear.prepare_gradient", quantize_fp8, matrix, block_shape=self.blocks["grad"], **kwargs)
        if self.need_dw and qgt is None:
            qgt = qg.transpose(contiguous=False)
        values = {}
        if qg is not None:
            values.update(qg=qg.data, qg_scale=qg.scale_inv)
        if qgt is not None:
            values.update(qgt=qgt.data, qgt_scale=qgt.scale_inv)
        if self.need_dx or self.need_dw:
            normal_blocks = (qg.block_shape or (-1, -1)) if qg is not None else (-1, -1)
            transpose_blocks = (qgt.block_shape or (-1, -1)) if qgt is not None else (-1, -1)
            values["gradient_blocks"] = torch.tensor(
                (*normal_blocks, *transpose_blocks), device="cpu", dtype=torch.int64)
        if self.need_db:
            values["db"] = (gradient_bias(supplied) if supplied is not None else
                            matrix.sum(0, dtype=torch.float32) if db is None else db).to(self.bias_meta.dtype)
        return result([values[name] for name, _ in self.gradient_schema()])

    def dx(self, gdata, gscale, wdata, wscale, blocks, selector=None):
        from deepopt.opt.quantization.tensor import QuantizedTensor
        from deepopt.opt.backends import gemm_nt

        block = tuple(blocks[:2].tolist())
        qg, qw = QuantizedTensor(gdata, gscale, None if block[0] < 0 else block), QuantizedTensor(wdata, wscale, self.blocks["weight"])
        operand = qw if self.pair_w else qw.transpose(contiguous=False)
        slot = self.gradient_policy.slot(selector)
        if slot is not None:
            with slot.output(self.input_meta.shape, self.input_meta.dtype,
                             training=self.state_training, check_finite=self.check_finite) as output:
                return self.call("linear.dx", gemm_nt, qg, operand, backend=self.route("dx"),
                    out_dtype=torch.float32, num_stages=self.options["num_stages"],
                    autotune=self.options["autotune"], diagnostic_role="dx", gradient_output=output)
        return self.call("linear.dx", gemm_nt, qg, operand, backend=self.route("dx"),
            out_dtype=self.input_meta.dtype, num_stages=self.options["num_stages"],
            autotune=self.options["autotune"], diagnostic_role="dx").reshape(self.input_meta.shape)

    def dw(self, gdata, gscale, xdata, xscale, blocks, *codec):
        from deepopt.opt.quantization.tensor import QuantizedTensor
        from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation
        from deepopt.opt.backends import gemm_nt
        from deepopt.opt.backends._shared.gemm_policy import prepare_native_operands

        block = tuple(blocks[2:].tolist())
        left = QuantizedTensor(gdata, gscale, None if block[0] < 0 else block)
        if self.compressed:
            packed = PackedActivation(xdata, *codec, self.saved_shape, self.compression["group_size"],
                self.blocks["activation"], self.compression["bits"])
            transpose = not self.saved_transpose
            blocks = self.blocks["activation"]
            if transpose and blocks is not None:
                blocks = tuple(reversed(blocks))
            qx = QuantizedTensor(self.call("linear.dw", dequantize_unpack_activation, packed,
                dtype=self.dtypes["activation"], transpose=transpose), xscale.T if transpose else xscale, blocks)
        else:
            qx = QuantizedTensor(xdata, xscale, self.blocks["activation"])
            if not (self.pair_x or self.saved_transpose):
                qx = qx.transpose(contiguous=False)
        selected = self.dw_backend if self.dw_backend != "inherit" and self.options["backend"] != "reference" else self.route("dw")
        preparation = None
        if self.native_layout == "prepare" and selected in ("native_fast", "native_precise"):
            left, qx, preparation = self.call("linear.dw", prepare_native_operands, left, qx)
        return self.call("linear.dw", gemm_nt, left, qx, backend=selected, out_dtype=self.output_meta.dtype,
            num_stages=self.options["num_stages"], autotune=self.options["autotune"],
            diagnostic_role="dw", diagnostic_preparation=preparation).to(self.weight_meta.dtype)


def vjp(ctx, node, gradients):
    spec = node.specification
    state = dict(zip(spec.saved_components, node.values[1:]))
    selector = ctx.saved(state["gradient_selector"]) if "gradient_selector" in state else None
    schema = spec.gradient_schema()
    values = tuple(TensorValue(f"{node.values[0].name}__{name}", metadata) for name, metadata in schema)
    preparation_args = (gradients[0].value,) + ((selector.value,) if selector is not None else ())
    ctx.builder.graph.add(CallNode("fp8.prepare_gradient", recorded(spec, spec.prepare_gradient, "backward"), preparation_args, {}, values))
    prepared = dict(zip((name for name, _ in schema), values))
    contributions = []
    if spec.need_dx:
        operands = (prepared["qg"], prepared["qg_scale"], ctx.saved(state["weight_payload"]), ctx.saved(state["weight_scale_inv"]),
                    prepared["gradient_blocks"])
        if selector is not None:
            operands += (selector,)
        output_meta = replace(spec.input_meta, representation="opaque") if spec.gradient_policy.enabled else spec.input_meta
        contributions.append(GradWrapper(node.args[0], emit(ctx, "fp8.gemm_dx", spec.dx, operands, output_meta)))
    if spec.need_dw:
        operands = (prepared["qgt"], prepared["qgt_scale"], ctx.saved(state["x_payload"]), ctx.saved(state["x_scale_inv"]),
                    prepared["gradient_blocks"])
        if spec.compressed:
            operands += (ctx.saved(state["codec_step"]), ctx.saved(state["codec_minimum"]))
        contributions.append(GradWrapper(node.args[1], emit(ctx, "fp8.gemm_dw", spec.dw, operands, spec.weight_meta)))
    if spec.need_db:
        contributions.append(GradWrapper(node.args[2], prepared["db"]))
    return contributions


def register_rules(registry):
    if not registry.has_backward_rule(FP8Linear.op):
        registry.register_backward_rule(FP8Linear.op, vjp, multi_output=True)
