"""Semantic training operators and VJPs over shared numerical kernels.

No operator in this module creates an autograd boundary. Forward residuals are
typed IR results and VJPs emit numerical operations with explicit operands.
"""

from dataclasses import replace
from math import prod
from contextlib import nullcontext
import os

import torch

from ..ir import CallNode, TensorMeta, TensorValue
from ..trace.metadata_utils import contiguous_stride
from .backward_rules import GradWrapper
from deepopt.opt.backends._shared.activation_codec import saved_activation_layout
from deepopt.opt.backends._shared.normalization import saved_normalization_variant


def meta(shape, dtype, device, *, representation="dense"):
    shape = tuple(shape)
    return TensorMeta(shape, dtype, device, contiguous_stride(shape), False,
                      representation=representation)


def packed_metas(shape, options, device, block=None):
    from deepopt.opt.quantization.activation import _dimensions

    groups, _, words = _dimensions(shape, options["group_size"], block, options["bits"])
    return (meta((groups, words), torch.int32, device, representation="packed"),
            meta((groups,), torch.float32, device), meta((groups,), torch.float32, device))


def record_normalization_state(ctx, node, spec, state, names):
    """Describe the existing normalization ABI and its domain geometry."""
    from .activation_saving import SavedUse

    if any(value.meta.shape is None or value.meta.dtype is None for value in state):
        # Backend-owned FP8/gradient transport remains opaque. Do not invent
        # physical layouts or let classification metadata block its compiler.
        return
    geometry = spec.geometry if hasattr(spec, "geometry") else None
    if geometry is None:
        recipe = {"source_shape": spec.metas[0].shape, "normalization_kind": "group",
                  "normalization_groups": spec.groups}
    elif hasattr(geometry, "normalized_shape"):
        recipe = {"source_shape": geometry.shape, "normalization_kind": "layer",
                  "normalized_shape": geometry.normalized_shape}
    else:
        recipe = {"source_shape": geometry.shape, "normalization_kind": geometry.kind,
                  "normalization_groups": geometry.groups, "batch_stats": geometry.batch_stats}
    slots = dict(zip(names, state))
    statistics_spec = getattr(spec, "native_group", None) or spec
    recipe["statistics_dtype"] = str(getattr(statistics_spec, "stats_dtype", torch.float32))
    compression = getattr(spec, "compression", {})
    if compression.get("enabled") and geometry is not None:
        components = tuple(slots[name] for name in (
            "saved_xhat" if "saved_xhat" in slots else "xhat_payload", "codec_step", "codec_minimum"))
        ctx.record_codec(node.args[0], components, representation="packed_normalized_value",
            recipe={**recipe, "bits": compression["bits"], "group_size": compression["group_size"],
                    "matrix_shape": geometry.matrix_shape, "block_shape": (1, compression["group_size"])},
            role="normalization.normalized_value")
    for name, value in slots.items():
        role = {"inverse": "rstd", "rstd": "rstd", "mean": "mean", "center_shift": "center_shift",
                "moment_gain": "moment_gain"}.get(name)
        if role and prod(value.meta.shape):
            ctx.record_residual(value, (value,), representation="normalization_auxiliary",
                component_roles=(role,), recipe=recipe,
                use=SavedUse(role=f"normalization.{role}"), retention_reason="normalization_domain_statistic")


def result(values):
    return values[0] if len(values) == 1 else tuple(values)


def call_node(node, spec, state_metas):
    output = node.output()
    values = (output,) + tuple(TensorValue(f"{output.name}__{name}", metadata)
                               for name, metadata in zip(spec.saved_components, state_metas))
    lowered = CallNode(spec.op, recorded(spec, spec.forward, "forward"), node.args, node.kwargs, values, spec)
    lowered.bf16_policy = node.bf16_policy
    return lowered


def recorded(spec, function, phase):
    # Callsite identity and role are immutable within a specialization. Bind
    # them once, while leaving the numerical policy and profiler state dynamic.
    from deepopt.kernel_policy import kernel_policy_scope
    from deepopt.opt.backends._shared.gemm_diagnostics import gemm_module_scope

    site = spec.module_path or spec.node_name
    label = (f"deepopt.linear::{site}::{phase}" if spec.precision == "fp8" else
             f"deepopt.compiler.{site}.{phase}.{function.__name__}")
    family = getattr(spec, "family", "linear")
    role = f"{family}.{phase}"
    if family == "linear" and phase == "backward":
        name = function.__name__
        role = "linear." + (name if name in ("dx", "dw", "db") else
                             "dx" if name in ("native_derivative_gemm", "packed_derivative_gemm") else "prepare_gradient")

    def execute(*args, **kwargs):
        profile = spec.profile_ranges or spec.precision != "fp8" and torch.autograd._profiler_enabled()
        scope = torch.profiler.record_function(label) if profile else nullcontext()
        with scope, kernel_policy_scope(spec.kernel_policy, role=role), gemm_module_scope(site):
            return function(*args, **kwargs)
    execute.numerical_function = function
    execute.specification = spec
    if phase == "backward" and function.__name__ == "dw":
        execute.scratch_requirements = getattr(spec, "scratch_requirements", ())
    return execute


def emit(ctx, name, function, args, output_meta):
    spec = getattr(function, "__self__", None)
    if spec is not None and hasattr(spec, "profile_ranges"):
        function = recorded(spec, function, "backward")
    return ctx.builder.emit_native(name, function, tuple(args), {}, replace(output_meta, requires_grad=False))


class BF16Linear:
    family = "linear"
    op = "autodiff_bf16_linear"
    precision = "bf16"
    reuses_existing_kernels = True
    activation = activation_state = epilogue = None
    residual = False
    profile_ranges = True

    def __init__(self, node, context):
        from deepopt.kernel_policy import resolve_kernel_policy, plan_kernel_policy

        self.plan = node.plan
        self.bf16_policy = node.bf16_policy
        self.profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"
        self.module_path, self.node_name = node.module_path, node.node_name
        self.module_training = node.module_training
        self.kernel_policy = resolve_kernel_policy(context.kernel_policy)
        if "kernel_policy" in self.plan and plan_kernel_policy(self.plan) != self.kernel_policy:
            raise ValueError("Compiler kernel policy conflicts with the prepared BF16 Linear policy")
        self.kernel_policy_fingerprint = self.kernel_policy.fingerprint
        self.execution, self.decode = context.bf16_gemm_backend, context.bf16_decode
        self.tile, self.num_stages = context.bf16_tile, context.bf16_num_stages
        self.backward_order = context.bf16_backward_order
        x, weight, bias = node.args
        self.input_meta, self.weight_meta = x.meta, weight.meta
        self.compute_dtype = torch.bfloat16
        self.bias_meta = bias.meta if bias is not None else None
        self.matrix_shape = (prod(x.meta.shape[:-1]), x.meta.shape[-1])
        self.need_dx, self.need_dw = x.meta.requires_grad, weight.meta.requires_grad
        self.need_db = bias is not None and bias.meta.requires_grad
        self.training = bool(node.value.meta.requires_grad)
        self.need_saved_input = self.need_dw and all((*self.matrix_shape, *weight.meta.shape))
        self.compressed = self.need_saved_input and self.plan["activation_compression"]["enabled"]
        self.generated_kernels = (self.execution == "generated" and torch.device(x.meta.device).type == "cuda"
                                  and all((*self.matrix_shape, *weight.meta.shape)))
        if not all((*self.matrix_shape, *weight.meta.shape)):
            self.execution = "native"
        self.kernel_routes = {role: self.backend(role) for role in ("forward", "dx", "dw", "db")}
        self.saved_components = (("x_payload", "codec_step", "codec_minimum") if self.compressed else
                                 ("bf16_input",) if self.need_saved_input else ())
        self.forward_steps = ("bf16_prepare", f"{self.execution}_bf16_gemm")
        self.backward_steps = tuple(f"bf16_gemm_{role}" for role in
                                    (("dw", "dx") if self.backward_order == "dw_first" else ("dx", "dw"))
                                    if getattr(self, f"need_{role}")) + (("bias_reduce",) if self.need_db else ())
        self.source = "DeepOpt native AutoDiff: " + repr(self.backward_steps)
        self.producer_activation = None
        self.producer_derivative = None
        self.retain_preactivation = False
        if (self.compressed and self.decode in ("tiled", "panel")
                and self.kernel_policy.pinned("activation_codec", "linear.dw")
                and self.kernel_policy.route("activation_codec", "linear.dw") != self.kernel_policy.route("gemm", "linear.dw")):
            raise ValueError("Conflicting Mixed routes for BF16 packed-read dW; select materialized decode")
        from deepopt.opt.backends._shared.bf16_resources import bf16_scratch_requirements

        self.scratch_requirements = (bf16_scratch_requirements(
            self.weight_meta.shape[0], self.matrix_shape[1], self.matrix_shape[0],
            route=("direct" if self.decode == "tiled" else self.decode)
            if self.generated_kernels else "materialized") if self.compressed else ())
        # Scratch belongs to the dW call, not the whole Linear forward. The
        # scheduler reads this bound target annotation at its actual call site.
        self.dw_scratch_requirements = self.scratch_requirements

    def backend(self, role):
        return ("native_reduction" if role == "db" else self.kernel_policy.route("gemm", f"linear.{role}")
                if self.generated_kernels else "native")

    def lower(self, node):
        state = (packed_metas(self.matrix_shape, self.plan["activation_compression"], self.input_meta.device)
                 if self.compressed else (meta(self.matrix_shape, self.compute_dtype, self.input_meta.device),)
                 if self.need_saved_input else ())
        return call_node(node, self, state)

    def forward(self, x, weight, bias, residual=None):
        from deepopt.opt.backends.bf16 import product
        from deepopt.opt.quantization.activation import _quantize_pack_strided_activation
        from deepopt.opt.nn.autograd._output import restore_fresh_output

        matrix = x.reshape(self.matrix_shape)
        dense = (matrix if self.generated_kernels and (self.compressed or not self.need_dw)
                 else matrix.to(torch.bfloat16))
        saved = []
        producer_saved = []
        if self.producer_activation is not None:
            from deepopt.opt.backends.bf16 import prepare

            prepared = prepare(matrix, activation=self.producer_activation,
                compression=self.plan["activation_compression"] if self.compressed else None,
                derivative_compression=self.producer_derivative,
                bf16_policy=self.bf16_policy,
                backend=getattr(self, "producer_preparation_backend",
                                self.kernel_policy.route("compiler_preparation", "compiler.preparation")))
            dense, packed = prepared[:2]
            saved = [packed.data, packed.step, packed.minimum] if packed is not None else [dense] if self.need_dw else []
            if self.producer_derivative is not None:
                coefficient = prepared[2]
                producer_saved = [coefficient.data, coefficient.step, coefficient.minimum]
        elif self.compressed:
            options = self.plan["activation_compression"]
            q = _quantize_pack_strided_activation(matrix, options["group_size"], bits=options["bits"],
                                                  source_dtype=torch.bfloat16)
            saved = [q.data, q.step, q.minimum]
        elif self.need_saved_input:
            saved = [dense]
        y = product(matrix if self.generated_kernels and self.producer_activation is None else dense, weight, bias=bias,
                    bf16_policy=self.bf16_policy,
                    backend=self.backend("forward"), tile=self.tile, num_stages=self.num_stages,
                    activation=self.epilogue, return_preactivation=self.retain_preactivation,
                    residual=residual.reshape(-1, weight.shape[0]) if residual is not None else None)
        preactivation = None
        if self.retain_preactivation:
            y, preactivation = y
        y = restore_fresh_output(y, lambda t: t.reshape(*x.shape[:-1], weight.shape[0]))
        if preactivation is not None:
            preactivation = preactivation.reshape(y.shape)
        return result([y, *([preactivation] if self.retain_preactivation else []), *saved, *producer_saved])

    def gradient(self, gradient):
        return gradient.reshape(self.matrix_shape[0], self.weight_meta.shape[0]).to(torch.bfloat16)

    def dx(self, gradient, weight):
        from deepopt.opt.backends.bf16 import product

        return product(gradient, weight.T, backend=self.backend("dx"), tile=self.tile,
                       bf16_policy=self.bf16_policy,
                       num_stages=self.num_stages).reshape(self.input_meta.shape).to(self.input_meta.dtype)

    def dw(self, gradient, *state):
        from deepopt.opt.backends.bf16 import product
        from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

        if not self.need_saved_input:
            return torch.zeros(self.weight_meta.shape, device=gradient.device, dtype=self.weight_meta.dtype)

        q = None
        if self.compressed:
            options = self.plan["activation_compression"]
            q = PackedActivation(*state, self.matrix_shape, options["group_size"], bits=options["bits"])
        backend = self.backend("dw")
        panel = q is not None and self.decode == "panel" and backend == "triton"
        packed_read = q is not None and (self.decode == "tiled" or panel)
        operand = (None if packed_read else dequantize_unpack_activation(q, dtype=torch.bfloat16).T
                   if q is not None and not self.generated_kernels else dequantize_unpack_activation(q, dtype=torch.bfloat16, transpose=True)
                   if q is not None else state[0].T)
        return product(gradient.T, operand, packed_b=q if packed_read else None, backend=backend,
                       bf16_policy=self.bf16_policy,
                       operand_route=(self.decode if packed_read else "materialized") if q is not None else None,
                       packed_route="panel" if panel else "direct", panel_size=1024,
                       tile=self.tile, num_stages=self.num_stages).to(self.weight_meta.dtype)

    def db(self, gradient):
        return gradient.sum(0).to(self.bias_meta.dtype)


def bf16_vjp(ctx, node, gradients):
    spec = node.specification
    if spec.compressed:
        options = spec.plan["activation_compression"]
        ctx.record_codec(node.args[0], node.values[1:4], representation="packed_linear_source",
            recipe={"bits": options["bits"], "group_size": options["group_size"],
                    "matrix_shape": spec.matrix_shape, "source_shape": spec.input_meta.shape},
            role="linear.weight_gradient_coefficient")
    grad = emit(ctx, "bf16.prepare_gradient", spec.gradient, (gradients[0],),
                meta((spec.matrix_shape[0], spec.weight_meta.shape[0]), spec.compute_dtype, spec.input_meta.device))
    contributions = []
    for role in (("dw", "dx") if spec.backward_order == "dw_first" else ("dx", "dw")):
        if not getattr(spec, f"need_{role}"):
            continue
        index = 0 if role == "dx" else 1
        operands = ((ctx.saved(node.args[1]),) if role == "dx" else
                    tuple(ctx.saved(value) for value in node.values[1:]))
        derivative = emit(ctx, f"bf16.gemm_{role}", getattr(spec, role), (grad, *operands), node.args[index].meta)
        contributions.append(GradWrapper(node.args[index], derivative))
    if spec.need_db:
        contributions.append(GradWrapper(node.args[2], emit(ctx, "bf16.bias_reduce", spec.db,
                                                         (grad,), spec.bias_meta)))
    return contributions


class DenseLinear(BF16Linear):
    """Native FP32/FP64 projection semantics used by composite decompositions."""

    op = "autodiff_dense_linear"

    def __init__(self, node, context):
        super().__init__(node, replace(context, bf16_gemm_backend="native", bf16_decode="materialized"))
        self.compute_dtype = node.value.meta.dtype
        self.precision = str(self.compute_dtype).removeprefix("torch.")
        self.compressed &= self.compute_dtype != torch.float64
        self.saved_components = (("x_payload", "codec_step", "codec_minimum") if self.compressed else
                                 ("native_input",) if self.need_saved_input else ())
        self.forward_steps = ("native_linear", "save_requested_operand")
        self.backward_steps = tuple(step.replace("bf16", "native") for step in self.backward_steps)
        self.scratch_requirements = (dict(name="decoded_operand",
            bytes=prod(self.matrix_shape) * self.compute_dtype.itemsize, lifetime="call",
            owner="invocation", reuse="invocation_stream_ordered", capture_safe_after_warmup=True),
            ) if self.compressed else ()

    def forward(self, x, weight, bias):
        from deepopt.opt.quantization.activation import quantize_pack_activation

        dense = x.reshape(self.matrix_shape).to(self.compute_dtype)
        with torch.autocast(x.device.type, enabled=False):
            value = torch.nn.functional.linear(dense, weight.to(self.compute_dtype),
                bias.to(self.compute_dtype) if bias is not None else None).reshape(*x.shape[:-1], weight.shape[0])
        saved = []
        if self.compressed:
            packed = quantize_pack_activation(dense, self.plan["activation_compression"]["group_size"],
                                               bits=self.plan["activation_compression"]["bits"])
            saved = [packed.data, packed.step, packed.minimum]
        elif self.need_saved_input:
            saved = [dense]
        return result([value, *saved])

    def gradient(self, gradient):
        return gradient.reshape(self.matrix_shape[0], self.weight_meta.shape[0]).to(self.compute_dtype)

    def dx(self, gradient, weight):
        return (gradient @ weight.to(self.compute_dtype)).reshape(self.input_meta.shape).to(self.input_meta.dtype)

    def dw(self, gradient, *state):
        from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

        if not self.need_saved_input:
            return torch.zeros(self.weight_meta.shape, device=gradient.device, dtype=self.weight_meta.dtype)

        dense = state[0]
        if self.compressed:
            options = self.plan["activation_compression"]
            dense = dequantize_unpack_activation(PackedActivation(*state, self.matrix_shape,
                options["group_size"], bits=options["bits"]), dtype=self.compute_dtype)
        return (gradient.T @ dense).to(self.weight_meta.dtype)


class NativeActivation:
    family = "activation"
    op = "autodiff_native_activation"
    precision = "native"
    execution = "existing_semantic_kernels"
    generated_kernels = False
    reuses_existing_kernels = True
    activation = activation_state = epilogue = decode = None
    residual = False
    profile_ranges = True

    def __init__(self, node, context):
        from deepopt.kernel_policy import resolve_kernel_policy, plan_kernel_policy
        from deepopt.opt.nn.activation.geometry import ReLUGeometry
        from deepopt.opt.backends._shared.activation_codec import activation_pointwise_tile, gelu_saved_state, silu_saved_state
        from deepopt.opt.backends._shared.activation_codec import activation_layout_policy

        self.module, self.plan = node.module, getattr(node, "plan", None) or node.module.plan
        self.profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"
        self.module_path, self.node_name = node.module_path, node.node_name
        self.module_training, self.training = node.module.training, node.value.meta.requires_grad
        self.kernel_policy = resolve_kernel_policy(context.kernel_policy)
        if "kernel_policy" in self.plan and plan_kernel_policy(self.plan) != self.kernel_policy:
            raise ValueError("Compiler kernel policy conflicts with the prepared native saved-state policy")
        self.kernel_policy_fingerprint = self.kernel_policy.fingerprint
        self.kernel_routes = {role: self.kernel_policy.route("activation", f"activation.{role}")
                              for role in ("forward", "backward")}
        self.input_meta, self.output_dtype = node.args[0].meta, node.value.meta.dtype
        self.bf16_policy = node.bf16_policy
        self.geometry = ReLUGeometry(self.input_meta.shape, "last")
        if (self.bf16_policy.storage_enabled and torch.device(self.input_meta.device).type == "cuda"
                and any(route != "triton" for route in self.kernel_routes.values())):
            raise ValueError("BF16 storage activation requires the Triton CUDA route")
        self.kind, self.approximate = getattr(self.module, "kind", "relu"), getattr(self.module, "approximate", "none")
        self.native_torch_activation = getattr(self.module, "_native_torch_activation", False)
        self.compression = self.plan["activation_compression"]
        self.need_dx = self.input_meta.requires_grad
        state = silu_saved_state() if self.kind == "silu" else gelu_saved_state()
        from ..scheduler import _is_contiguous

        shape = self.input_meta.shape
        channels_last = (len(shape) == 4 and self.input_meta.stride is not None and all(
            size <= 1 or actual == expected for size, actual, expected in zip(shape, self.input_meta.stride,
                (shape[1] * shape[2] * shape[3], 1, shape[3] * shape[1], shape[1]))))
        self.recompute = (self.kind in ("gelu", "silu") and state == "recompute"
                          and not self.compression["enabled"] and self.input_meta.dtype == torch.bfloat16
                          and (_is_contiguous(self.input_meta) or self.kind == "silu" and channels_last))
        if self.bf16_policy.storage_enabled and self.kind in ("gelu", "silu", "sigmoid", "tanh"):
            self.recompute = not self.compression["enabled"]
        self.pointwise_tile = activation_pointwise_tile()
        self.activation_kernel_policy = activation_layout_policy()
        self.saved_components = (() if not self.need_dx else ("coefficient", "codec_step", "codec_minimum")
                                  if self.compression["enabled"] and self.kind not in ("relu", "relu6") else ("coefficient",))
        self.forward_steps, self.backward_steps = (node.op, "save_configured_state"), (f"{node.op}_vjp",)
        self.source = "DeepOpt native AutoDiff activation VJP"

    def lower(self, node):
        shape, device = self.geometry.matrix_shape, self.input_meta.device
        if self.native_torch_activation:
            return call_node(node, self, (replace(self.input_meta, requires_grad=False),) if self.need_dx else ())
        packed = self.compression["enabled"]
        if not self.need_dx:
            state = ()
        elif self.kind in ("relu", "relu6"):
            count = prod(shape)
            state = (meta(((count + 31) // 32,) if packed else (count,),
                          torch.int32 if packed else torch.bool, device),)
        elif packed:
            state = packed_metas(shape, self.compression, device)
        else:
            state = (meta(shape, self.input_meta.dtype if self.recompute else torch.float32, device),)
        return call_node(node, self, state)

    def forward(self, x, weight, bias):
        from deepopt.opt.backends.activation import executor
        if self.native_torch_activation:
            output = torch.ops.aten.silu.default(x)
            return result([output, x] if self.need_dx else [output])
        from deepopt.opt._execution import finite_checks_enabled
        from deepopt.opt.nn.autograd._output import restore_fresh_output

        if self.recompute and self.bf16_policy.storage_enabled:
            x = x.clone(memory_format=torch.contiguous_format)
        if self.kind == "relu":
            output, inverse, mask, _, _, info = executor(x).forward(
                x, self.geometry, dtype=self.output_dtype, block_shape=None, packed=self.compression["enabled"],
                save_mask=self.need_dx, inplace=False, reuse=False, scale=None,
                check_finite=finite_checks_enabled() and self.compression["enabled"])
            self.module.last_execution = {**info, "compression": self.compression["enabled"] and self.need_dx,
                "saved_mask_bytes": mask.numel() * mask.element_size() if mask is not None else 0}
            return result([restore_fresh_output(output, self.geometry.restore), *([mask] if mask is not None else [])])
        output, inverse, saved, _, _, info = executor(x, self.kind).forward(
            x, self.geometry, kind=self.kind, approximate=self.approximate, dtype=self.output_dtype,
            block_shape=None, packed=self.compression["enabled"], bits=self.compression["bits"],
            group_size=self.compression["group_size"], save_mask=self.need_dx, inplace=False,
            check_finite=finite_checks_enabled() and self.compression["enabled"], recompute=self.recompute)
        self.module.last_execution = {**info, "compression": self.compression["enabled"] and self.need_dx,
                                      "saved_state_bytes": sum(t.numel() * t.element_size() for t in saved.tensors) if saved else 0}
        state = saved.tensors if saved else ()
        if self.recompute and state:
            # CUDA can retain the private contiguous N-D input snapshot while
            # the reference executor already returns a matrix. Export the
            # declared matrix ABI as a view of that same snapshot; no payload
            # copy, dtype conversion or relaxation of backward guards is needed.
            state = (state[0].view(self.geometry.matrix_shape), *state[1:])
        return result([restore_fresh_output(output, self.geometry.restore), *state])

    @saved_activation_layout
    def dx(self, gradient, *state):
        if self.native_torch_activation:
            return torch.ops.aten.silu_backward.default(gradient, state[0])
        from deepopt.opt.nn.activation.state import SavedActivation
        from deepopt.opt.backends.activation import executor

        if self.kind == "relu":
            return self.geometry.restore(executor(gradient).backward(gradient, state[0], self.geometry,
                packed=self.compression["enabled"], dtype=self.input_meta.dtype))
        saved = SavedActivation(state[0], *(state[1:] if len(state) == 3 else (None, None)),
            group_size=self.compression["group_size"], bits=self.compression["bits"],
            mask=self.kind == "relu6", packed=self.compression["enabled"],
            recompute=(self.kind, self.approximate) if self.recompute else None)
        return self.geometry.restore(executor(gradient, self.kind).backward(
            gradient, saved, self.geometry, dtype=self.input_meta.dtype, pointwise_tile=self.pointwise_tile))


def activation_vjp(ctx, node, gradients):
    if not ctx.needs_grad(node.args[0]):
        return ()
    spec = node.specification
    derivative = emit(ctx, "activation.coefficient_vjp", spec.dx,
                      (gradients[0], *(ctx.saved(value) for value in node.values[1:])), node.args[0].meta)
    if spec.compression["enabled"] and spec.kind not in ("relu", "relu6"):
        ctx.record_codec(node.args[0], node.values[1:], representation="packed_activation_derivative",
            recipe={"bits": spec.compression["bits"], "group_size": spec.compression["group_size"],
                    "matrix_shape": spec.geometry.matrix_shape, "source_shape": spec.input_meta.shape,
                    "activation": spec.kind}, role="activation.derivative")
    return (GradWrapper(node.args[0], derivative),)


class NativeLayerNorm(NativeActivation):
    family = "normalization"
    op = "autodiff_native_layer_norm"

    def __init__(self, node, context):
        # LayerNorm has no activation kind/approximation fields.
        from deepopt.kernel_policy import resolve_kernel_policy, plan_kernel_policy

        self.bf16_policy = node.bf16_policy
        from deepopt.opt.nn.normalization.policy import validate_normalization_backend

        validate_normalization_backend(self.bf16_policy, context.kernel_policy.backend,
                                       torch.device(node.args[0].meta.device).type)
        from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
        from deepopt.opt.backends._shared.normalization import normalization_variant, normalization_tuning

        self.module, self.plan = node.module, getattr(node, "plan", None) or node.module.plan
        self.profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"
        self.module_path, self.node_name = node.module_path, node.node_name
        self.module_training, self.training = node.module.training, node.value.meta.requires_grad
        self.kernel_policy = resolve_kernel_policy(context.kernel_policy)
        if "kernel_policy" in self.plan and plan_kernel_policy(self.plan) != self.kernel_policy:
            raise ValueError("Compiler kernel policy conflicts with the prepared native saved-state policy")
        self.kernel_policy_fingerprint = self.kernel_policy.fingerprint
        self.normalization_variant, self.normalization_tuning = normalization_variant(), normalization_tuning()
        self.kernel_routes = {role: self.kernel_policy.route("normalization", f"normalization.{role}")
                              for role in ("forward", "backward")}
        self.input_meta, self.output_dtype = node.args[0].meta, node.value.meta.dtype
        options = getattr(node, "options", None) or {"normalized_shape": self.module.normalized_shape, "eps": self.module.eps}
        self.geometry = LayerNormGeometry(self.input_meta.shape, tuple(options["normalized_shape"]), options["eps"])
        self.empty = prod(self.input_meta.shape) == 0
        self.needs = tuple(value is not None and value.meta.requires_grad for value in node.args)
        self.need_h = (self.needs[0] or self.needs[1]) and prod(self.input_meta.shape) > 0
        self.compression = dict(self.plan["activation_compression"])
        self.compression["enabled"] &= self.need_h
        if self.need_h:
            self.compression["group_size"] = min(self.compression["group_size"], self.geometry.elements)
        self.saved_components = (("xhat_payload", "codec_step", "codec_minimum", "inverse", "center_shift", "moment_gain")
                                 if self.compression["enabled"] else ("mean", "inverse") if self.need_h else ())
        self.forward_steps, self.backward_steps = ("layer_norm", "save_statistics"), ("layer_norm_vjp",)
        self.source = "DeepOpt native AutoDiff LayerNorm VJP"
        self.affine_metas = tuple(value.meta if value is not None else None for value in node.args[1:])

    def lower(self, node):
        statistic = meta((self.geometry.rows,), torch.float32, self.input_meta.device)
        state = ((*packed_metas(self.geometry.matrix_shape, self.compression, self.input_meta.device,
                                (1, self.compression["group_size"])), statistic, statistic, statistic)
                 if self.compression["enabled"] else (statistic, statistic) if self.need_h else ())
        return call_node(node, self, state)

    def forward(self, x, weight, bias):
        from deepopt.opt.nn.autograd.normalization.layer import executor
        from deepopt.opt._execution import finite_checks_enabled
        from deepopt.opt.nn.autograd._output import restore_fresh_output

        if self.empty:
            self.module.last_execution = dict(backend="native_empty", compression=False, saved_state_bytes=0)
            return torch.empty_like(x, dtype=self.output_dtype)
        center = torch.empty(self.geometry.rows, device=x.device, dtype=torch.float32) if self.compression["enabled"] else None
        gain = torch.empty_like(center) if center is not None else None
        output, _, state, _, _, info = executor(x, self.kernel_policy.backend).forward(
            x, self.geometry, weight=weight, bias=bias, dtype=self.output_dtype, block_shape=None,
            compression=self.compression, check_finite=finite_checks_enabled() and self.compression["enabled"],
            center_shift=center, moment_gain=gain)
        saved = ([state.packed.data, state.packed.step, state.packed.minimum, state.inverse, center, gain]
                 if state.packed is not None else [state.mean, state.inverse] if self.need_h else [])
        self.module.last_execution = {**info, "compression": self.compression["enabled"],
            "compression_target": "x_hat", "norm_xhat_recenter": center is not None,
            "layer_norm_xhat_moment_correction": gain is not None,
            "saved_state_bytes": sum(t.numel() * t.element_size() for t in saved)}
        return result([restore_fresh_output(output, self.geometry.transport.restore), *saved])

    @saved_normalization_variant
    def gradients(self, gradient, x, weight, bias, *state):
        from deepopt.opt.nn.autograd.normalization.layer import executor
        from deepopt.opt.quantization.activation import PackedActivation

        if self.empty:
            return result([torch.zeros(metadata.shape, dtype=metadata.dtype, device=gradient.device)
                           for metadata, needed in zip((self.input_meta, *self.affine_metas), self.needs) if needed])
        packed = center = gain = mean = inverse = None
        if self.compression["enabled"]:
            data, step, minimum, inverse, center, gain = state
            group = self.compression["group_size"]
            packed = PackedActivation(data, step, minimum, self.geometry.matrix_shape, group,
                                      (1, group), self.compression["bits"])
        elif self.need_h:
            mean, inverse = state
        values = executor(gradient, self.kernel_policy.backend).backward(
            gradient, weight, self.geometry, packed=packed, x=x, mean=mean, inverse=inverse,
            center_shift=center, moment_gain=gain, need_dx=self.needs[0], need_dw=self.needs[1],
            need_db=self.needs[2], input_dtype=self.input_meta.dtype, execution_info=self.module.last_execution)
        return result([value.to(metadata.dtype) for value, needed, metadata in
                       zip(values, self.needs, (self.input_meta, *self.affine_metas)) if needed])


class LibraryLayerNorm(NativeLayerNorm):
    """Exact native normalization with statistics exported by its forward."""

    def forward(self, x, weight, bias):
        output, mean, inverse = torch.ops.aten.native_layer_norm.default(
            x, self.geometry.normalized_shape, weight, bias, self.geometry.eps)
        return result([output, *([mean.reshape(-1), inverse.reshape(-1)] if self.need_h else [])])

    def gradients(self, gradient, x, weight, bias, *state):
        if self.empty:
            return NativeLayerNorm.gradients(self, gradient, x, weight, bias, *state)
        mean, inverse = state if self.need_h else (None, None)
        stats_shape = (*self.input_meta.shape[:-len(self.geometry.normalized_shape)],
                       *((1,) * len(self.geometry.normalized_shape)))
        # db-only still requires native statistics operands, although the
        # kernel need not read their contents with this output mask.
        if mean is None:
            mean = torch.empty(stats_shape, device=gradient.device, dtype=torch.float32)
            inverse = torch.empty_like(mean)
        values = torch.ops.aten.native_layer_norm_backward.default(
            gradient, x if x is not None else torch.empty(self.input_meta.shape, device=gradient.device,
                                                        dtype=self.input_meta.dtype),
            self.geometry.normalized_shape, mean.reshape(stats_shape), inverse.reshape(stats_shape),
            weight, bias, list(self.needs))
        return result([value.to(metadata.dtype) for value, needed, metadata in
                       zip(values, self.needs, (self.input_meta, *self.affine_metas)) if needed])


def layer_norm_vjp(ctx, node, gradients):
    spec = node.specification
    x, weight, bias = node.args
    original = ctx.saved(x) if spec.need_h and not spec.compression["enabled"] else None
    saved_weight = ctx.saved(weight) if (spec.needs[0] or isinstance(spec, LibraryLayerNorm)) and weight is not None else None
    saved_bias = ctx.saved(bias) if isinstance(spec, LibraryLayerNorm) and bias is not None else None
    args = (gradients[0].value, original.value if original is not None else None,
            saved_weight.value if saved_weight is not None else None,
            saved_bias.value if saved_bias is not None else None,
            *(ctx.saved(value).value for value in node.values[1:]))
    record_normalization_state(ctx, node, spec, node.values[1:], spec.saved_components)
    targets = tuple(value for value, needed in zip(node.args, spec.needs) if needed)
    values = tuple(TensorValue(f"{node.values[0].name}_grad_{index}", replace(target.meta, requires_grad=False))
                   for index, target in enumerate(targets))
    ctx.builder.graph.add(CallNode("normalization.layer_norm_vjp", recorded(spec, spec.gradients, "backward"), args, {}, values))
    return tuple(GradWrapper(target, value) for target, value in zip(targets, values))


def register_rules(registry):
    for name, rule in ((BF16Linear.op, bf16_vjp), (NativeActivation.op, activation_vjp),
                       (NativeLayerNorm.op, layer_norm_vjp), (DenseLinear.op, bf16_vjp)):
        if not registry.has_backward_rule(name):
            registry.register_backward_rule(name, rule, multi_output=True)
