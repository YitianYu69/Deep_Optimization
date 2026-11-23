"""Built-in FP8 Linear lowering and recipe-specialized training launch programs.

These programs call quantization/GEMM/codec executors directly. They never
invoke DOLinear.forward or its imperative autograd Function. Saved operands
are explicit in the program schema and remain visible to autograd hooks.
"""

from dataclasses import dataclass
from math import prod
import os
from textwrap import dedent

import torch
from torch.autograd.function import once_differentiable

from .ir import IRNode, TensorValue, find_tensor_values


@dataclass(eq=False)
class DeepOptLinearNode(IRNode):
    op: str
    args: tuple
    kwargs: dict
    value: TensorValue
    target: object
    plan: object
    activation: tuple[str, str] | None = None
    module_path: str | None = None
    module: object = None

    def inputs(self):
        return list(find_tensor_values((self.args, self.kwargs)))

    def outputs(self):
        return (self.value,)


def linear_module(target):
    from deepopt.opt.nn.linear import DOLinear
    from deepopt.opt.nn.functional_linear import DOFunctionalLinear
    from deepopt.opt.nn.hf_conv1d import DOHFConv1D

    module = getattr(target, "module", target)
    return module if type(module) in (DOLinear, DOFunctionalLinear, DOHFConv1D) and "fp8_training" in module.plan else None


def resolve_plan(module, device_type):
    from deepopt.opt._autocast import _get_activation_compression, _get_fp8_plan
    from deepopt.opt.nn.functional import _prepare_fp8_plan

    active = _get_fp8_plan(device_type)
    if active is False:
        return None
    plan = module.plan if active is None else active
    compression = _get_activation_compression(device_type)
    if compression is None:
        compression = module.plan["fp8_training"]["activation_compression"]
    if compression != plan["fp8_training"]["activation_compression"]:
        plan = _prepare_fp8_plan({
            "fp8_training": {**plan["fp8_training"], "activation_compression": compression},
            "block_shapes": dict(plan["block_shapes"]),
        })
    return plan


def eligibility(target, meta, *, native_autodiff=False):
    module = linear_module(target)
    if module is None:
        return None, "unregistered operation"
    fp8_input = (meta is not None and native_autodiff and meta.storage_dtype in
                 (torch.float8_e4m3fn, torch.float8_e5m2) and (meta.layout == "last" or len(meta.shape) == 2))
    if meta is None or meta.representation != "dense" and not fp8_input or meta.device is None:
        return None, "Linear input has no supported dense representation"
    root = getattr(target, "root", module)
    settings = getattr(root, "_deepopt_fp8_config", None)
    if not native_autodiff and settings is not None and settings.backward_precision != "native":
        return None, "planned FP8 gradient transport retains its existing owner"
    plan = resolve_plan(module, torch.device(meta.device).type)
    if plan is None:
        return None, "scope selects native Linear"
    if not native_autodiff and plan["fp8_training"]["scaling_recipe"] != "current":
        return None, "delayed scaling retains its existing state owner"
    if not native_autodiff and any(block is not None for block in plan["block_shapes"].values()):
        return None, "blockwise scaling retains its existing backward owner"
    if not meta.shape or getattr(module, "in_features", meta.shape[-1]) != meta.shape[-1]:
        return None, "Linear shape contract is unsupported"
    return plan, ""


def extension():
    from .registry import OperatorExtension

    def match(node, gm):
        from deepopt.opt.nn import functional as dof
        from deepopt.opt._autocast import _get_fp8_plan
        from .lowering import metadata_from_fx

        if node.op not in ("call_function", "call_module"):
            return False
        target = gm.get_submodule(node.target) if node.op == "call_module" else node.target
        if linear_module(target) is not None:
            return True
        function = getattr(target, "original", target)
        if function not in (dof.do_linear, dof.linear, torch.nn.functional.linear):
            return False
        args, kwargs = node.args, node.kwargs
        if getattr(target, "_deepopt_compiler_operation", False):
            args, kwargs = args[:2]
        plan = kwargs.get("plan")
        output_meta = metadata_from_fx(node)
        active = _get_fp8_plan(torch.device(output_meta.device).type) if output_meta is not None and output_meta.device is not None else None
        return not (isinstance(plan, dict) and plan.get("fp8") is False) and active is not False and (
            isinstance(plan, dict) and "fp8_training" in plan or isinstance(active, dict))

    def lower(context, node):
        from deepopt.opt.nn.functional_linear import DOFunctionalLinear
        from deepopt.opt.nn.hf_conv1d import DOHFConv1D

        target = context.resolve_target(node)
        module = linear_module(target)
        if module is None:
            from types import SimpleNamespace
            from deepopt.opt._autocast import _get_fp8_plan
            from deepopt.opt.nn.functional import _prepare_fp8_plan
            from deepopt.opt.quantization.cache import WeightCache

            call_args, call_kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
            if getattr(target, "_deepopt_compiler_operation", False):
                call_args, call_kwargs, _, _ = call_args
            x = call_args[0] if call_args else call_kwargs.get("x", call_kwargs.get("input"))
            weight = call_args[1] if len(call_args) > 1 else call_kwargs["weight"]
            bias = call_args[2] if len(call_args) > 2 else call_kwargs.get("bias")
            root = getattr(target, "root", context.graph_module)
            plan = call_kwargs.get("plan") or _get_fp8_plan(torch.device(x.meta.device).type)
            if not isinstance(plan, dict):
                return context.external(node, reason="Functional FP8 Linear requires a resolved FP8 plan")
            plan = _prepare_fp8_plan(plan)
            if plan["fp8_training"]["scaling_recipe"] == "delayed" and call_kwargs.get("_scaling_state") is None:
                return context.external(node, reason="Functional delayed Linear requires prepared scaling state")
            paths = getattr(root, "_deepopt_backward_functional", {})
            module = SimpleNamespace(plan=plan, training=root.training, in_features=weight.meta.shape[1],
                out_features=weight.meta.shape[0], _fp8_weight_cache=WeightCache(),
                fp8_scaling=call_kwargs.get("_scaling_state"),
                _gradient_slots={mode: tuple(paths.get(mode, {}).get("linear", ())) for mode in (True, False)},
                _gradient_key="functional:linear", _compiler_gradient_key="gradient_functional:linear")
            facade = SimpleNamespace(module=module, root=root)
            selected = resolve_plan(module, torch.device(x.meta.device).type)
            lowered = DeepOptLinearNode("deepopt_linear", (x, weight, bias), {}, context.tensor_value(node),
                facade, selected, module_path=getattr(target, "path", node.name), module=module)
            context.add_node(node, lowered)
            return lowered
        if node.op == "call_module":
            args, kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
            x = args[0] if args else kwargs.get("x", kwargs.get("input"))
            if type(module) is DOFunctionalLinear:
                weight = args[1] if len(args) > 1 else kwargs["weight"]
                bias = args[2] if len(args) > 2 else kwargs.get("bias")
            else:
                weight = context.parameter(node, "weight", module.weight)
                bias = context.parameter(node, "bias", module.bias) if module.bias is not None else None
        else:
            args = context.map_arg(node.args)
            call_args, call_kwargs, parameters, _buffers = args
            x = call_args[0] if call_args else call_kwargs["x"]
            # Prepared capture uses module-local parameter operands for Linear.
            if type(module) is DOFunctionalLinear:
                weight = call_args[1] if len(call_args) > 1 else call_kwargs["weight"]
                bias = call_args[2] if len(call_args) > 2 else call_kwargs.get("bias")
            else:
                weight = parameters[0]
                bias = parameters[1] if module.bias is not None else None
        if type(module) is DOHFConv1D:
            from dataclasses import replace
            from .trace.builder import GraphBuilder

            weight = GraphBuilder(context.graph, value_prefix=f"{node.name}_weight_").emit_native(
                "autodiff_primitive_transpose", torch.ops.aten.transpose.int, (weight, 0, 1), {},
                replace(weight.meta, shape=tuple(reversed(weight.meta.shape)), stride=tuple(reversed(weight.meta.stride)))).value
        plan, reason = eligibility(target, getattr(x, "meta", None), native_autodiff=context.native_autodiff)
        if plan is None:
            return context.external(node, reason=reason)
        value = context.tensor_value(node)
        if value.meta is None:
            return context.external(node, reason="Linear output metadata is unavailable")
        lowered = DeepOptLinearNode(
            op="deepopt_linear", args=(x, weight, bias), kwargs={}, value=value,
            target=target, plan=plan,
            module_path=node.target if node.op == "call_module" else getattr(target, "path", None),
        )
        context.add_node(node, lowered)
        return lowered

    def inference_codegen(node, context):
        if any(prod(value.meta.shape) == 0 for value in node.args[:2]):
            raise ValueError("FP8 linear currently requires non-empty input and weight")
        if node.module is not None or node.plan["fp8_training"]["scaling_recipe"] != "current" or any(block is not None for block in node.plan["block_shapes"].values()):
            from .autodiff.fp8 import FP8Linear

            return FP8Linear(node, context)
        return LinearProgram(node, kernel_policy=context.kernel_policy)

    return OperatorExtension(
        name="deepopt_linear", node_type=DeepOptLinearNode,
        matcher=match, lowering=lower, pure=False,
        kernel_codegen=inference_codegen,
        training_codegen=lambda node, context: LinearProgram(node, kernel_policy=context.kernel_policy),
    )


def fake_linear(module, x, parameters, *, call_args=(), call_kwargs=None):
    """Allocation-only output inference; no model clone, kernel or cache access."""
    plan = resolve_plan(module, x.device.type)
    dtype = (plan["fp8_training"]["out_dtype"] if plan is not None else None)
    if dtype is None:
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    out_features = getattr(module, "out_features", None)
    if out_features is None:
        kwargs = call_kwargs or {}
        weight = call_args[1] if len(call_args) > 1 else kwargs["weight"]
        bias = call_args[2] if len(call_args) > 2 else kwargs.get("bias")
        parameters = (weight,) + ((bias,) if bias is not None else ())
        out_features = weight.shape[0]
    shape = (*x.shape[:-1], out_features)
    if x.numel() == 0 or any(tensor.numel() == 0 and tensor.ndim == 2 for tensor in parameters):
        if plan is not None:
            raise ValueError("FP8 linear currently requires non-empty input and weight")
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    alignment = 1
    if plan is not None and plan["fp8_training"]["backend"] != "reference":
        alignment = 4 if dtype == torch.float32 else 8
    row_stride = ((shape[-1] + alignment - 1) // alignment) * alignment
    strides = [1] * len(shape)
    stride = row_stride
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = stride
        stride *= shape[index]
    requires_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (x, *parameters))
    return torch.empty_strided(shape, tuple(strides), dtype=dtype, device=x.device, requires_grad=requires_grad)


def _configure(ctx, program, x, weight, bias):
    from deepopt.opt._execution import backward_finite_check_policy
    from deepopt.opt.backends._shared.gemm_policy import linear_backward_policy, gradient_reduction_policy
    from deepopt.opt.backends._shared.gemm_diagnostics import capture_gemm_recording
    from deepopt.opt.precision.gradient import capture_gradient

    capture_gemm_recording(ctx)
    ctx.dw_backend_override, ctx.native_layout = linear_backward_policy()
    ctx.gradient_reduction = gradient_reduction_policy()
    ctx.gradient_slot = capture_gradient(x)
    ctx.input_shape, ctx.input_dtype = x.shape, x.dtype
    ctx.weight_grad_dtype = program.output_dtype
    ctx.bias_dtype = bias.dtype if bias is not None else None
    ctx.out_features = weight.shape[0]
    ctx.fp8_dtypes = dict(program.plan["fp8_training"]["fp8_dtypes"])
    ctx.block_shapes = dict(program.plan["block_shapes"])
    ctx.backend = program.plan["fp8_training"]["backend"]
    selected = program.plan["fp8_training"]["backward_backend"]
    ctx.backward_backend = ctx.backend if selected == "inherit" else selected
    ctx.num_stages = program.plan["fp8_training"]["num_stages"]
    ctx.autotune = program.plan["fp8_training"]["autotune"]
    ctx.check_finite = backward_finite_check_policy()
    ctx.scaling_state, ctx.state_training = None, program.module.training


class _CompiledLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, program):
        from deepopt.kernel_policy import kernel_policy_scope

        ctx.program = program
        ctx.kernel_policy = program.kernel_policy
        _configure(ctx, program, x, weight, bias)
        with kernel_policy_scope(program.kernel_policy, role="linear.forward"):
            return program.forward(ctx, x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from deepopt.kernel_policy import kernel_policy_scope

        with kernel_policy_scope(ctx.kernel_policy):
            gradients = ctx.program.backward(ctx, gradient)
        return (*gradients, None)


class LinearProgram:
    """Generated forward/backward launch schedule for a static Linear recipe."""

    reuses_existing_kernels = True

    def __init__(self, node, *, kernel_policy=None):
        from deepopt.opt.precision.gradient import gradient_operation
        from deepopt.kernel_policy import plan_kernel_policy, resolve_kernel_policy

        self.module = linear_module(node.target)
        self.module_path = node.module_path
        self.node_name = node.value.name
        self.module_training = bool(self.module.training)
        self.plan = node.plan
        self.kernel_policy = (plan_kernel_policy(self.plan) if kernel_policy is None
                              else resolve_kernel_policy(kernel_policy))
        self.kernel_policy_fingerprint = self.kernel_policy.fingerprint
        self.weight_cache_recipe = (None if self.kernel_policy.backend == "triton" else
                                    (("compiler-linear", self.kernel_policy_fingerprint), "current"))
        plan_policy = plan_kernel_policy(self.plan)
        if self.plan["fp8_training"]["backend"] != "reference" and plan_policy != self.kernel_policy:
            raise ValueError("Compiler kernel policy conflicts with the prepared Linear policy")
        self.kernel_routes = {
            role: self.kernel_policy.route("gemm", f"linear.{role}")
            for role in ("forward", "dx", "dw", "db")
        }
        self.activation = node.activation
        meta = node.value.meta
        self.output_dtype = meta.dtype
        x, weight, bias = node.args
        self.training = bool(meta.requires_grad and torch.is_grad_enabled())
        need_dx = self.training and x.meta.requires_grad
        need_dw = self.training and weight.meta.requires_grad
        need_db = self.training and bias is not None and bias.meta.requires_grad
        compression = self.plan["fp8_training"]["activation_compression"]
        packed = need_dw and compression["enabled"]
        paired = self.training and self.plan["fp8_training"]["backend"] != "reference" and torch.device(x.meta.device).type == "cuda"
        pair_x = paired and need_dw and not packed and prod(x.meta.shape) > 4096
        pair_w = paired and need_dx and prod(weight.meta.shape) > 4096
        pair_g = paired and need_dw and prod(meta.shape) > 4096
        self.saved_components = tuple(
            (["x_payload", "x_scale_inv"] if need_dw else [])
            + (["weight_payload", "weight_scale_inv"] if need_dx else [])
            + (["codec_step", "codec_minimum"] if packed else [])
            + (["activation_input"] if self.activation and need_dx else [])
        )
        self.forward_steps = (
            f"fused_{self.activation[0]}_operand_preparation" if self.activation else
            "quantize_pack" if packed else "quantize_pair" if pair_x else "quantize",
            "cached_weight_pair" if pair_w else "cached_weight",
            "fp8_gemm", "restore_output",
        )
        self.backward_steps = tuple(
            (["quantize_gradient"] if need_dx or need_dw else [])
            + (["fp8_gemm_dx"] if need_dx else [])
            + (["unpack_transpose"] if packed else [])
            + (["fp8_gemm_dw"] if need_dw else [])
            + (["bias_reduce"] if need_db else [])
        )
        self.source = _generate_source(
            need_dx, need_dw, need_db, packed, pair_x, pair_w, pair_g, self.training,
            self.activation,
        )
        namespace = _namespace(self)
        exec(compile(self.source, "<deepopt_linear_program>", "exec"), namespace)
        from deepopt.opt.backends._shared.gemm_diagnostics import restore_gemm_recording

        self.forward = namespace["forward"]
        self.backward = restore_gemm_recording(namespace["backward"])
        # Instrument only an explicitly requested diagnostic pass. Normal
        # execution retains the direct generated callables and imports no
        # profiler API from this module.
        self.profile_ranges = os.environ.get("DEEPOPT_PROFILE_RANGES") == "1"
        if self.profile_ranges:
            from torch.profiler import record_function

            def profiled(function, phase):
                path = "<unknown>" if self.module_path is None else self.module_path or "<root>"
                label = f"deepopt.linear::{path}::{phase}"

                def execute(*args, **kwargs):
                    with record_function(label):
                        return function(*args, **kwargs)

                return execute

            self.forward = profiled(self.forward, "forward")
            self.backward = profiled(self.backward, "backward")

        @gradient_operation("linear")
        def invoke(module, x, weight, bias):
            return _CompiledLinearFunction.apply(x, weight, bias, self)

        self.invoke = invoke

    def __call__(self, x, weight, bias):
        from deepopt.opt._autocast import _suspend_fp8_interception

        with _suspend_fp8_interception():
            return self.invoke(self.module, x, weight, bias)


def _namespace(program):
    from deepopt.kernel_policy import kernel_policy_scope, kernel_backend
    from .activation_quantize import prepare as prepare_activation
    from deepopt.opt.backends import gemm_nt
    from deepopt.opt.backends._shared.gemm_policy import linear_forward_backend, prepare_native_operands
    from deepopt.opt.nn.autograd._output import restore_fresh_output
    from deepopt.opt.quantization.quantizer import quantize_fp8, _quantize_tensorwise_pair, _quantize_tensorwise_transpose, _quantize_gradient_pair_and_bias
    from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
    from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation
    from deepopt.opt.quantization.tensor import QuantizedTensor
    from deepopt.opt.quantization.gradient import gradient_operand, gradient_bias, gradient_transpose, enabled
    from deepopt.opt.precision.gradient import record_gradient_input

    def role_call(role, function, *args, **kwargs):
        with kernel_policy_scope(program.kernel_policy, role=role):
            return function(*args, **kwargs)

    def product_backend(ctx, role):
        if ctx.backend == "reference":
            return "reference"
        if role != "forward" and ctx.backward_backend in ("native_fast", "native_precise"):
            return ctx.backward_backend
        return kernel_backend("gemm", backend=program.kernel_policy, role=f"linear.{role}")

    return dict(
        torch=torch, program=program, quantize=quantize_fp8,
        quantize_pair=_quantize_tensorwise_pair, quantize_transpose=_quantize_tensorwise_transpose,
        quantize_gradient_pair=_quantize_gradient_pair_and_bias,
        pack=quantize_pack_fp8_activation, unpack=dequantize_unpack_activation,
        PackedActivation=PackedActivation, QuantizedTensor=QuantizedTensor,
        gemm=gemm_nt, restore=restore_fresh_output,
        forward_backend=linear_forward_backend, prepare_native=prepare_native_operands,
        gradient_operand=gradient_operand, gradient_bias=gradient_bias,
        gradient_transpose=gradient_transpose,
        gradient_enabled=enabled, record_gradient=record_gradient_input,
        prepare_activation=prepare_activation,
        role_call=role_call, product_backend=product_backend,
    )


def _generate_source(dx, dw, db, packed, pair_x, pair_w, pair_g, training, activation):
    return dedent(f'''
    def forward(ctx, x, weight, bias):
        matrix = x.reshape(-1, weight.shape[1])
        if {bool(activation)!r}:
            options = program.plan['fp8_training']['activation_compression'] if {packed!r} else None
            qx, qxt, saved = role_call('compiler.preparation', prepare_activation, matrix, activation=program.activation, fp8_dtype=ctx.fp8_dtypes['activation'], paired={pair_x!r}, compression=options)
            if {packed!r}:
                ctx.compression_shape, ctx.compression_group_size = saved.shape, saved.group_size
                ctx.compression_bits, ctx.compression_block_shape = saved.bits, saved.block_shape
        elif {packed!r}:
            options = program.plan['fp8_training']['activation_compression']
            qx, saved = role_call('linear.prepare_input', pack, matrix, fp8_dtype=ctx.fp8_dtypes['activation'], group_size=options['group_size'], bits=options['bits'])
            ctx.compression_shape, ctx.compression_group_size = saved.shape, saved.group_size
            ctx.compression_bits, ctx.compression_block_shape = saved.bits, saved.block_shape
        elif {pair_x!r}:
            qx, qxt = role_call('linear.prepare_input', quantize_pair, matrix, fp8_dtype=ctx.fp8_dtypes['activation'])
        else:
            qx = role_call('linear.prepare_input', quantize, matrix, fp8_dtype=ctx.fp8_dtypes['activation'])
        weight_quantizer = lambda *args, **kwargs: role_call('linear.prepare_weight', quantize_pair if {pair_w!r} else quantize, *args, **kwargs)
        weight_value = program.module._fp8_weight_cache.quantize(weight, fp8_dtype=ctx.fp8_dtypes['weight'], block_shape=None, quantizer=weight_quantizer, paired={pair_w!r}, recipe=program.weight_cache_recipe)
        qw, qwt = weight_value if {pair_w!r} else (weight_value, None)
        override = forward_backend()
        selected = override if override != 'inherit' and ctx.backend != 'reference' else product_backend(ctx, 'forward')
        output = role_call('linear.forward', gemm, qx, qw, backend=selected, bias=bias, out_dtype=program.output_dtype, num_stages=ctx.num_stages, autotune=ctx.autotune, diagnostic_role='forward')
        if {training!r}:
            ctx.save_for_backward(
                (saved.data if {packed!r} else qxt.data if {pair_x!r} else qx.data) if {dw!r} else None,
                qx.scale_inv if {dw!r} else None,
                (qwt.data if {pair_w!r} else qw.data) if {dx!r} else None,
                (qwt.scale_inv if {pair_w!r} else qw.scale_inv) if {dx!r} else None,
                *((saved.step, saved.minimum) if {packed!r} else ()),
                *((x,) if {bool(activation) and dx!r} else ()),
            )
        return restore(output, lambda values: values.reshape(*x.shape[:-1], weight.shape[0]))

    def backward(ctx, gradient):
        saved_tensors = ctx.saved_tensors
        if {bool(activation) and dx!r}:
            activation_input = saved_tensors[-1]
            saved_tensors = saved_tensors[:-1]
        xdata, xscale, wdata, wscale, *codec = saved_tensors
        supplied = gradient_operand(gradient, 'last')
        grad_blocks = supplied.block_shape if supplied is not None else None
        matrix = None if supplied is not None else gradient.reshape(-1, ctx.out_features)
        record_gradient(ctx.gradient_slot, reused=supplied is not None, transpose=bool({dw!r} and supplied is not None and grad_blocks is not None and grad_blocks[0] == 1 and grad_blocks[1] > 1))
        dx = dw = db = None
        if {dx or dw!r}:
            if supplied is not None:
                qg = supplied
            elif {pair_g!r} and {db!r} and ctx.gradient_reduction == 'shared':
                qg, qgt, db = role_call('linear.prepare_gradient', quantize_gradient_pair, matrix, fp8_dtype=ctx.fp8_dtypes['grad'], check_finite=ctx.check_finite, transpose_only={not dx!r})
            elif {pair_g and dx!r}:
                qg, qgt = role_call('linear.prepare_gradient', quantize_pair, matrix, fp8_dtype=ctx.fp8_dtypes['grad'], check_finite=ctx.check_finite)
            elif {pair_g!r}:
                qgt = role_call('linear.prepare_gradient', quantize_transpose, matrix, fp8_dtype=ctx.fp8_dtypes['grad'], check_finite=ctx.check_finite)
            else:
                qg = role_call('linear.prepare_gradient', quantize, matrix, fp8_dtype=ctx.fp8_dtypes['grad'], check_finite=ctx.check_finite)
        if {dx!r}:
            qw = QuantizedTensor(wdata, wscale)
            operand = qw if {pair_w!r} else qw.transpose(contiguous=False)
            if gradient_enabled(ctx.gradient_slot):
                with ctx.gradient_slot.output(ctx.input_shape, ctx.input_dtype, check_finite=ctx.check_finite) as result:
                    dx = role_call('linear.dx', gemm, qg, operand, backend=product_backend(ctx, 'dx'), out_dtype=torch.float32, num_stages=ctx.num_stages, autotune=ctx.autotune, gradient_output=result, diagnostic_role='dx')
            else:
                dx = role_call('linear.dx', gemm, qg, operand, backend=product_backend(ctx, 'dx'), out_dtype=ctx.input_dtype, num_stages=ctx.num_stages, autotune=ctx.autotune, diagnostic_role='dx').reshape(ctx.input_shape)
        if {dw!r}:
            if {packed!r}:
                saved = PackedActivation(xdata, *codec, ctx.compression_shape, ctx.compression_group_size, ctx.compression_block_shape, ctx.compression_bits)
                qx = QuantizedTensor(role_call('linear.dw', unpack, saved, dtype=ctx.fp8_dtypes['activation'], transpose=True), xscale.T)
            else:
                qx = QuantizedTensor(xdata, xscale)
                if {not pair_x!r}:
                    qx = qx.transpose(contiguous=False)
            if grad_blocks is not None and grad_blocks[0] == 1 and grad_blocks[1] > 1:
                left = role_call('linear.dw', gradient_transpose, gradient, ctx.fp8_dtypes['grad'], grad_blocks, ctx.check_finite)
            elif supplied is not None:
                left = supplied.transpose(contiguous=False)
            elif {pair_g!r}:
                left = qgt
            else:
                left = qg.transpose(contiguous=False)
            selected = ctx.dw_backend_override if ctx.dw_backend_override != 'inherit' and ctx.backend != 'reference' else product_backend(ctx, 'dw')
            preparation = None
            if ctx.native_layout == 'prepare' and selected in ('native_fast', 'native_precise'):
                left, qx, preparation = role_call('linear.dw', prepare_native, left, qx)
            dw = role_call('linear.dw', gemm, left, qx, backend=selected, out_dtype=ctx.weight_grad_dtype, num_stages=ctx.num_stages, autotune=ctx.autotune, diagnostic_role='dw', diagnostic_preparation=preparation)
        if {db!r} and db is None:
            db = (role_call('linear.db', gradient_bias, supplied) if supplied is not None else matrix.sum(dim=0, dtype=torch.float32)).to(ctx.bias_dtype)
        elif {db!r}:
            db = db.to(ctx.bias_dtype)
        if {bool(activation) and dx!r}:
            if {bool(activation and activation[0] == "gelu")!r}:
                dx = torch.ops.aten.gelu_backward.default(dx, activation_input, approximate={activation[1] if activation else "none"!r})
            else:
                dx = torch.ops.aten.silu_backward.default(dx, activation_input)
        return dx, dw, db
    ''')


def _fusion_signature(value):
    """Static metadata only: do not inspect a tensor or materialize its data."""
    meta = getattr(value, "meta", None)
    if meta is None:
        return None

    def dimension(value):
        return value if type(value) is int else str(value)

    return {
        "shape": [dimension(item) for item in meta.shape],
        "stride": [dimension(item) for item in meta.stride] if meta.stride is not None else None,
        "dtype": str(meta.dtype),
        "device": str(meta.device) if meta.device is not None else None,
        "requires_grad": bool(meta.requires_grad),
        "representation": meta.representation,
    }


def fuse_activations(graph, *, enabled, unsafe_values, kernel_policy=None, diagnostics=None):
    """Fuse supported producers, optionally recording every Linear site.

    ``diagnostics`` receives JSON-safe static decisions. A selected region is
    compilation evidence; it does not claim a runtime launch was observed.
    The returned pattern tuple remains compatible with existing callers.
    """
    if not enabled and diagnostics is None:
        return ()
    from .ir import GetAttrNode, PointwiseNode, register_uses
    from .scheduler import _is_contiguous
    from deepopt.kernel_policy import kernel_backend
    from .activation_quantize import preparation_backend

    removed = set()
    patterns = []
    for index, node in enumerate(graph.nodes):
        if not isinstance(node, DeepOptLinearNode):
            continue
        value = node.args[0]
        producer = value.producer
        discovered = isinstance(producer, PointwiseNode) and producer.op in ("gelu", "silu")
        source = producer.args[0] if discovered and producer.args else value
        module = linear_module(node.target)
        candidate = {
            "module_path": node.module_path,
            "node_name": node.value.name,
            "phase": "train" if module.training else "eval",
            "module_training": bool(module.training),
            "training": bool(node.value.meta.requires_grad and torch.is_grad_enabled()),
            "requested": bool(enabled),
            "effective_enabled": bool(enabled),
            "discovered": discovered,
            "eligible": None if not enabled else False,
            "selected": False,
            "status": "disabled" if not enabled else "rejected",
            "reason": "disabled" if not enabled else "no_activation",
            "reason_detail": "activation fusion disabled" if not enabled else "no adjacent supported native activation",
            "activation": {
                "kind": producer.op,
                "approximate": producer.kwargs.get("approximate", "none"),
            } if discovered else None,
            "input_signature": _fusion_signature(source),
            "activation_output_signature": _fusion_signature(value),
            "selected_schedule": None,
            "observed_cuda_launches": None,
        }
        if diagnostics is not None:
            diagnostics.append(candidate)

        def reject(reason, detail):
            candidate.update(reason=reason, reason_detail=detail)

        if not enabled:
            continue
        if not isinstance(producer, PointwiseNode) or producer.op not in ("gelu", "silu"):
            continue
        if kernel_backend("compiler_pointwise", backend=kernel_policy, role=f"compiler.{producer.op}") != kernel_backend("compiler_preparation", backend=kernel_policy, role="compiler.preparation"):
            reject("backend", "activation and operand preparation have different backend routes")
            continue
        if value in unsafe_values:
            reject("unsafe", "activation output participates in an unsafe alias or effect")
            continue
        if value.users != [node]:
            reject("fanout", "activation output must have exactly one Linear consumer")
            continue
        if len(producer.args) != 1:
            reject("metadata", "activation must have exactly one source operand")
            continue
        source = producer.args[0]
        if not isinstance(source, TensorValue) or source.meta is None or value.meta is None:
            reject("metadata", "activation source or output tensor metadata is unavailable")
            continue
        if (
            source.meta.representation != "dense" or source.meta.dtype != value.meta.dtype
        ):
            reject("layout", "fusion requires dense operands with matching dtypes")
            continue
        if source.meta.requires_grad != value.meta.requires_grad:
            reject("gradient", "activation changes the input gradient requirement")
            continue
        if source.meta.shape != value.meta.shape or not _is_contiguous(source.meta):
            reject("layout", "fusion requires equal shapes and a contiguous activation source")
            continue
        if prod(source.meta.shape) <= 4096:
            reject("size", "activation source must contain more than 4096 elements")
            continue
        if torch.device(source.meta.device).type != "cuda":
            reject("device", "fused activation preparation requires a CUDA source")
            continue
        start = graph.nodes.index(producer)
        if any(not isinstance(item, GetAttrNode) for item in graph.nodes[start + 1:index]):
            reject("effects", "intervening operations prevent atomic activation preparation")
            continue
        approximate = producer.kwargs.get("approximate", "none")
        if approximate not in ("none", "tanh"):
            reject("approximation", "activation approximation is unsupported")
            continue
        compression = node.plan["fp8_training"]["activation_compression"]
        if compression["enabled"] and compression["group_size"] > 16384:
            reject("codec", "fused activation codec group size exceeds 16384")
            continue
        try:
            preparation_backend(compression=compression if compression["enabled"] else None, policy=kernel_policy)
        except ValueError as error:
            # The original activation and Linear nodes retain their explicit
            # routes; this rejects only the optional atomic fusion candidate.
            reject("backend", str(error))
            continue
        node.activation = producer.op, approximate
        node.args = (source, *node.args[1:])
        removed.add(id(producer))
        patterns.append(f"{producer.op} -> FP8 Linear operand preparation")
        candidate.update(
            eligible=True, selected=True, status="selected", reason=None,
            reason_detail="supported activation selected for operand preparation",
            selected_schedule=f"fused_{producer.op}_operand_preparation",
        )
    if removed:
        graph.nodes[:] = [node for node in graph.nodes if id(node) not in removed]
        for node in graph.nodes:
            for value in node.outputs():
                value.users.clear()
        for node in graph.nodes:
            register_uses(node)
        graph.validate()
    return tuple(patterns)
