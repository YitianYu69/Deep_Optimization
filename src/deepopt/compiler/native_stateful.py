"""Native activation/LayerNorm lowering with explicit saved-state ownership.

The programs reuse the numerical executors also used by eager autograd. They
never invoke an eager autograd Function or substitute a native derivative for
compressed derivative/xhat state.
"""

from dataclasses import dataclass
import os

import torch
from torch.autograd.function import once_differentiable

from .ir import IRNode, ExternKernelNode, find_tensor_values


def native_module(target):
    from deepopt.opt.nn.activation.elementwise_modules import ACTIVATION_TYPES
    from deepopt.opt.nn.activation.modules import DOReLU
    from deepopt.opt.nn.normalization.modules import DOLayerNorm, DOFunctionalNormalization

    module = getattr(target, "module", target)
    return module if (type(module) in (*ACTIVATION_TYPES, DOReLU, DOLayerNorm, torch.nn.SiLU)
                      or type(module) is DOFunctionalNormalization and module.kind == "layer") else None


def functional_activation(target):
    from deepopt.opt.nn.activation.spec import FUNCTIONS

    return FUNCTIONS.get(getattr(target, "original", target))


def functional_exp(target):
    from deepopt.opt.nn.primitive_state import model_exp

    return getattr(target, "original", target) in (torch.exp, torch.Tensor.exp, model_exp)


@dataclass(eq=False)
class NativeExpNode(ExternKernelNode):
    pass


@dataclass(eq=False)
class NativeStoragePrimitiveNode(ExternKernelNode):
    pass


def storage_primitive_extension():
    import operator
    from .registry import OperatorExtension

    cats = (torch.cat, torch.concat, torch.concatenate)
    adds = (torch.add, operator.add, torch.Tensor.add, torch.Tensor.__add__)
    def match(node, gm):
        return (node.op == "call_function" and getattr(node.target, "_deepopt_compiler_operation", False)
                and getattr(node.target, "original", None) in (*cats, *adds))
    def lower(context, node):
        from deepopt.opt.nn.primitive_state import model_add, model_cat

        args, kwargs, _, _ = context.map_arg(node.args)
        if node.target.original in cats:
            values = args[0] if args else kwargs["tensors"]
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            operation, function, operands = "storage_cat", model_cat, (values, dim)
        else:
            operands = tuple(args[i] if len(args) > i else kwargs[name] for i, name in enumerate(("input", "other")))
            operation, function = "storage_add", model_add
        result = NativeStoragePrimitiveNode(operation, function, operands, {}, context.tensor_value(node), autograd_owner="pytorch")
        context.add_node(node, result)
        return result
    return OperatorExtension(name="native_storage_primitive", node_type=NativeStoragePrimitiveNode,
        matcher=match, lowering=lower, kernel_codegen=lambda node, context: node.target)


def exp_extension():
    from .registry import OperatorExtension

    def match(node, gm):
        return node.op == "call_function" and functional_exp(node.target)

    def lower(context, node):
        from deepopt.opt.nn.primitive_state import model_exp

        target = context.resolve_target(node)
        if getattr(target, "_deepopt_compiler_operation", False):
            call_args, call_kwargs, _, _ = context.map_arg(node.args)
        else:
            call_args, call_kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
        if set(call_kwargs) - {"input"}:
            return context.external(node, reason="Model exp requires the functional no-out ABI")
        x = call_args[0] if call_args else call_kwargs["input"]
        result = NativeExpNode("exp", model_exp, (x,), {}, context.tensor_value(node), autograd_owner="pytorch")
        context.add_node(node, result)
        return result

    return OperatorExtension(name="native_model_exp", node_type=NativeExpNode, matcher=match,
        lowering=lower, kernel_codegen=lambda node, context: node.target)


def fake_output(module, x, parameters, buffers=(), *, args=None, kwargs=None):
    from deepopt.opt.nn.normalization.modules import DOLayerNorm, DOFunctionalNormalization

    from deepopt.opt.precision.bf16 import invocation_bf16_policy

    if type(module) is torch.nn.SiLU:
        return torch.empty_like(x, requires_grad=x.requires_grad)
    call_args, call_kwargs = args if args is not None else (x,), kwargs or {}
    inplace = call_kwargs.get("inplace", call_args[1] if len(call_args) > 1 else None)
    inplace = getattr(module, "inplace", False) if inplace is None else inplace
    if inplace and "fp8_training" not in module.plan:
        return torch.empty_like(x, requires_grad=x.requires_grad)
    if (invocation_bf16_policy(module.plan, x).storage_enabled
            or "fp8_training" in module.plan or type(module) is DOFunctionalNormalization):
        from .native_operations import fake_output as fake_leaf

        return fake_leaf(module, args if args is not None else (x,), kwargs or {}, parameters, buffers)

    if not x.is_contiguous() or x.dtype == torch.float64:
        from copy import deepcopy
        from torch._subclasses.fake_tensor import FakeCopyMode
        from deepopt.opt.precision.planner import _frame
        from deepopt.opt.precision.gradient import _active

        memo = {id(real): fake for real, fake in zip(module.parameters(), parameters)}
        with x.fake_mode, FakeCopyMode(x.fake_mode):
            cloned = deepcopy(module, memo)
            frame, gradient = _frame.set(None), _active.set(None)
            try:
                return cloned(*(args if args is not None else (x,)), **(kwargs or {}))
            finally:
                _active.reset(gradient)
                _frame.reset(frame)
    dtype = (torch.float32 if type(module) is DOLayerNorm and x.device.type == "cuda"
             and torch.is_autocast_enabled("cuda") else x.dtype)
    return torch.empty(x.shape, device=x.device, dtype=dtype,
                       requires_grad=torch.is_grad_enabled() and any(t.requires_grad for t in (x, *parameters)))


@dataclass(eq=False)
class NativeStatefulNode(IRNode):
    op: str
    args: tuple
    kwargs: dict
    value: object
    module: object
    module_path: str | None
    node_name: str
    root: object = None
    plan: object = None
    options: object = None

    def inputs(self):
        return list(find_tensor_values(self.args))

    def outputs(self):
        return (self.value,)


def extension():
    from .registry import OperatorExtension

    def match(node, gm):
        target = gm.get_submodule(node.target) if node.op == "call_module" else node.target
        return node.op in ("call_module", "call_function") and (native_module(target) is not None
            or getattr(target, "_deepopt_compiler_operation", False)
            and functional_activation(target) is not None)

    def lower(context, node):
        from deepopt.opt.nn.normalization.modules import DOLayerNorm, DOFunctionalNormalization

        target = context.resolve_target(node)
        module = native_module(target)
        if module is None:
            from deepopt.opt._autocast import _get_fp8_plan, _get_activation_compression
            from deepopt.opt.nn.activation.spec import arguments
            from deepopt.opt.nn.activation.elementwise_modules import adapter
            from deepopt.opt.nn.functional import _prepare_linear_plan

            call_args, call_kwargs, _, _ = context.map_arg(node.args)
            kind, x, options = arguments(target.original, call_args, call_kwargs)
            device = torch.device(x.meta.device).type
            active, compression = _get_fp8_plan(device), _get_activation_compression(device)
            plan = (_prepare_linear_plan(active) if isinstance(active, dict) else
                    _prepare_linear_plan({"fp8": False, "activation_compression": compression or {}}))
            if "fp8_training" in plan and compression is not None:
                plan = _prepare_linear_plan({**plan, "fp8_training": {**plan["fp8_training"], "activation_compression": compression}})
            module = adapter(kind, plan=plan, **options)
            module.train(target.root.training)
            parameters = ()
        elif node.op == "call_module":
            call_args, call_kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
            x = call_args[0] if call_args else call_kwargs["input"]
            parameters = [context.parameter(node, name, p) for name, p in module.named_parameters(recurse=False)]
        else:
            call_args, call_kwargs, parameters, _buffers = context.map_arg(node.args)
            x = call_args[0] if call_args else call_kwargs["input"]
        from deepopt.opt.nn.activation.elementwise_modules import ACTIVATION_TYPES
        from deepopt.opt.nn.activation.modules import DOReLU

        if type(module) is torch.nn.SiLU:
            from deepopt.opt.nn.activation.elementwise_modules import adapter

            module = adapter("silu", original=module, plan={"fp8": False, "activation_compression": {"enabled": False}})
            module._native_torch_activation = True

        if type(module) in (*ACTIVATION_TYPES, DOReLU):
            from copy import copy
            from inspect import signature

            # A prepared adapter may be shared by calls with different options.
            # Bind its public signature, retaining None's module-default meaning,
            # and snapshot scalars without changing the shared module or state.
            bound = signature(module.forward).bind(*call_args, **call_kwargs)
            effective = {}
            for name in ("inplace", "approximate"):
                if hasattr(module, name):
                    value = bound.arguments.get(name)
                    effective[name] = getattr(module, name) if value is None else value
            if "inplace" in effective and type(effective["inplace"]) is not bool:
                raise TypeError("inplace must be a boolean")
            approximate = effective.get("approximate", "none")
            if approximate not in ("none", "tanh") or getattr(module, "kind", "relu") != "gelu" and approximate != "none":
                raise ValueError("approximate must be 'none' or 'tanh' for GELU")
            if any(value != getattr(module, name) for name, value in effective.items()):
                module = copy(module)
                for name, value in effective.items():
                    setattr(module, name, value)
        if "fp8_training" in module.plan and not context.native_autodiff:
            return context.external(node, reason="FP8 saved-state lowering requires native AutoDiff")
        if getattr(module, "inplace", False):
            if context.write_rewrite(node) is None:
                return context.external(node, reason="In-place native activation retains its mutation owner")
            from copy import copy

            module = copy(module)
            module.inplace = False
        fp8_input = x.meta is not None and x.meta.storage_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        if x.meta is None or x.meta.representation != "dense" and not fp8_input or x.meta.dtype not in (torch.float32, torch.bfloat16):
            return context.external(node, reason="Native saved-state program requires dense BF16/FP32 input")
        from .scheduler import _is_contiguous

        strided = not fp8_input and not _is_contiguous(x.meta)
        is_norm = type(module) is DOLayerNorm or type(module) is DOFunctionalNormalization
        by_name = dict(zip((name for name, _ in module.named_parameters(recurse=False)), parameters))
        args = (x, by_name.get("weight"), by_name.get("bias"))
        options = None
        if type(module) is DOFunctionalNormalization:
            def argument(index, name, default=None):
                return call_args[index] if len(call_args) > index else call_kwargs.get(name, default)
            args = (x, argument(2, "weight"), argument(3, "bias"))
            options = {"normalized_shape": tuple(argument(1, "normalized_shape")), "eps": argument(4, "eps", 1e-5)}
        from deepopt.opt._autocast import _get_fp8_plan, _get_activation_compression
        from deepopt.opt.nn.functional import _prepare_linear_plan

        plan = module.plan
        active = _get_fp8_plan(torch.device(x.meta.device).type)
        compression = _get_activation_compression(torch.device(x.meta.device).type)
        if "fp8_training" in plan:
            if active is False:
                plan = _prepare_linear_plan({"fp8": False, "activation_compression": compression or {}})
            elif active is not None:
                plan = _prepare_linear_plan(active)
            if compression is not None and "fp8_training" in plan:
                plan = _prepare_linear_plan({**plan, "fp8_training": {**plan["fp8_training"], "activation_compression": compression}})
        output_value = context.tensor_value(node)
        if strided:
            from dataclasses import replace

            # Numerical executors support strided reads. Their selected CUDA
            # variant may return a different physical layout from fake/meta
            # inference, so do not expose a guessed flat-kernel ABI.
            output_value.meta = replace(output_value.meta, stride=None)
        result = NativeStatefulNode(
            "native_layer_norm" if is_norm else f"native_{getattr(module, 'kind', 'relu')}", args, {},
            output_value, module,
            node.target if node.op == "call_module" else getattr(target, "path", None), node.name,
            getattr(target, "root", context.graph_module), plan, options,
        )
        context.add_node(node, result)
        return result

    def inference_codegen(node, context):
        if "fp8_training" in node.module.plan:
            from .autodiff.stateful_fp8 import StatefulFP8
            from deepopt.opt.scaling.delayed import fp8_scaling_scope
            from contextlib import nullcontext

            spec = StatefulFP8(node, context)
            def execute(x, weight, bias):
                if torch.is_grad_enabled() and node.value.meta.requires_grad:
                    return node.module(x)
                with fp8_scaling_scope() if spec.delayed else nullcontext():
                    return spec.forward(x, weight, bias)[0]
            return execute
        return NativeStatefulProgram(node, context)

    return OperatorExtension(
        name="native_saved_state", node_type=NativeStatefulNode, matcher=match, lowering=lower, pure=False,
        kernel_codegen=inference_codegen,
        training_codegen=inference_codegen,
    )


class _StatefulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, program):
        ctx.program = program
        with program.scope("forward"):
            return program.forward(ctx, x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with ctx.program.scope("backward"):
            return (*ctx.program.backward(ctx, gradient), None)


class NativeStatefulProgram:
    reuses_existing_kernels = True
    precision = "native"
    execution = "existing_semantic_kernels"
    generated_kernels = False
    activation = epilogue = None
    decode = None

    def __init__(self, node, context):
        from deepopt.kernel_policy import resolve_kernel_policy

        self.module = node.module
        self.bf16_policy = node.bf16_policy
        self.node_options = node.options
        self.operation = "layer_norm" if node.op == "native_layer_norm" else "relu" if node.op == "native_relu" else "activation"
        self.module_path, self.node_name = node.module_path, node.node_name
        self.module_training = bool(self.module.training)
        self.training = bool(node.value.meta.requires_grad and torch.is_grad_enabled())
        self.kernel_policy = resolve_kernel_policy(context.kernel_policy)
        self.kernel_policy_fingerprint = self.kernel_policy.fingerprint
        self.profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"
        self.plan = node.plan if node.plan is not None else self.module.plan
        if "kernel_policy" in self.plan:
            from deepopt.kernel_policy import plan_kernel_policy

            if plan_kernel_policy(self.plan) != self.kernel_policy:
                raise ValueError("Compiler kernel policy conflicts with the prepared native saved-state policy")
        self.output_dtype = node.value.meta.dtype
        family = "normalization" if self.operation == "layer_norm" else "activation"
        self.kernel_routes = {role: self.kernel_policy.route(family, f"{family}.{role}") for role in ("forward", "backward")}
        compressed = self.training and self.plan["activation_compression"]["enabled"]
        need_dx = self.training and node.args[0].meta.requires_grad
        need_dw = self.training and node.args[1] is not None and node.args[1].meta.requires_grad
        if self.operation == "layer_norm":
            self.saved_components = (
                ("xhat_payload", "codec_step", "codec_minimum", "inverse", "center_shift", "moment_gain")
                if compressed and (need_dx or need_dw) else ("input", "mean", "inverse") if need_dx or need_dw else ()
            ) + (("weight",) if need_dx and node.args[1] is not None else ())
        else:
            self.saved_components = (() if not need_dx else ("mask_payload",) if getattr(self.module, "kind", "relu") in ("relu", "relu6") else
                ("derivative_payload", "codec_step", "codec_minimum") if compressed else ("derivative",))
        self.forward_steps = (node.op, "save_configured_state")
        self.backward_steps = (f"{node.op}_backward",)
        self.source = repr({"forward": self.forward_steps, "backward": self.backward_steps, "saved": self.saved_components})

    def __call__(self, x, weight, bias):
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt._autocast import _suspend_fp8_interception

        with _suspend_fp8_interception(), kernel_policy_scope(self.kernel_policy):
            return _StatefulFunction.apply(x, weight, bias, self)

    def scope(self, phase):
        from contextlib import nullcontext

        if self.profile_ranges or torch.autograd._profiler_enabled():
            return torch.profiler.record_function(f"deepopt.compiler.{self.module_path or self.node_name}.{phase}.{self.operation}")
        return nullcontext()

    def forward(self, ctx, x, weight, bias):
        from deepopt.opt._execution import finite_checks_enabled

        compression = self.plan["activation_compression"]
        self.module.last_execution = {}
        if self.operation == "relu":
            from deepopt.opt.nn.autograd.activation.relu import ReLUFunction
            from deepopt.opt.nn.activation.geometry import ReLUGeometry

            return ReLUFunction.forward(ctx, x, ReLUGeometry(tuple(x.shape), "last"),
                dict(dtype=self.output_dtype, block_shape=None, packed=compression["enabled"], inplace=False,
                     bf16_policy=self.bf16_policy,
                     reuse=False, scale=None, check_finite=finite_checks_enabled() and compression["enabled"]),
                None, self.module_training, self.training and x.requires_grad, self.module.last_execution)
        if self.operation == "activation":
            from deepopt.opt.nn.activation.geometry import ReLUGeometry
            from deepopt.opt.nn.autograd.activation.elementwise import activation_forward

            options = dict(dtype=self.output_dtype, block_shape=None, packed=compression["enabled"],
                           bf16_policy=self.bf16_policy,
                           inplace=False, reuse=False, scale=None, check_finite=finite_checks_enabled() and compression["enabled"],
                           kind=self.module.kind, approximate=self.module.approximate,
                           bits=compression["bits"], group_size=compression["group_size"])
            return activation_forward(ctx, x, ReLUGeometry(tuple(x.shape), "last"), options,
                                      None, self.module_training, self.training and x.requires_grad,
                                      self.module.last_execution)
        from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
        from deepopt.opt.nn.autograd.normalization.layer import layer_norm_forward

        options_shape = self.node_options or {"normalized_shape": self.module.normalized_shape, "eps": self.module.eps}
        geometry = LayerNormGeometry(tuple(x.shape), tuple(options_shape["normalized_shape"]), options_shape["eps"], "last")
        options = dict(dtype=self.output_dtype, logical_dtype=self.output_dtype, block_shape=None,
                       backward_enabled=self.training, compression=compression, scale=None,
                       backend=self.kernel_policy.backend if x.device.type == "cuda" else "reference",
                       check_finite=finite_checks_enabled() and compression["enabled"])
        return layer_norm_forward(ctx, x, weight, bias, geometry, options, None, self.module_training,
                                  self.module.last_execution)

    def backward(self, ctx, gradient):
        if self.operation == "relu":
            from deepopt.opt.nn.autograd.activation.relu import ReLUFunction

            return ReLUFunction.backward(ctx, gradient)[0], None, None
        if self.operation == "activation":
            from deepopt.opt.nn.autograd.activation.elementwise import activation_backward

            return activation_backward(ctx, gradient)[0], None, None
        from deepopt.opt.nn.autograd.normalization.layer import layer_norm_backward

        return layer_norm_backward(ctx, gradient)[:3]
