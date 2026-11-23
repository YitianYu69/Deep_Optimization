"""Compiler-specific opaque calls preserving prepared operation autograd.

Dynamo records the callable; this backend executes it without AOTAutograd
tracing through the current DeepOpt Function. Parameter and buffer operands
are explicit graph inputs. Private caches retain their imperative ownership.
These facades are installed only for the DeepOpt backend, never globally on
the public functional primitives.
"""

from copy import copy, deepcopy
import operator

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensor
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_leaves

from .operations import is_deepopt_target


class _Operation:
    _deepopt_compiler_operation = True
    __module__ = "deepopt.compiler.capture"

    def __init__(self, root, path, original):
        self.root = root
        self.path = path
        self.original = original
        module = root.get_submodule(path) if path else root
        self.module = module
        from .linear import linear_module
        from .native_linear import native_module
        from .native_stateful import native_module as stateful_module
        from .native_operations import native_module as spatial_module
        from .native_mha import native_module as mha_module

        self.native_linear = native_module(module) is not None and path is not None
        self.native_function = path is None and original is torch.nn.functional.linear
        self.native_stateful = stateful_module(module) is not None and path is not None
        self.native_spatial = spatial_module(module) is not None and path is not None
        self.native_mha = mha_module(module) is not None and path is not None
        self.local_operands = path is not None and (linear_module(module) is not None or self.native_linear or self.native_stateful or self.native_spatial or self.native_mha)
        self.native_compression = None
        self.writes_input = (
            bool(getattr(module, "inplace", False)) if path is not None else True
        )
        self.__name__ = (
            "deepopt_" + (path.replace(".", "_") or "root")
            if path is not None
            else original.__name__
        )

    def __repr__(self):
        return self.__name__

    def __call__(self, args, kwargs, parameters, buffers):
        fake = next(
            (
                getattr(value, "_data", value)
                for value in tree_leaves((args, kwargs, parameters, buffers))
                if isinstance(getattr(value, "_data", value), FakeTensor)
            ),
            None,
        )
        if fake is None:
            if (
                self.path is not None
                and self.root.get_submodule(self.path) is not self.module
            ):
                raise ValueError(
                    "A prepared operation was replaced after capture; reconstruct compilation"
                )
            # No enclosing compiler Function: the existing operation attaches
            # its own gradient graph, packed state and exact scale snapshots.
            return self.original(*args, **kwargs)
        if self.local_operands:
            if self.native_spatial or self.native_mha:
                from .native_operations import fake_output

                return fake_output(self.module, args, kwargs, parameters, buffers)
            if self.native_stateful:
                from .native_stateful import fake_output

                x = args[0] if args else kwargs.get("input", kwargs.get("x"))
                return fake_output(self.module, x, parameters, buffers, args=args, kwargs=kwargs)
            elif self.native_linear:
                from .native_linear import fake_linear
            else:
                from .linear import fake_linear

                x = args[0] if args else kwargs.get("input", kwargs.get("x"))
                return fake_linear(self.module, x, parameters, call_args=args, call_kwargs=kwargs)

            x = args[0] if args else kwargs.get("input", kwargs.get("x"))
            return fake_linear(self.module, x, parameters)
        if self.path is None:
            from deepopt.opt.nn.functional import do_linear
            from deepopt.opt._autocast import _get_fp8_plan
            from .native_operations import functional_convolution, fake_functional_convolution

            from .native_stateful import functional_activation
            from deepopt.opt.nn.primitive_state import primitive_arguments, model_add, model_cat

            primitive = primitive_arguments(self.original, args, kwargs)
            if primitive is not None:
                kind, values, dim = primitive
                return model_cat(values, dim) if kind == "cat" else model_add(*values)

            if self.original in (torch.exp, torch.Tensor.exp):
                from deepopt.opt.nn.primitive_state import model_exp

                return model_exp(args[0] if args else kwargs["input"])
            from deepopt.opt.nn.primitive_state import NEAREST2D_FUNCTIONS, nearest2d_call

            if self.original in NEAREST2D_FUNCTIONS:
                return nearest2d_call(self.original, args, kwargs)
            if functional_activation(self.original):
                from deepopt.opt.nn.activation.spec import arguments
                from deepopt.opt.nn.functional import _activation

                kind, x, options = arguments(self.original, args, kwargs)
                return _activation(kind, x, **options)
            if self.original is torch.nn.functional.scaled_dot_product_attention:
                from deepopt.opt._autocast import _suspend_fp8_interception

                # Functional operations have no module state. Discover the output contract
                # without entering the eager saved-tensor compression boundary.
                with _suspend_fp8_interception():
                    return self.original(*args, **kwargs)

            if functional_convolution(self.original) is not None:
                return fake_functional_convolution(self.original, args, kwargs)

            plan = kwargs.get("plan")
            if self.original in (do_linear, torch.nn.functional.linear):
                active = _get_fp8_plan(fake.device.type)
                selected = plan if isinstance(plan, dict) and plan.get("fp8") is False else active if active is not None else plan
                if isinstance(selected, dict) and "fp8_training" in selected:
                    from types import SimpleNamespace
                    from .linear import fake_linear

                    x = args[0] if args else kwargs.get("x", kwargs.get("input"))
                    weight = args[1] if len(args) > 1 else kwargs["weight"]
                    bias = args[2] if len(args) > 2 else kwargs.get("bias")
                    if x.ndim and weight.ndim == 2 and x.shape[-1] == weight.shape[1]:
                        return fake_linear(SimpleNamespace(plan=selected, out_features=weight.shape[0]), x,
                            (weight,) + ((bias,) if bias is not None else ()))
            native = self.native_function and _get_fp8_plan(fake.device.type) in (None, False)
            if native or self.original is do_linear and isinstance(plan, dict) and plan.get("fp8") is False:
                x = args[0] if args else kwargs.get("x", kwargs.get("input"))
                weight = args[1] if len(args) > 1 else kwargs["weight"]
                bias = args[2] if len(args) > 2 else kwargs.get("bias")
                if not torch.is_autocast_enabled(x.device.type) and any(
                        t.dtype != x.dtype for t in (weight, bias) if t is not None):
                    raise RuntimeError("Native Linear operands must have the same dtype without autocast")
                dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
                return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=dtype,
                    requires_grad=torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias) if t is not None))
        # Clone under the caller's FakeTensorMode. No real buffers, caches or
        # Python counters are mutated while discovering the output contract.
        memo = {}
        for real, symbolic in zip(self.root.parameters(), parameters):
            memo[id(real)] = symbolic
        for real, symbolic in zip(self.root.buffers(), buffers):
            memo[id(real)] = symbolic
        mode = fake.fake_mode
        from deepopt.opt.precision.planner import _Frame, _frame
        from deepopt.opt.precision.gradient import _active

        frame = _frame.get()
        with mode, FakeCopyMode(mode):
            cloned = deepcopy(self.root, memo)
            cloned_frame = (
                _Frame(cloned, frame.training, dict(frame.positions))
                if frame is not None
                else None
            )
            token = _frame.set(cloned_frame)
            gradient_token = _active.set(None)
            try:
                if self.path is None:
                    return self.original(*args, **kwargs)
                module = cloned.get_submodule(self.path) if self.path else cloned
                return module(*args, **kwargs)
            finally:
                _active.reset(gradient_token)
                _frame.reset(token)


class _CaptureForward:
    """An instance-local forward facade; deep copies recover ordinary eager code."""

    _deepopt_compiler_capture = True

    def __init__(self, root, path, original):
        self.root = root
        self.original = original
        # Root operations can wrap the already separated numerical body.
        # Preserve that ownership marker during same-policy preparation.
        self._deepopt_compiler_precision_body = getattr(original, "_deepopt_compiler_precision_body", False)
        self.operation = torch.compiler.allow_in_graph(_Operation(root, path, original))
        self.operands = self.operation.module if self.operation.local_operands else root

    def __deepcopy__(self, memo):
        return deepcopy(self.original, memo)

    def __call__(self, *args, **kwargs):
        if not torch.compiler.is_compiling():
            return self.original(*args, **kwargs)
        if self.operation.native_linear or self.operation.native_function:
            x = args[0] if args else kwargs.get("input", kwargs.get("x"))
            # Preserve the existing FP32 capture/AOT path. BF16 AMP can have
            # FP32 input storage, so inspect its effective compute dtype.
            dtype = (torch.get_autocast_dtype(x.device.type)
                     if torch.is_autocast_enabled(x.device.type) else x.dtype)
            if dtype != torch.bfloat16:
                return self.original(*args, **kwargs)
        return self.operation(
            args, kwargs, tuple(self.operands.parameters()), tuple(self.operands.buffers())
        )


class _CaptureFunction(_CaptureForward):
    """Prepared FX functional calls use the same opaque operand contract."""

    __module__ = "deepopt.compiler.capture"
    _deepopt_compiler_operation = True

    def __init__(self, root, original):
        super().__init__(root, None, original)
        self.__name__ = original.__name__ + "_compiler"


class _GraphForward:
    """FX's Dynamo integration inspects forward.__code__ even for callable forwards."""

    def __init__(self, original):
        self.original = original
        self._deepopt_compiler_precision_body = getattr(original, "_deepopt_compiler_precision_body", False)

    @property
    def __code__(self):
        return type(self.original).__call__.__code__

    def __call__(self, *args, **kwargs):
        return self.original(*args, **kwargs)

    def __deepcopy__(self, memo):
        return deepcopy(self.original, memo)


class _FunctionalCaptureMode(TorchFunctionMode):
    """Expose functional saved-state calls within this model's compiler entry."""

    _deepopt_compiler_capture_mode = True

    def __init__(self, captures):
        super().__init__()
        self.captures = captures

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if torch.compiler.is_compiling() and func in self.captures:
            from deepopt.opt.nn.primitive_state import primitive_arguments

            if primitive_arguments(func, args, kwargs) is not None:
                return self.captures[func](*args, **kwargs)
            from deepopt.opt.nn.primitive_state import nearest2d_arguments

            resize = nearest2d_arguments(func, args, kwargs)
            if resize is not None:
                from deepopt._bf16_policy import current_bf16_policy
                from deepopt.opt._autocast import _get_bf16_compression

                x = resize[0]
                if (_get_bf16_compression(x) is not None
                        or current_bf16_policy(x.device.type).storage_enabled):
                    return self.captures[func](*args, **kwargs)
                return func(*args, **kwargs)
            if func in (torch.exp, torch.Tensor.exp):
                from deepopt._bf16_policy import current_bf16_policy

                x = args[0] if args else kwargs.get("input")
                if x is not None and x.dtype == torch.bfloat16 and current_bf16_policy(x.device.type).storage_enabled:
                    return self.captures[func](*args, **kwargs)
                return func(*args, **kwargs)
            if func is torch.nn.functional.scaled_dot_product_attention:
                return self.captures[func](*args, **kwargs)
            from .native_operations import functional_convolution

            if functional_convolution(func) is not None:
                return self.captures[func](*args, **kwargs)
            from deepopt.opt._autocast import _get_bf16_compression, _get_fp8_plan
            from deepopt.opt.nn.activation.spec import arguments

            activation = arguments(func, args, kwargs)
            if activation is not None:
                kind, x, options = activation
                root = self.captures[func].root
                planned = getattr(root, "_deepopt_precision_functional", {}).get(root.training, {})
                from deepopt._bf16_policy import current_bf16_policy

                if (not planned.get(kind)
                        and _get_fp8_plan(x.device.type) in (None, False)
                        and (_get_bf16_compression(x) is not None or current_bf16_policy(x.device.type).storage_enabled)):
                    return self.captures[func](*args, **kwargs)
        return func(*args, **kwargs)


class _CaptureHost:
    """Establish instance-owned functional capture outside Dynamo's graph."""

    def __init__(self, original, captures):
        self.original, self.captures = original, captures

    def __call__(self, *args, **kwargs):
        with _FunctionalCaptureMode(self.captures):
            return self.original(*args, **kwargs)


class _PrecisionBody:
    """Compile the numerical body; establish precision frames on the host."""

    _deepopt_compiler_precision_body = True

    def __init__(self, original):
        self.original = original
        self.function = original.override or type(original.owner).forward.__get__(original.owner)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @property
    def __code__(self):
        code = getattr(self.function, "__code__", None)
        return code if code is not None else type(self.function).__call__.__code__

    def __deepcopy__(self, memo):
        return deepcopy(self.original, memo)


def prepare_precision_body(model):
    original = model.forward
    if not (getattr(original, "_deepopt_precision_forward", False)
            or getattr(original, "_deepopt_bf16_forward", False)):
        return None
    model.forward = _PrecisionBody(original)
    return original


def bind_compiled_precision(model, original):
    compiled = model._compiled_call_impl
    captures = getattr(model, "_deepopt_compiler_functional_capture", None)
    if captures is not None:
        compiled = _CaptureHost(compiled, captures)
    if original is not None:
        host = copy(original)
        host.override = compiled
        compiled = host
    model._compiled_call_impl = compiled


def prepare_capture(model, *, native_compression=None):
    """Expose prepared module calls to this compiler, preserving model identity."""
    from .native_operations import native_module
    from .native_mha import native_module as mha_module
    from .native_stateful import native_module as stateful_module

    from deepopt.opt.nn.activation.spec import FUNCTIONS
    from deepopt.opt.nn.primitive_state import NEAREST2D_FUNCTIONS

    convolutions = tuple(getattr(torch.nn.functional, f"{prefix}{rank}d")
                         for prefix in ("conv", "conv_transpose") for rank in (1, 2, 3))
    model._deepopt_compiler_functional_capture = {function: _CaptureFunction(model, function)
        for function in (torch.nn.functional.scaled_dot_product_attention, *FUNCTIONS, *convolutions, *NEAREST2D_FUNCTIONS, torch.exp, torch.Tensor.exp,
                         torch.cat, torch.concat, torch.concatenate, torch.add, operator.add, torch.Tensor.add, torch.Tensor.__add__)}
    for path, module in model.named_modules():
        if (is_deepopt_target(module) or type(module) is torch.nn.Linear or native_module(module) is not None
                or stateful_module(module) is not None or mha_module(module) is not None) and not getattr(
            module.forward, "_deepopt_compiler_capture", False
        ):
            module.forward = _CaptureForward(model, path, module.forward)
            module.forward.operation.native_compression = native_compression
        if isinstance(module, torch.fx.GraphModule):
            changed = False
            for node in module.graph.nodes:
                if (
                    node.op == "call_function"
                    and (is_deepopt_target(node.target) or node.target is torch.nn.functional.linear)
                    and (
                        not isinstance(node.target, _CaptureFunction)
                        or node.target.root is not model
                    )
                ):
                    original = (
                        node.target.original
                        if isinstance(node.target, _CaptureFunction)
                        else node.target
                    )
                    node.target = _CaptureFunction(model, original)
                    changed = True
            if changed:
                precision_body = module.forward if isinstance(module.forward, _PrecisionBody) else None
                module.recompile()
                if precision_body is not None:
                    precision_body.function = type(module).forward.__get__(module)
                    module.forward = precision_body
            if not hasattr(module.forward, "__code__"):
                module.forward = _GraphForward(module.forward)
    return model


def remove_capture(model):
    """Restore eager forwards after a failed setup or a deliberate backend switch."""
    for module in model.modules():
        compiled = getattr(module, "_compiled_call_impl", None)
        if isinstance(compiled, _CaptureHost) or isinstance(getattr(compiled, "override", None), _CaptureHost):
            module._compiled_call_impl = None
        if hasattr(module, "_deepopt_compiler_functional_capture"):
            del module._deepopt_compiler_functional_capture
        if isinstance(module.forward, _CaptureForward):
            module.forward = module.forward.original
        if isinstance(module.forward, _GraphForward):
            module.forward = module.forward.original
        if isinstance(module.forward, _PrecisionBody):
            module.forward = module.forward.original
        if isinstance(module, torch.fx.GraphModule):
            changed = False
            for node in module.graph.nodes:
                if node.op == "call_function" and isinstance(
                    node.target, _CaptureFunction
                ):
                    node.target = node.target.original
                    changed = True
            if changed:
                module.recompile()
