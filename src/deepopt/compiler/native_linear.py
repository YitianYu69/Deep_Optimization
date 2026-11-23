"""BF16 Linear lowering, compiler-owned saved state, and fusion decisions."""

from dataclasses import dataclass
from math import prod
import os

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .ir import IRNode, TensorValue, find_tensor_values


@dataclass(eq=False)
class NativeLinearNode(IRNode):
    op: str
    args: tuple
    kwargs: dict
    value: TensorValue
    plan: dict
    module_path: str | None
    node_name: str
    module_training: bool
    activation: tuple | None = None
    epilogue: tuple | None = None
    activation_state: dict | None = None
    activation_module: object = None

    def inputs(self):
        return list(find_tensor_values(self.args))

    def outputs(self):
        return (self.value,)


def native_module(target):
    from deepopt.opt.nn.linear import DOLinear

    module = getattr(target, "module", target)
    native = type(module) is nn.Linear or type(module) is DOLinear and module.plan.get("fp8") is False
    return module if native else None


def fake_linear(module, x, parameters):
    from deepopt.opt._autocast import _get_fp8_plan, _active_interception

    active = _get_fp8_plan(x.device.type)
    if (x.numel() and all(parameter.numel() for parameter in parameters if parameter.ndim == 2)
            and getattr(module, "plan", {}).get("fp8") is not False and active is not None and active is not False
            and _active_interception.get().get(x.device.type, True)):
        from types import SimpleNamespace
        from .linear import fake_linear as fake_fp8

        return fake_fp8(SimpleNamespace(plan=active, out_features=module.out_features), x, parameters)
    if not torch.is_autocast_enabled(x.device.type) and any(p.dtype != x.dtype for p in parameters):
        raise RuntimeError("Native Linear operands must have the same dtype without autocast")
    dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    return torch.empty((*x.shape[:-1], module.out_features), device=x.device, dtype=dtype,
                       requires_grad=torch.is_grad_enabled() and any(t.requires_grad for t in (x, *parameters)))


def _original(target):
    return getattr(target, "original", target)


def extension():
    from .registry import OperatorExtension

    def match(node, gm):
        from .lowering import metadata_from_fx
        from deepopt.opt.nn.functional import do_linear

        if node.op not in ("call_module", "call_function"):
            return False
        target = gm.get_submodule(node.target) if node.op == "call_module" else node.target
        supported = native_module(target) is not None or _original(target) in (
            nn.functional.linear, torch.ops.aten.linear.default, do_linear,
        )
        meta = metadata_from_fx(node)
        if meta is not None and meta.device is not None:
            from deepopt.opt._autocast import _get_fp8_plan, _active_interception

            active = _get_fp8_plan(torch.device(meta.device).type)
            args, kwargs = node.args, node.kwargs
            if getattr(target, "_deepopt_compiler_operation", False):
                args, kwargs = args[:2]
            plan = kwargs.get("plan", getattr(native_module(target), "plan", None))
            if not (isinstance(plan, dict) and plan.get("fp8") is False) and active is not False and (
                    isinstance(plan, dict) and "fp8_training" in plan or isinstance(active, dict)
                    and _active_interception.get().get(torch.device(meta.device).type, True)):
                return False
        # Other numerical modes retain their current native/autograd lowering.
        return supported and meta is not None and meta.dtype in (torch.bfloat16, torch.float32)

    def lower(context, node):
        from deepopt.opt._autocast import _get_fp8_plan, _get_activation_compression, _active_interception
        from deepopt.opt.nn.functional import _prepare_linear_plan, do_linear

        target = context.resolve_target(node)
        module = native_module(target)
        plan = getattr(module, "plan", None)
        if node.op == "call_module":
            x = context.map_arg(node.args[0] if node.args else node.kwargs["input"])
            weight = context.parameter(node, "weight", module.weight)
            bias = context.parameter(node, "bias", module.bias) if module.bias is not None else None
        else:
            args, kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
            if getattr(target, "_deepopt_compiler_operation", False):
                args, kwargs, parameters, _buffers = args
                module_names = {"input"} if type(module) is nn.Linear else {"x", "input"}
                if module is not None and (len(args) > 1 or set(kwargs) - module_names
                        or args and any(name in kwargs for name in ("x", "input"))):
                    return context.external(node, reason="Native Linear argument binding requires original validation")
                if module is not None:
                    x = args[0] if args else kwargs["input"]
                    weight, bias = parameters[0], parameters[1] if module.bias is not None else None
                else:
                    x = args[0] if args else kwargs.get("x", kwargs.get("input"))
                    weight = args[1] if len(args) > 1 else kwargs["weight"]
                    bias = args[2] if len(args) > 2 else kwargs.get("bias")
            else:
                x = args[0] if args else kwargs.get("input", kwargs.get("x"))
                weight = args[1] if len(args) > 1 else kwargs["weight"]
                bias = args[2] if len(args) > 2 else kwargs.get("bias")
            if module is None:
                names = ("x", "weight", "bias") if _original(target) is do_linear else ("input", "weight", "bias")
                allowed = set(names) | ({"plan", "_weight_cache", "_scaling_state", "_state_training"} if _original(target) is do_linear else set())
                if len(args) > 3 or set(kwargs) - allowed or any(name in kwargs for name in names[:len(args)]):
                    return context.external(node, reason="Native Linear argument binding requires original validation")
            if _original(target) is do_linear:
                plan = kwargs.get("plan")
        if not all(isinstance(v, TensorValue) and v.meta is not None and v.meta.representation == "dense"
                   for v in (x, weight) + ((bias,) if bias is not None else ())):
            return context.external(node, reason="BF16 Linear requires dense operand metadata")
        if (len(weight.meta.shape) != 2 or not x.meta.shape
                or x.meta.shape[-1] != weight.meta.shape[1]):
            return context.external(node, reason="BF16 Linear shape uses native execution")
        if not context.native_autodiff and (prod(x.meta.shape) == 0 or prod(weight.meta.shape) == 0):
            return context.external(node, reason="Empty Linear uses native execution in the legacy compiler")
        if any(v.meta.dtype not in (torch.bfloat16, torch.float32) for v in (x, weight) + ((bias,) if bias is not None else ())):
            return context.external(node, reason="BF16 Linear operand dtype is unsupported")
        if x.meta.device is None or torch.device(x.meta.device).type not in ("cpu", "cuda"):
            return context.external(node, reason="BF16 Linear codegen requires CPU/reference or NVIDIA CUDA")
        device = torch.device(x.meta.device).type
        explicit = isinstance(plan, dict) and plan.get("fp8") is False
        active = _get_fp8_plan(device)
        if active is False and isinstance(plan, dict) and "fp8_training" in plan:
            plan = None
        root = getattr(target, "root", module)
        settings = getattr(root, "_deepopt_fp8_config", None)
        if not context.native_autodiff and settings is not None and settings.enabled and settings.backward_precision != "native":
            return context.external(node, reason="Planned gradient transport retains its existing owner")
        if not explicit and (plan is not None or active is not None and active is not False
                             and _active_interception.get().get(device, True)):
            return context.external(node, reason="Scoped FP8 Linear retains its existing executor")
        compression = (_prepare_linear_plan(plan)["activation_compression"] if explicit else
                       _get_activation_compression(device))
        if compression is None:
            compression = getattr(target, "native_compression", None)
        selected = (_prepare_linear_plan(plan) if explicit else
                    _prepare_linear_plan({"fp8": False, "activation_compression": compression or {}}))
        value = context.tensor_value(node)
        if value.meta.dtype != torch.bfloat16 and (not explicit or getattr(module, "native_bf16_only", False)):
            selected = _prepare_linear_plan({"fp8": False, "activation_compression": {}})
        path = node.target if node.op == "call_module" else getattr(target, "path", None)
        lowered = NativeLinearNode(
            "bf16_linear", (x, weight, bias), {}, value, selected, path, node.name,
            bool(getattr(module or getattr(target, "root", None), "training", context.graph_module.training)),
        )
        context.add_node(node, lowered)
        return lowered

    def codegen(node, context):
        if node.value.meta.dtype != torch.bfloat16 or any(prod(value.meta.shape) == 0 for value in node.args[:2]):
            from deepopt.opt.nn.functional import do_linear

            return lambda x, weight, bias: do_linear(x, weight, bias, plan=node.plan)
        return NativeLinearProgram(node, context)

    return OperatorExtension(
        name="bf16_linear", node_type=NativeLinearNode, matcher=match, lowering=lower, pure=False,
        kernel_codegen=codegen,
        training_codegen=codegen,
    )


class _BF16Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, residual, program):
        ctx.program = program
        return program.forward(ctx, x, weight, bias, residual)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return (*ctx.program.backward(ctx, gradient), None)


class NativeLinearProgram:
    # The wrapper collects launch programs separately from pointwise artifacts.
    # generated_kernels below distinguishes this program's actual implementation.
    reuses_existing_kernels = True
    precision = "bf16"

    def __init__(self, node, context):
        from deepopt.kernel_policy import resolve_kernel_policy

        from deepopt._bf16_policy import resolve_bf16_policy

        self.bf16_policy = resolve_bf16_policy(node.plan.get("bf16_policy"))
        self.module_path, self.node_name = node.module_path, node.node_name
        self.module_training, self.plan = node.module_training, node.plan
        self.kernel_policy = resolve_kernel_policy(context.kernel_policy)
        if "kernel_policy" in self.plan:
            from deepopt.kernel_policy import plan_kernel_policy

            if plan_kernel_policy(self.plan) != self.kernel_policy:
                raise ValueError("Compiler kernel policy conflicts with the prepared BF16 Linear policy")
        self.kernel_policy_fingerprint = self.kernel_policy.fingerprint
        self.execution = context.bf16_gemm_backend
        self.decode = context.bf16_decode
        self.backward_order = context.bf16_backward_order
        self.tile, self.num_stages = context.bf16_tile, context.bf16_num_stages
        self.activation, self.epilogue = node.activation, node.epilogue
        x, w, b = node.args[:3]
        self.residual = len(node.args) == 4
        self.training = bool(node.value.meta.requires_grad and torch.is_grad_enabled())
        self.need_dx = self.training and x.meta.requires_grad
        self.activation_state = node.activation_state if self.need_dx else None
        self.activation_module = node.activation_module
        self.need_dw = self.training and w.meta.requires_grad
        self.need_db = self.training and b is not None and b.meta.requires_grad
        self.need_residual = self.training and self.residual and node.args[3].meta.requires_grad
        self.compressed = self.need_dw and self.plan["activation_compression"]["enabled"]
        if (self.compressed and self.decode in ("tiled", "panel")
                and self.kernel_policy.pinned("activation_codec", "linear.dw")
                and self.kernel_policy.route("activation_codec", "linear.dw") != self.kernel_policy.route("gemm", "linear.dw")):
            raise ValueError("Conflicting Mixed routes for BF16 packed-read dW; select materialized decode")
        self.generated_kernels = self.execution == "generated" and torch.device(x.meta.device).type == "cuda"
        self.profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"
        self.kernel_routes = {role: ("native_reduction" if role == "db" else self.kernel_policy.route("gemm", f"linear.{role}")
                                    if self.generated_kernels else "native_library")
                              for role in ("forward", "dx", "dw", "db")}
        from deepopt.opt.backends._shared.bf16_resources import bf16_scratch_requirements

        self.scratch_requirements = (bf16_scratch_requirements(
            w.meta.shape[0], x.meta.shape[-1], prod(x.meta.shape[:-1]),
            route=("direct" if self.decode == "tiled" else self.decode)
            if self.generated_kernels else "materialized") if self.compressed else ())
        self.saved_components = tuple(
            (["x_payload", "codec_step", "codec_minimum"] if self.compressed else ["bf16_input"] if self.need_dw else [])
            + (["weight"] if self.need_dx else [])
            + (["derivative_payload", "derivative_step", "derivative_minimum"] if self.activation_state else
               ["activation_input"] if self.activation and self.need_dx else [])
            + (["preactivation"] if self.epilogue and self.training else [])
        )
        self.forward_steps = tuple(
            (["fused_activation_pack"] if self.activation and self.compressed else
             ["cast_pack_bf16"] if self.compressed and self.execution == "generated" else ["bf16_prepare"])
            + [f"{self.execution}_bf16_gemm" + ("_activation" if self.epilogue else "_residual" if self.residual else "")]
        )
        self.backward_steps = tuple(
            (["activation_backward"] if self.epilogue else [])
            + (["bf16_gemm_dx_activation" if self.activation else "bf16_gemm_dx"] if self.need_dx else [])
            + (["panel_decode_bf16_gemm_dw"] if self.need_dw and self.compressed and self.decode == "panel" else
               ["packed_read_bf16_gemm_dw"] if self.need_dw and self.compressed and self.decode == "tiled" else
               ["decode_bf16", "bf16_gemm_dw"] if self.compressed else ["bf16_gemm_dw"] if self.need_dw else [])
            + (["bias_reduce"] if self.need_db else [])
            + (["residual_gradient"] if self.need_residual else [])
        )
        if self.backward_order == "dw_first":
            prefix = [s for s in self.backward_steps if s == "activation_backward"]
            dx = [s for s in self.backward_steps if s.startswith("bf16_gemm_dx")]
            dw = [s for s in self.backward_steps if s in ("decode_bf16", "bf16_gemm_dw", "packed_read_bf16_gemm_dw", "panel_decode_bf16_gemm_dw")]
            tail = [s for s in self.backward_steps if s in ("bias_reduce", "residual_gradient")]
            self.backward_steps = tuple(prefix + dw + dx + tail)
        self.source = repr({"forward": self.forward_steps, "backward": self.backward_steps,
                            "saved": self.saved_components, "precision": self.precision,
                            "bf16_policy": self.bf16_policy.as_dict()})

    def _scope(self, phase, role):
        from contextlib import ExitStack
        from deepopt.kernel_policy import kernel_policy_scope

        stack = ExitStack()
        stack.enter_context(kernel_policy_scope(self.kernel_policy, role=role))
        if self.profile_ranges or torch.autograd._profiler_enabled():
            suffix = (f".epilogue_{self.epilogue[0]}" if role == "linear.forward" and self.epilogue else
                      f".{self.activation[0]}_pack" if role == "linear.prepare_input" and self.activation and self.compressed else
                      f".{self.decode}_decode" if role == "linear.dw" and self.compressed and self.decode in ("tiled", "panel") else "")
            stack.enter_context(torch.profiler.record_function(
                f"deepopt.compiler.{self.module_path or self.node_name}.{phase}.{role}{suffix}"))
        return stack

    def _backend(self, role):
        return self.kernel_routes[role].replace("native_library", "native")

    def __call__(self, x, weight, bias, residual=None):
        from deepopt.opt._autocast import _suspend_fp8_interception

        with _suspend_fp8_interception():
            return _BF16Function.apply(x, weight, bias, residual, self)

    def forward(self, ctx, x, weight, bias, residual=None):
        from deepopt.opt.backends.bf16 import prepare, product
        from deepopt.opt.quantization.activation import quantize_pack_activation
        from deepopt.opt.nn.autograd._output import restore_fresh_output

        ctx.bf16_policy = self.bf16_policy
        ctx.input_shape, ctx.input_dtype, ctx.weight_dtype = tuple(x.shape), x.dtype, weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        matrix = x.reshape(-1, weight.shape[1])
        compression = self.plan["activation_compression"] if self.compressed else None
        prep_backend = self.kernel_policy.route("compiler_preparation", "compiler.preparation")
        fused_pack = bool(self.activation and compression)
        derivative_saved = None
        with self._scope("forward", "linear.prepare_input"):
            if self.compressed and self.execution == "generated" and not self.activation:
                from deepopt.opt.quantization.activation import _quantize_pack_strided_activation

                # The GEMM rounds its loads to BF16; the codec rounds the same
                # source in registers, avoiding a dense cast buffer for dW saves.
                dense = matrix
                packed = _quantize_pack_strided_activation(matrix, compression["group_size"],
                    bits=compression["bits"], source_dtype=torch.bfloat16)
            else:
                prepared = prepare(matrix, activation=self.activation, compression=compression if fused_pack else None,
                                   backend=prep_backend if fused_pack else "native", derivative_compression=self.activation_state,
                                   bf16_policy=ctx.bf16_policy)
                dense, packed = prepared[:2]
                derivative_saved = prepared[2] if self.activation_state else None
            if self.compressed and packed is None:
                packed = quantize_pack_activation(dense, compression["group_size"], bits=compression["bits"])
        with self._scope("forward", "linear.forward"):
            result = product(dense, weight, backend=self._backend("forward"), bias=bias,
                             activation=self.epilogue, tile=self.tile, num_stages=self.num_stages,
                             return_preactivation=bool(self.epilogue and self.training),
                             residual=residual.reshape(-1, weight.shape[0]) if residual is not None else None,
                             bf16_policy=ctx.bf16_policy)
        if self.epilogue and self.training:
            result, preactivation = result
        else:
            preactivation = None
        ctx.save_for_backward(
            packed.data if packed is not None else dense if self.need_dw else None,
            packed.step if packed is not None else None,
            packed.minimum if packed is not None else None,
            weight if self.need_dx else None,
            derivative_saved.data if derivative_saved is not None else matrix if self.activation and self.need_dx else None,
            preactivation,
            derivative_saved.step if derivative_saved is not None else None,
            derivative_saved.minimum if derivative_saved is not None else None,
        )
        if self.activation_module is not None:
            self.activation_module.last_execution = {"backend": prep_backend, "compiler_fused_linear": True,
                "producer_pack_fused": True, "compression": derivative_saved is not None,
                "activation": self.activation[0], "input": str(x.dtype), "output": str(dense.dtype),
                "input_storage_dtype": str(x.dtype), "output_storage_dtype": str(dense.dtype),
                "saved_state": "derivative" if derivative_saved is not None else "none",
                "saved_state_bytes": sum(t.numel() * t.element_size() for t in
                    (derivative_saved.data, derivative_saved.step, derivative_saved.minimum)) if derivative_saved is not None else 0}
        return restore_fresh_output(result, lambda y: y.reshape(*x.shape[:-1], weight.shape[0]))

    def backward(self, ctx, gradient):
        from deepopt.opt.backends.bf16 import derivative
        from deepopt.opt._autocast import _suspend_fp8_interception

        data, step, minimum, weight, activation_input, preactivation, dstep, dminimum = ctx.saved_tensors
        grad = gradient.reshape(-1, gradient.shape[-1]).to(torch.bfloat16)
        if self.epilogue:
            grad = derivative(grad, preactivation, self.epilogue)
        dx = dw = db = None
        with _suspend_fp8_interception(), torch.autocast(gradient.device.type, enabled=False):
            if self.backward_order == "dw_first":
                dw = self.weight_gradient(ctx, grad, data, step, minimum)
                dx = self.input_gradient(ctx, grad, weight, activation_input, dstep, dminimum)
            else:
                dx = self.input_gradient(ctx, grad, weight, activation_input, dstep, dminimum)
                dw = self.weight_gradient(ctx, grad, data, step, minimum)
            if self.need_db:
                with self._scope("backward", "linear.db"):
                    db = grad.sum(dim=0).to(ctx.bias_dtype)
        return dx, dw, db, gradient if self.need_residual else None

    def input_gradient(self, ctx, grad, weight, activation_input, dstep, dminimum):
        if not self.need_dx:
            return None
        from deepopt.opt.backends.bf16 import product
        from deepopt.opt.quantization.activation import PackedActivation

        saved = (PackedActivation(activation_input, dstep, dminimum, (grad.shape[0], ctx.input_shape[-1]),
            self.activation_state["group_size"], bits=self.activation_state["bits"]) if self.activation_state else None)
        with self._scope("backward", "linear.dx"):
            return product(grad, weight.T, backend=self._backend("dx"),
                derivative_input=activation_input if saved is None else None,
                derivative_activation=self.activation if saved is None else None, derivative_packed=saved,
                tile=self.tile, num_stages=self.num_stages, bf16_policy=ctx.bf16_policy).reshape(ctx.input_shape).to(ctx.input_dtype)

    def weight_gradient(self, ctx, grad, data, step, minimum):
        if not self.need_dw:
            return None
        from deepopt.opt.backends.bf16 import product
        from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

        with self._scope("backward", "linear.dw"):
            packed = None
            if self.compressed:
                options = self.plan["activation_compression"]
                packed = PackedActivation(data, step, minimum,
                    (grad.shape[0], ctx.input_shape[-1]), options["group_size"], bits=options["bits"])
            backend = self._backend("dw")
            panel = packed is not None and self.decode == "panel" and backend == "triton"
            packed_read = packed is not None and (self.decode == "tiled" or panel)
            operand = None if packed_read else (dequantize_unpack_activation(packed, dtype=torch.bfloat16, transpose=True)
                                          if packed is not None else data.T)
            return product(grad.T, operand, packed_b=packed if packed_read else None,
                backend=backend, packed_route="panel" if panel else "direct", panel_size=1024,
                tile=self.tile, num_stages=self.num_stages, bf16_policy=ctx.bf16_policy,
                operand_route=(self.decode if packed_read else "materialized")
                              if packed is not None else "dense").to(ctx.weight_dtype)


def fuse_native(graph, *, enabled, unsafe_values, config, kernel_policy, diagnostics):
    """Select BF16 producer-pack or GEMM epilogues without crossing effects."""
    from .ir import GetAttrNode, PointwiseNode, register_uses
    from .scheduler import _is_contiguous
    from .linear import _fusion_signature
    from .native_stateful import NativeStatefulNode

    removed, patterns = set(), []
    for node in list(graph.nodes):
        if not isinstance(node, NativeLinearNode):
            continue
        x = node.args[0]
        producer = x.producer
        stateful = (isinstance(producer, NativeStatefulNode) and producer.op in ("native_gelu", "native_silu"))
        activation = stateful or isinstance(producer, PointwiseNode) and producer.op in ("gelu", "silu")
        kind = producer.module.kind if stateful else producer.op if activation else None
        compression = node.plan["activation_compression"]
        candidate = {"module_path": node.module_path, "node_name": node.node_name,
                     "phase": "train" if node.module_training else "eval", "precision": "bf16",
                     "requested": enabled, "effective_enabled": enabled, "discovered": activation,
                     "eligible": None, "selected": False, "status": "disabled" if not enabled else "rejected",
                     "reason": "disabled" if not enabled else "no_activation", "selected_schedule": None,
                     "input_signature": _fusion_signature(x), "observed_cuda_launches": None}
        diagnostics.append(candidate)
        if not enabled:
            continue
        # BF16 activation-output/pack avoids an extra pack pass, while pure
        # BF16 prefers a producer GEMM epilogue below.
        if activation and compression["enabled"] and node.args[1].meta.requires_grad and node.value.meta.requires_grad:
            if stateful:
                if "kernel_policy" in producer.module.plan:
                    from deepopt.kernel_policy import plan_kernel_policy

                    if plan_kernel_policy(producer.module.plan) != kernel_policy:
                        raise ValueError("Compiler kernel policy conflicts with the prepared activation policy")
                state = producer.module.plan["activation_compression"]
                if not state["enabled"] or any(state[key] != compression[key] for key in ("group_size", "bits")):
                    candidate["eligible"], candidate["reason"] = False, "derivative_codec_contract"
                    continue
            source = producer.args[0]
            between = graph.nodes[graph.nodes.index(producer) + 1:graph.nodes.index(node)]
            legal = (x.users == [node] and x not in unsafe_values and source not in unsafe_values
                     and source.meta is not None and source.meta.dtype == torch.bfloat16
                     and _is_contiguous(source.meta) and source.meta.requires_grad == x.meta.requires_grad
                     and torch.device(source.meta.device).type == "cuda" and compression["group_size"] <= 16384
                     and all(isinstance(item, GetAttrNode) for item in between))
            if not legal:
                candidate["eligible"], candidate["reason"] = False, "layout_effects_or_fanout"
                continue
            preparation = kernel_policy.route("compiler_preparation", "compiler.preparation")
            activation_route = kernel_policy.route("activation", "activation.forward") if stateful else kernel_policy.route("compiler_pointwise", f"compiler.{kind}")
            if (activation_route != preparation
                    or kernel_policy.pinned("activation_codec", "linear.prepare_input")
                    and kernel_policy.route("activation_codec", "linear.prepare_input") != preparation
                    or source.meta.requires_grad and kernel_policy.pinned("compiler_pointwise", "compiler.backward")
                    and kernel_policy.route("compiler_pointwise", "compiler.backward") != kernel_policy.route("gemm", "linear.dx")):
                candidate["eligible"], candidate["reason"] = False, "backend_conflict"
                continue
            if (stateful and source.meta.requires_grad and config.bf16_gemm_backend == "generated"
                    and (kernel_policy.pinned("activation", "activation.backward")
                         and kernel_policy.route("activation", "activation.backward") != kernel_policy.route("gemm", "linear.dx")
                         or kernel_policy.pinned("activation_codec", "activation.backward")
                         and kernel_policy.route("activation_codec", "activation.backward") != kernel_policy.route("gemm", "linear.dx"))):
                candidate["eligible"], candidate["reason"] = False, "backend_conflict"
                continue
            node.activation = kind, producer.module.approximate if stateful else producer.kwargs.get("approximate", "none")
            if stateful:
                node.activation_state, node.activation_module = dict(state), producer.module
            node.args = (source, *node.args[1:])
            removed.add(id(producer))
            name = f"{kind} -> BF16 Linear saved-input packing" + (" + packed derivative" if stateful else "")
        elif config.bf16_gemm_backend == "generated":
            users = node.value.users
            consumer = users[0] if len(users) == 1 else None
            residual = None
            if isinstance(consumer, PointwiseNode) and consumer.op == "add" and len(consumer.args) == 2:
                residual = consumer.args[1] if consumer.args[0] is node.value else consumer.args[0]
            activation_args = isinstance(consumer, PointwiseNode) and consumer.args == (node.value,)
            residual_args = (isinstance(residual, TensorValue) and residual.producer is not node and residual not in unsafe_values
                             and residual.meta is not None and _is_contiguous(residual.meta)
                             and residual.meta.shape == node.value.meta.shape and residual.meta.dtype == torch.bfloat16)
            legal = (isinstance(consumer, PointwiseNode) and consumer.op in ("gelu", "silu", "add")
                     and node.value not in unsafe_values and consumer.value not in unsafe_values
                     and (activation_args or residual_args) and node.value.meta.dtype == consumer.value.meta.dtype
                     and torch.device(node.value.meta.device).type == "cuda"
                     and all(isinstance(item, GetAttrNode) for item in
                             graph.nodes[graph.nodes.index(node) + 1:graph.nodes.index(consumer)]))
            if not legal:
                continue
            # Prefer one producer-output/pack kernel over an epilogue followed
            # by a separate consumer pack when the next Linear needs that save.
            next_users = consumer.value.users
            if (len(next_users) == 1 and isinstance(next_users[0], NativeLinearNode)
                    and next_users[0].plan["activation_compression"]["enabled"]
                    and next_users[0].args[1].meta.requires_grad):
                candidate["reason"] = "consumer_pack_preferred"
                continue
            if kernel_policy.route("compiler_pointwise", f"compiler.{consumer.op}") != kernel_policy.route("gemm", "linear.forward"):
                candidate["eligible"], candidate["reason"] = False, "backend_conflict"
                continue
            if consumer.op == "add":
                node.args = (*node.args, residual)
            else:
                node.epilogue = consumer.op, consumer.kwargs.get("approximate", "none")
            node.value = consumer.value
            node.value.producer = node
            removed.add(id(consumer))
            candidate["discovered"] = True
            name = f"BF16 GEMM -> {consumer.op} epilogue"
        else:
            continue
        patterns.append(name)
        candidate.update(eligible=True, selected=True, status="selected", reason=None, selected_schedule=name)
    if removed:
        graph.nodes[:] = [n for n in graph.nodes if id(n) not in removed]
        for n in graph.nodes:
            for value in n.outputs():
                value.users.clear()
        for n in graph.nodes:
            register_uses(n)
        graph.validate()
    return tuple(patterns)
