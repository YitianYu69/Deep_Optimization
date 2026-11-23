from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..ir import (
    AuxiliaryTensorMeta,
    ExternKernelNode,
    GetAttrNode,
    IRNode,
    InputNode,
    OutputNode,
    PointwiseNode,
    TensorValue,
)
from ..registry import BackendRegistry, KernelCodegenContext
from ..scheduler import FusionGroup, Schedule, SinglePointwiseOp, scheduled_outputs
from ..runtime.memory import MemoryManager
from .runtime import launch_pointwise


def fetch_attr(root: Any, target: str) -> Any:
    """Resolve a GraphModule attribute path such as ``layer1.weight``."""
    value = root
    for part in target.split("."):
        value = getattr(value, part)
    return value


def intermediate_dtype_error(value, contract, *, storage=False):
    """Describe a failed edge using metadata only, including non-tensor results."""
    actual = {"type": f"{type(value).__module__}.{type(value).__qualname__}",
              "dtype": None, "shape": None}
    if isinstance(value, torch.Tensor):
        actual.update(dtype=str(value.dtype), shape=list(value.shape))
        if storage:
            backing = getattr(value, "_data", value)
            actual["storage_dtype"] = str(backing.dtype) if isinstance(backing, torch.Tensor) else None
    details = {**contract, "actual": actual, "check": "storage_dtype" if storage else "dtype"}
    error = TypeError(
        f"Intermediate {contract['value']!r} violates captured dtype {contract['expected']['dtype']}; "
        f"operation={contract['operation']!r}, module={contract['module']!r}, stage={contract['stage']!r}, "
        f"source_node={contract.get('source_node')!r}, "
        f"expected={contract['expected']}, actual={actual}, "
        f"bf16_policy={contract['bf16_policy']}, primitive_recipe={contract['primitive_recipe']}"
    )
    error.dtype_contract = details
    return error


@dataclass
class WrapperArtifact:
    fn: Any
    source: str
    kernels: list[Any]
    triton_artifacts: list[Any]
    tilelang_artifacts: list[Any] = field(default_factory=list)
    custom_kernels: list[Any] = field(default_factory=list)
    autograd_regions: list[Any] = field(default_factory=list)
    native_autograd_regions: list[Any] = field(default_factory=list)
    eager_autograd_ops: tuple[str, ...] = ()
    deepopt_autograd_ops: tuple[str, ...] = ()
    deepopt_programs: list[Any] = field(default_factory=list)
    kernel_routes: tuple[dict, ...] = ()
    joint_autograd_regions: list[Any] = field(default_factory=list)
    legacy_backward_operations: tuple[str, ...] = ()
    final_schedule: tuple[dict, ...] = ()
    logical_lifetimes: tuple[dict, ...] = ()
    scratch_lifetimes: tuple[dict, ...] = ()
    observed_fn: Any = None
    reuse_fn: Any = None
    reusable_values: tuple[str, ...] = ()
    boundary_metadata: tuple[dict, ...] = ()
    native_region_index: int | None = None
    native_scope_identity: str | None = None
    native_forward_signature: str | None = None
    native_backward_signature: str | None = None
    native_recipe_identity_sha256: str | None = None
    native_profile_unavailable_reason: str | None = None


class WrapperCodegen:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        schedule: Schedule,
        *,
        registry: BackendRegistry | None = None,
        cache_dir: str | Path | None = None,
        block_size: int = 1024,
        enable_autodiff: bool = True,
        require_full_aot: bool = False,
        unsafe_values: frozenset[TensorValue] = frozenset(),
        kernel_policy=None,
        bf16_options=None,
        consume_inputs: bool = False,
        gradient_input_count: int = 0,
        diagnostic_phase: str = "forward",
        packed_offload_plan=None,
    ):
        self.gm = gm
        self.schedule = schedule
        self.registry = registry if registry is not None else BackendRegistry()
        from deepopt.kernel_policy import resolve_kernel_policy

        self.kernel_policy = resolve_kernel_policy(kernel_policy)
        self.block_size = block_size
        self.cache_dir = cache_dir
        self.kernel_codegen_context = KernelCodegenContext(
            backend=self.kernel_policy.backend,
            block_size=block_size,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            kernel_policy=self.kernel_policy,
            **(bf16_options or {}),
        )
        self.kernels: list[Any] = []
        self.custom_kernels: list[Any] = []
        self.deepopt_programs: list[Any] = []
        self.triton_artifacts: list[Any] = []
        self.tilelang_artifacts: list[Any] = []
        self.autograd_regions: list[Any] = []
        self.joint_autograd_regions: list[Any] = []
        self.native_autograd_regions: list[Any] = []
        self.native_autograd_indices: dict[int, int] = {}
        self.enable_autodiff = enable_autodiff
        self.consume_inputs = consume_inputs
        self.gradient_input_count = gradient_input_count
        self.diagnostic_phase = diagnostic_phase
        self.packed_offload_plan = packed_offload_plan if packed_offload_plan is not None and packed_offload_plan.bundles else None
        components = (component for bundle in self.packed_offload_plan.bundles for component in bundle.components) if self.packed_offload_plan else ()
        self.offload_components = {component.name: component for component in components}
        self.offload_bundles = {bundle.bundle_id: bundle for bundle in self.packed_offload_plan.bundles} if self.packed_offload_plan else {}
        self.offload_consumers = {consumer.schedule_index: consumer for consumer in self.packed_offload_plan.consumers} if self.packed_offload_plan else {}
        saved_indices = {component.saved_index for component in self.offload_components.values()}
        self.offload_backward_inputs = {
            item.value: item.index - gradient_input_count for item in schedule.items
            if isinstance(item, InputNode) and diagnostic_phase == "backward"
            and item.index - gradient_input_count in saved_indices
        }
        runtime_exports = frozenset(value for item in schedule.items for value in scheduled_outputs(item)
                                   if value.name in self.offload_components) if diagnostic_phase == "forward" else frozenset()
        self.require_full_aot = require_full_aot
        self.unsafe_values = unsafe_values
        self.runtime_memory_manager = MemoryManager(schedule=schedule, runtime_exports=runtime_exports)
        self.runtime_memory_manager.collect_schedule_level_remaining_users()
        self.native_ops: list[Any] = []
        self.consts: list[Any] = []
        self.runtime_names: dict[TensorValue, str] = {}
        self.lines = ["def compiled(args):" if consume_inputs else "def compiled(*args):"]
        if self.packed_offload_plan is not None:
            self.lines[0] = "def compiled(args, *, _offload):" if consume_inputs else "def compiled(*args, _offload):"
        self.next_var_id = 0
        self.input_count = max(
            (item.index + 1 for item in schedule.items if isinstance(item, InputNode)),
            default=0,
        )
        self.lines.append(
            f"    if len(args) != {self.input_count}:\n"
            f"        raise TypeError('compiled graph expects {self.input_count} positional inputs')"
        )

    def fresh_var(self) -> str:
        name = f"v{self.next_var_id}"
        self.next_var_id += 1
        return name

    def bind(self, value: TensorValue, runtime_name: str) -> None:
        self.runtime_names[value] = runtime_name

    def lookup(self, value: TensorValue) -> str:
        try:
            return self.runtime_names[value]
        except KeyError as exc:
            raise RuntimeError(
                f"No runtime value exists for TensorValue({value.name})"
            ) from exc

    def constant(self, value: Any) -> str:
        index = len(self.consts)
        self.consts.append(value)
        return f"_consts[{index}]"

    def intermediate_dtype_guards(self, values):
        """Check actual typed results without casts, copies or device reads."""
        from deepopt._bf16_policy import STORAGE_PRIMITIVE_RECIPE_VERSION, resolve_bf16_policy

        lines = []
        for value in values:
            meta = value.meta
            if meta is None or meta.dtype is None:
                continue
            runtime = self.lookup(value)
            optional = isinstance(meta, AuxiliaryTensorMeta) and meta.presence == "optional"
            indent = "    "
            if optional:
                lines.append(f"    if {runtime} is not None:")
                indent += "    "
            producer = value.producer
            policy = resolve_bf16_policy(getattr(producer, "bf16_policy", None))
            contract = self.constant({"value": value.name,
                "source_node": getattr(producer, "source_node", None),
                "operation": getattr(producer, "op", type(producer).__name__),
                "module": getattr(producer, "module_path", None), "stage": self.diagnostic_phase,
                "expected": {"type": "torch.Tensor", "dtype": str(meta.dtype),
                             "shape": list(meta.shape) if meta.shape is not None else None,
                             "storage_dtype": str(getattr(meta, "storage_dtype", None))},
                "bf16_policy": policy.as_dict(),
                "primitive_recipe": STORAGE_PRIMITIVE_RECIPE_VERSION if policy.storage_enabled else None})
            expected = self.constant(meta.dtype)
            lines.extend((f"{indent}if not isinstance({runtime}, torch.Tensor) or {runtime}.dtype != {expected}:",
                          f"{indent}    raise _intermediate_dtype_error({runtime}, {contract})"))
            storage_dtype = getattr(meta, "storage_dtype", None)
            if storage_dtype is not None:
                storage = self.constant(storage_dtype)
                lines.extend((f"{indent}if getattr({runtime}, '_data', {runtime}).dtype != {storage}:",
                              f"{indent}    raise _intermediate_dtype_error({runtime}, {contract}, storage=True)"))
        return lines

    def emit_runtime_value(self, value: Any) -> str:
        if isinstance(value, TensorValue):
            return self.lookup(value)
        if isinstance(value, tuple):
            items = [self.emit_runtime_value(item) for item in value]
            body = ", ".join(items)
            return f"({body}{',' if len(items) == 1 else ''})"
        if isinstance(value, list):
            return (
                "[" + ", ".join(self.emit_runtime_value(item) for item in value) + "]"
            )
        if isinstance(value, dict):
            items = [
                f"{self.constant(key)}: {self.emit_runtime_value(item)}"
                for key, item in value.items()
            ]
            return "{" + ", ".join(items) + "}"
        return self.constant(value)

    def codegen_inputs(self, node: InputNode) -> None:
        runtime_name = self.fresh_var()
        if node.value in self.offload_backward_inputs:
            # The argument is a canonical host save. Bind and guard its restored
            # CUDA value only at a numerical consumer, not at backward entry.
            self.bind(node.value, runtime_name)
            return
        self.lines.append(f"    {runtime_name} = args[{node.index}]")
        self.guard_tensor(node.value.meta, runtime_name, f"Input {node.value.name!r}",
                          allow_fp8=node.index < self.gradient_input_count)
        self.bind(node.value, runtime_name)
        if self.consume_inputs:
            self.lines.append(f"    args[{node.index}] = None")

    def guard_tensor(self, meta, runtime_name, label, *, allow_fp8=False):
        # Only explicitly typed auxiliary results can carry backend absence.
        # Keep every existing check for present tensors and dense inputs.
        optional = isinstance(meta, AuxiliaryTensorMeta) and meta.presence == "optional"
        guard_start = len(self.lines)
        if meta is not None:
            message = self.constant(
                f"{label} does not match the compiled tensor metadata; recompile the graph"
            )
            self.lines.append(f"    if not isinstance({runtime_name}, torch.Tensor):")
            self.lines.append(f"        raise TypeError({message})")
            if meta.representation == "dense":
                from torch._subclasses.fake_tensor import FakeTensor

                allowed = (torch.Tensor, torch.nn.Parameter, FakeTensor)
                if allow_fp8:
                    from deepopt.opt.quantization.activation_tensor import FP8Activation

                    allowed += (FP8Activation,)
                dense_types = self.constant(allowed)
                self.lines.append(f"    if type({runtime_name}) not in {dense_types}:")
                self.lines.append(f"        raise TypeError({message})")
            if meta.shape is not None:
                shape = self.constant(tuple(meta.shape))
                self.lines.append(f"    if tuple({runtime_name}.shape) != {shape}:")
                self.lines.append(f"        raise ValueError({message})")

            if meta.dtype is not None:
                dtype = self.constant(meta.dtype)
                self.lines.append(f"    if {runtime_name}.dtype != {dtype}:")
                self.lines.append(f"        raise ValueError({message})")
            if meta.device is not None:
                device = self.constant(torch.device(meta.device))
                self.lines.append(
                    f"    if {runtime_name}.device.type != {device}.type or "
                    f"({device}.index is not None and {runtime_name}.device.index != {device}.index):"
                )
                self.lines.append(f"        raise ValueError({message})")
            if meta.stride is not None:
                # Singleton strides do not contribute to any addressed element.
                # Factories such as randn_like can preserve a narrowed view's
                # singleton stride while FakeTensor reports its compact form.
                # Guard every observable stride; accepting an arbitrary layout
                # on a non-singleton axis would invalidate generated kernels.
                axes = tuple(index for index, size in enumerate(meta.shape) if size > 1)
                stride = self.constant(tuple(meta.stride[index] for index in axes))
                runtime_stride = ", ".join(f"{runtime_name}.stride({axis})" for axis in axes)
                self.lines.append(f"    if ({runtime_stride}{',' if len(axes) == 1 else ''}) != {stride}:")
                self.lines.append(f"        raise ValueError({message})")
            if self.enable_autodiff:
                self.lines.append(
                    f"    if {runtime_name}.requires_grad != {meta.requires_grad!r}:"
                )
                self.lines.append(f"        raise ValueError({message})")

        if optional:
            guards = self.lines[guard_start:]
            self.lines[guard_start:] = [f"    if {runtime_name} is not None:",
                                       *("    " + line for line in guards)]

    def codegen_get_attr(self, node: GetAttrNode) -> None:
        runtime_name = self.fresh_var()
        target_const = self.constant(node.target)
        self.lines.append(f"    {runtime_name} = _fetch_attr(_gm, {target_const})")
        self.guard_tensor(node.value.meta, runtime_name, f"Attribute {node.target!r}")
        self.bind(node.value, runtime_name)

    def emit_native_call(self, target: Any, args: Any, kwargs: Any) -> str:
        if target is None or not callable(target):
            raise TypeError(f"Native fallback target must be callable, got {target!r}")
        op_index = len(self.native_ops)
        self.native_ops.append(target)
        arg_exprs = [self.emit_runtime_value(value) for value in args]
        if kwargs:
            arg_exprs.append(f"**{self.emit_runtime_value(kwargs)}")
        return f"_native_ops[{op_index}]({', '.join(arg_exprs)})"

    def codegen_extern(self, node: ExternKernelNode) -> None:
        native_region_index = self.native_autograd_indices.get(id(node))
        if native_region_index is not None:
            runtime_inputs = [
                self.lookup(value) for value in dict.fromkeys(node.inputs())
            ]
            runtime_name = self.fresh_var()
            self.lines.append(
                f"    {runtime_name} = _native_autograd_regions[{native_region_index}]"
                f"({', '.join(runtime_inputs)})"
            )
            if node.value is None:
                return
            self.bind(node.value, runtime_name)
            return

        # External calls outside an AOT layer boundary keep ordinary PyTorch
        # autograd, including operations with unsupported output structures.
        call_expr = self.emit_native_call(node.target, node.args, node.kwargs)
        if node.value is None:
            self.lines.append(f"    {call_expr}")
            return
        runtime_name = self.fresh_var()
        self.lines.append(f"    {runtime_name} = {call_expr}")
        self.bind(node.value, runtime_name)

    def codegen_native_node(self, node: IRNode, *, panel_consumer=None) -> None:
        native_region_index = self.native_autograd_indices.get(id(node))
        if native_region_index is not None:
            runtime_inputs = [
                self.lookup(value) for value in dict.fromkeys(node.inputs())
            ]
            runtime_name = self.fresh_var()
            self.lines.append(
                f"    {runtime_name} = _native_autograd_regions[{native_region_index}]"
                f"({', '.join(runtime_inputs)})"
            )
            self.bind(node.output(), runtime_name)
            return
        target = getattr(node, "target", None)
        args = getattr(node, "args", None)
        kwargs = getattr(node, "kwargs", None)
        outputs = node.outputs()
        output = outputs[0] if len(outputs) == 1 else None
        extension = self.registry.for_node(node)
        if extension is not None and (extension.kernel_codegen is not None or extension.training_codegen is not None):
            needs_grad = self.enable_autodiff and (
                any(value.meta is None or value.meta.requires_grad for value in outputs)
                and any(
                    value.meta is None or value.meta.requires_grad
                    for value in node.inputs()
                )
            )
            compiler = extension.training_codegen if needs_grad else extension.kernel_codegen
            if compiler is not None:
                target = compiler(node, self.kernel_codegen_context)
                if not callable(target):
                    raise TypeError(
                        f"Kernel codegen for {extension.name!r} must return a callable"
                    )
                if getattr(target, "reuses_existing_kernels", False):
                    self.deepopt_programs.append(target)
                else:
                    self.custom_kernels.append(target)
            elif needs_grad:
                if extension.native_fallback is None:
                    raise NotImplementedError(
                        f"Standalone kernel {extension.name!r} requires native_fallback for training; "
                        "use a registered pointwise backward rule for compiler-owned fusion"
                    )
                target = lambda *args, **kwargs: extension.native_fallback(
                    node, args, kwargs
                )
        if target is None or args is None or kwargs is None or not outputs:
            raise NotImplementedError(
                f"Cannot compile custom node {type(node).__name__} outside pointwise kernels: "
                "it must expose args, kwargs, and outputs, and define kernel_codegen "
                "or a callable target"
            )
        if panel_consumer is not None:
            panel = getattr(target, "packed_offload_panel", None)
            if kwargs or not callable(getattr(panel, "allocate_outputs", None)) or not callable(getattr(panel, "run", None)):
                raise ValueError("Packed panel consumer requires its captured native panel ABI")
            ordinary = [value for value in args if not isinstance(value, TensorValue)
                        or value not in self.offload_backward_inputs]
            if len(ordinary) != 1:
                raise ValueError("SwiGLU packed panel consumer requires exactly one dense cotangent")
            op_index = len(self.native_ops)
            self.native_ops.append(target)
            call_expr = (f"_offload.run_panel_consumer({panel_consumer.consumer_id!r}, _native_ops[{op_index}], "
                         f"({self.emit_runtime_value(ordinary[0])},))")
        else:
            call_expr = self.emit_native_call(target, args, kwargs)
        runtime_name = self.fresh_var()
        self.lines.append(f"    {runtime_name} = {call_expr}")
        if output is not None:
            self.bind(output, runtime_name)
        else:
            self.lines.append(
                f"    if not isinstance({runtime_name}, (tuple, list)) or len({runtime_name}) != {len(outputs)}:"
            )
            self.lines.append(
                "        raise TypeError('custom kernel results do not match the declared output schema')"
            )
            for index, value in enumerate(outputs):
                name = self.fresh_var()
                self.lines.append(f"    {name} = {runtime_name}[{index}]")
                self.bind(value, name)
            self.lines.append(f"    del {runtime_name}")

    def codegen_pointwise(self, artifact_index: int) -> None:
        artifact = self.artifacts[artifact_index]
        if getattr(artifact, "backend", "triton") == "tilelang":
            self.tilelang_artifacts.append(artifact)
        else:
            self.triton_artifacts.append(artifact)
        self.kernels.append(artifact.kernel)
        input_names = [self.lookup(value) for value in artifact.inputs]
        if not input_names:
            raise RuntimeError("Triton partition has no runtime tensor inputs")
        for value, name in zip(artifact.inputs, input_names):
            self.guard_tensor(value.meta, name, f"Kernel operand {value.name!r}")

        output = artifact.output
        if output.meta is None:
            raise ValueError(f"Output {output.name!r} has no metadata")
        for input_name in input_names[1:]:
            self.lines.append(f"    if {input_name}.device != {input_names[0]}.device:")
            self.lines.append(
                "        raise ValueError('Fused kernel inputs must be on one device')"
            )

        if self.enable_autodiff:
            output_name = self.fresh_var()
            self.lines.append(
                f"    {output_name} = _autograd_regions[{artifact_index}]"
                f"({', '.join(input_names)})"
            )
            self.bind(output, output_name)
            return

        output_name = self.fresh_var()
        shape_expr = self.constant(tuple(output.meta.shape))
        dtype_expr = self.constant(output.meta.dtype)
        self.lines.append(
            f"    {output_name} = torch.empty({shape_expr}, dtype={dtype_expr}, "
            f"device={input_names[0]}.device)"
        )
        launch_indent = "    "
        if output.meta.device is not None and torch.device(output.meta.device).type == "cuda":
            self.lines.append(f"    with torch.cuda.device({output_name}.device):")
            launch_indent += "    "
        self.lines.append(
            f"{launch_indent}_launch_kernel(_artifacts[{artifact_index}], "
            f"({', '.join(input_names)},), {output_name}, {self.block_size})"
        )
        self.bind(output, output_name)

    def codegen_output(self, node: OutputNode) -> None:
        if self.packed_offload_plan is not None and self.diagnostic_phase == "forward":
            outputs, saved = node.result
            exported = [f"_offload.export_slot({index})" if value.name in self.offload_components
                        else self.emit_runtime_value(value) for index, value in enumerate(saved)]
            saved_expression = "(" + ", ".join(exported) + ("," if len(exported) == 1 else "") + ")"
            self.lines.append(f"    return ({self.emit_runtime_value(outputs)}, {saved_expression})")
            return
        self.lines.append(f"    return {self.emit_runtime_value(node.result)}")

    def offload_before(self, index, item):
        if self.packed_offload_plan is None or self.diagnostic_phase != "backward":
            return
        from ..scheduler import scheduled_inputs

        if index in self.packed_offload_plan.boundary_ids:
            self.lines.append(f"    _offload.advance({index})")
        if isinstance(item, (InputNode, OutputNode)):
            return
        consumer = self.offload_consumers.get(index)
        panel = getattr(consumer, "panel", None)
        for value in dict.fromkeys(scheduled_inputs(item)):
            saved_index = self.offload_backward_inputs.get(value)
            if saved_index is None:
                continue
            if panel is not None:
                if self.consume_inputs:
                    self.lines.append(f"    args[{saved_index + self.gradient_input_count}] = None")
                continue
            runtime_name = self.lookup(value)
            self.lines.append(f"    {runtime_name} = _offload.resolve({saved_index})")
            self.guard_tensor(value.meta, runtime_name, f"Input {value.name!r}")
            if self.consume_inputs:
                self.lines.append(f"    args[{saved_index + self.gradient_input_count}] = None")

    def offload_publish(self, index):
        if self.packed_offload_plan is None or self.diagnostic_phase != "forward":
            return
        for boundary, bundle_id in self.packed_offload_plan.forward_publish_at:
            if boundary != index:
                continue
            bundle = self.offload_bundles[bundle_id]
            by_name = {value.name: value for value in self.runtime_names}
            values = ", ".join(self.lookup(by_name[component.name]) for component in bundle.components)
            self.lines.append(f"    _offload.pack_ready({bundle_id!r}, ({values},))")

    def offload_after(self, index, item):
        if self.packed_offload_plan is None:
            return
        if self.diagnostic_phase == "forward":
            for boundary, bundle_id in self.packed_offload_plan.forward_release_at:
                if boundary == index:
                    self.lines.append(f"    _offload.last_forward_use({bundle_id!r})")
        else:
            from ..scheduler import scheduled_inputs

            consumer = self.offload_consumers.get(index)
            if getattr(consumer, "panel", None) is not None:
                return  # The panel ABI owns its compact staging/retirement.
            if not isinstance(item, (InputNode, OutputNode)):
                for value in dict.fromkeys(scheduled_inputs(item)):
                    if value in self.offload_backward_inputs:
                        self.lines.append(f"    del {self.lookup(value)}")
            if consumer is not None:
                for bundle_id in consumer.bundle_ids:
                    self.lines.append(f"    _offload.release_after({bundle_id!r}, {consumer.consumer_id!r})")

    def run(self) -> WrapperArtifact:
        self.runtime_memory_manager.validate_current()
        from ..autodiff.joint import JointRegionNode

        self.joint_autograd_regions = [item.target for item in self.schedule.items if isinstance(item, JointRegionNode)]
        if any(getattr(region, "_offload_candidates", None) is not None for region in self.joint_autograd_regions):
            from ..model_offload_plan import finalize_model_offload_regions

            finalize_model_offload_regions(self.joint_autograd_regions)
        for index, region in enumerate(self.joint_autograd_regions):
            if getattr(region, "set_native_scope_metadata", None) is not None:
                region.set_native_scope_metadata(index, getattr(region.forward_artifact, "native_scope_identity", None))
            self.deepopt_programs.extend(spec for spec in region.specifications if hasattr(spec, "forward_steps"))
        partitions = [
            item
            for item in self.schedule.items
            if isinstance(item, (SinglePointwiseOp, FusionGroup))
        ]
        from deepopt.kernel_policy import kernel_backend

        self.artifacts = []
        kernel_routes = []
        for partition in partitions:
            node = partition.node if isinstance(partition, SinglePointwiseOp) else partition.nodes[0]
            backend = getattr(partition, "kernel_backend", None)
            if backend is None:
                backend = kernel_backend(
                    "compiler_pointwise", backend=self.kernel_policy,
                    role=getattr(partition, "kernel_role", f"compiler.{node.op}"),
                )
            nodes = [partition.node] if isinstance(partition, SinglePointwiseOp) else partition.nodes
            for routed_node in nodes:
                role = getattr(partition, "kernel_role", f"compiler.{routed_node.op}")
                requested = kernel_backend("compiler_pointwise", backend=self.kernel_policy, role=role)
                kernel_routes.append({
                    "operation": routed_node.op, "role": role,
                    "requested_backend": requested, "effective_backend": backend,
                    "fallback_reason": "requested expression emitter unavailable" if requested != backend else None,
                })
            if backend == "tilelang":
                from deepopt._bf16_policy import resolve_bf16_policy

                if any(resolve_bf16_policy(getattr(node, "bf16_policy", None)).storage_enabled for node in nodes):
                    raise ValueError("bf16_storage TileLang pointwise arithmetic is not qualified")
                from .tilelang.pointwise import TileLangCodegen

                emitter = TileLangCodegen(
                    registry=self.registry, cache_dir=self.cache_dir,
                    block_size=self.block_size, kernel_policy=self.kernel_policy,
                )
            else:
                from .triton.pointwise import TritonCodegen

                emitter = TritonCodegen(
                    registry=self.registry, cache_dir=self.cache_dir, block_size=self.block_size,
                    kernel_policy=self.kernel_policy if self.kernel_policy.backend == "mixed" else None,
                )
            self.artifacts.extend(emitter.compile_partitions([partition]))
        if self.enable_autodiff:
            from ..autodiff_runtime import build_autograd_region

            self.autograd_regions = [
                build_autograd_region(
                    partition,
                    artifact,
                    self.gm,
                    registry=self.registry,
                    cache_dir=self.cache_dir,
                    block_size=self.block_size,
                    kernel_policy=self.kernel_policy,
                )
                for partition, artifact in zip(partitions, self.artifacts)
            ]

            from ..autodiff_runtime import (
                build_native_autograd_region,
                is_native_layer_node,
                is_native_pointwise_node,
            )

            for item in self.schedule.items:
                extension = (
                    self.registry.for_node(item) if isinstance(item, IRNode) else None
                )
                if extension is not None and (extension.kernel_codegen is not None or extension.training_codegen is not None):
                    continue
                if not (
                    is_native_layer_node(item)
                    or is_native_pointwise_node(item, self.registry)
                ):
                    continue
                if isinstance(item, ExternKernelNode):
                    from ..autodiff.native_layer_backward_rules import _native_layer_kind

                    # The generic LayerNorm derivative recomputes a complete
                    # forward just to recover statistics. Native autograd
                    # already retains those statistics without that extra pass.
                    if _native_layer_kind(item) == "layer_norm":
                        continue
                if item.value in self.unsafe_values:
                    continue
                region = build_native_autograd_region(
                    item,
                    self.gm,
                    registry=self.registry,
                    cache_dir=self.cache_dir,
                    block_size=self.block_size,
                    kernel_policy=self.kernel_policy,
                )
                self.native_autograd_indices[id(item)] = len(
                    self.native_autograd_regions
                )
                self.native_autograd_regions.append(region)
        covered_ids = set(self.native_autograd_indices) | {
            id(item) for item in self.schedule.items if isinstance(item, JointRegionNode)
        }
        eager_autograd: list[str] = []
        deepopt_autograd = tuple(
            f"{item.value.name}: {item.op}"
            for item in self.schedule.items
            if isinstance(item, ExternKernelNode)
            and item.autograd_owner == "deepopt"
            and item.value is not None
        )
        legacy_backward = []
        for item in self.schedule.items:
            if (
                not isinstance(item, IRNode)
                or isinstance(item, (InputNode, GetAttrNode, OutputNode))
                or id(item) in covered_ids
            ):
                continue
            extension = self.registry.for_node(item)
            if extension is not None and extension.training_codegen is not None:
                if any(output.meta is not None and output.meta.requires_grad for output in item.outputs()):
                    legacy_backward.append(f"{item.outputs()[0].name}: {getattr(item, 'op', type(item).__name__)}")
                continue
            outputs = item.outputs()
            if isinstance(item, ExternKernelNode) and item.autograd_owner == "deepopt":
                continue
            differentiable = tuple(output for output in outputs if output.meta is None or output.meta.requires_grad)
            if not differentiable:
                continue
            if any(
                value.meta is not None and value.meta.requires_grad
                for value in item.inputs()
            ):
                eager_autograd.extend(
                    f"{output.name}: {getattr(item, 'op', type(item).__name__)}" for output in differentiable
                )
        eager_autograd_ops = tuple(eager_autograd)
        if self.require_full_aot and (eager_autograd_ops or deepopt_autograd or legacy_backward):
            raise NotImplementedError(
                "Native AutoDiff coverage incomplete; operators retain other autograd backward owners: "
                + ", ".join((*eager_autograd_ops, *deepopt_autograd, *legacy_backward))
            )
        from ..scheduler import scheduled_inputs, scheduled_outputs, scheduled_nodes
        from deepopt._bf16_policy import resolve_bf16_policy
        lifetimes = {row["name"]: row for row in self.runtime_memory_manager.lifetime_plan()}
        pointwise_indices = {i for i, item in enumerate(self.schedule.items)
                             if isinstance(item, (SinglePointwiseOp, FusionGroup))}
        differentiable_values = any(value.meta is None or value.meta.requires_grad
                                    for item in self.schedule.items
                                    for value in (*scheduled_inputs(item), *scheduled_outputs(item)))
        reusable = frozenset(row["name"] for row in lifetimes.values()
            if not self.enable_autodiff and not differentiable_values and row["birth"] in pointwise_indices
            and row["owner"] == "invocation" and not row["escaped"]
            and set(row["consumers"]).issubset(pointwise_indices))
        if self.packed_offload_plan is not None:
            # The existing arena is deliberately single-stream. DMA storage is
            # owned by the separate transport leases, never borrowed by it.
            reusable = frozenset()
        artifact_index = 0
        reuse_lines = list(self.lines)
        observed_lines = list(self.lines)

        for schedule_index, item in enumerate(self.schedule.items):
            line_start = len(self.lines)
            observed_lines.append("    if getattr(_physical_observer, 'begin', None) is not None:")
            observed_lines.append(f"        _physical_observer.begin({schedule_index})")
            self.offload_before(schedule_index, item)
            if isinstance(item, InputNode):
                self.codegen_inputs(item)
            elif isinstance(item, GetAttrNode):
                self.codegen_get_attr(item)
            elif isinstance(item, (SinglePointwiseOp, FusionGroup)):
                self.codegen_pointwise(artifact_index)
                artifact_index += 1
            elif isinstance(item, PointwiseNode):
                self.codegen_native_node(item)
            elif isinstance(item, ExternKernelNode):
                self.codegen_extern(item)
            elif isinstance(item, OutputNode):
                self.codegen_output(item)
            elif isinstance(item, IRNode):
                consumer = self.offload_consumers.get(schedule_index)
                self.codegen_native_node(item, panel_consumer=consumer if getattr(consumer, "panel", None) is not None else None)
            else:
                raise TypeError(f"Unsupported schedule item: {type(item).__name__}")
            # Validate before any consumer can conceal a wrong intermediate
            # dtype by casting it. The policy is captured on IR, never ambient.
            produced = scheduled_outputs(item) if not isinstance(item, (InputNode, GetAttrNode, OutputNode)) else ()
            dtype_guards = self.intermediate_dtype_guards(produced)
            strict_storage = any(resolve_bf16_policy(node.bf16_policy).storage_enabled
                                 for node in scheduled_nodes(item))
            if strict_storage:
                self.lines.extend(dtype_guards)
            self.offload_publish(schedule_index)
            emitted = list(self.lines[line_start:])
            if isinstance(item, (SinglePointwiseOp, FusionGroup)) and not self.enable_autodiff:
                output = item.output_value
                runtime_name = self.lookup(output)
                emitted = [line.replace(f"{runtime_name} = torch.empty(",
                    f"{runtime_name} = _reuse_arena.acquire({output.name!r}, ") for line in emitted]
                emitted.append(f"    _reuse_arena.complete({output.name!r}, {runtime_name})")
            reuse_lines.extend(emitted)
            for value in scheduled_inputs(item):
                if value.name in reusable and self.runtime_memory_manager.remaining_users.get(value) == 1:
                    reuse_lines.append(f"    _reuse_arena.retire({value.name!r}, {self.lookup(value)})")
            observed_lines.extend(self.lines[line_start:])
            if not strict_storage:
                observed_lines.extend(dtype_guards)
            if isinstance(item, OutputNode):
                # Observe outputs before returning; preserve the original ABI.
                from ..scheduler import scheduled_inputs
                values = [value for value in scheduled_inputs(item)
                          if value not in self.runtime_memory_manager.runtime_exports]
                expression = "{" + ", ".join(f"{value.name!r}: {self.lookup(value)}" for value in values) + "}"
                observed_lines.insert(len(observed_lines) - 1,
                                      f"    _physical_observer.boundary({schedule_index}, {expression})")
            else:
                from ..scheduler import scheduled_inputs, scheduled_outputs
                values = list(dict.fromkeys((*scheduled_inputs(item), *scheduled_outputs(item))))
                if (isinstance(item, InputNode)
                        or getattr(self.offload_consumers.get(schedule_index), "panel", None) is not None):
                    # Selected saved arguments have no restored local binding
                    # at entry. Observe them at their numerical consumer; a
                    # diagnostic must never restore early merely to inspect.
                    values = [value for value in values if value not in self.offload_backward_inputs]
                expression = "{" + ", ".join(f"{value.name!r}: {self.lookup(value)}" for value in values) + "}"
                observed_lines.append(f"    _physical_observer.boundary({schedule_index}, {expression})")
            line_start = len(self.lines)
            if not isinstance(item, OutputNode):
                self.offload_after(schedule_index, item)
                self.runtime_memory_manager.emit_del(
                    item, self.lines, self.runtime_names,
                    skip_values=self.offload_backward_inputs,
                )
            observed_lines.extend(self.lines[line_start:])
            reuse_lines.extend(self.lines[line_start:])

        offload_regions = [region for region in self.joint_autograd_regions if region.offload_runtime is not None]
        def scope_source(source):
            if not offload_regions:
                return source
            lines = source.splitlines()
            return (lines[0] + "\n    with _packed_forward_scope():\n" +
                    "\n".join("    " + line for line in lines[1:]) + "\n")

        source = scope_source("\n".join(self.lines) + "\n")
        namespace = {
            "torch": torch,
            "_kernels": [artifact.kernel for artifact in self.artifacts],
            "_artifacts": self.artifacts,
            "_launch_kernel": launch_pointwise,
            "_autograd_regions": self.autograd_regions,
            "_native_autograd_regions": self.native_autograd_regions,
            "_native_ops": self.native_ops,
            "_consts": self.consts,
            "_gm": self.gm,
            "_fetch_attr": fetch_attr,
            "_intermediate_dtype_error": intermediate_dtype_error,
        }
        if offload_regions:
            from functools import partial
            from ..runtime.manager import compiled_forward_scope

            namespace["_packed_forward_scope"] = partial(compiled_forward_scope,
                tuple(region.offload_plan for region in offload_regions), offload_regions[0].offload_runtime)
        exec(compile(source, "<compiled_backend_wrapper>", "exec"), namespace)
        observed_lines[0] = ("def observed(args, *, _physical_observer):" if self.consume_inputs else
                             "def observed(*args, _physical_observer):")
        if self.packed_offload_plan is not None:
            observed_lines[0] = ("def observed(args, *, _physical_observer, _offload):" if self.consume_inputs else
                                 "def observed(*args, _physical_observer, _offload):")
        observed_source = (observed_lines[0] + "\n    _physical_error = None\n    try:\n" +
                           "\n".join("    " + line for line in "\n".join(observed_lines[1:]).splitlines()) +
                           "\n    except BaseException as error:\n        _physical_error = error\n        raise\n" +
                           "    finally:\n        _physical_observer.finish(_physical_error)\n")
        observed_source = scope_source(observed_source)
        exec(compile(observed_source, "<observed_backend_wrapper>", "exec"), namespace)
        reuse_lines[0] = ("def reused(args, *, _reuse_arena):" if self.consume_inputs else
                          "def reused(*args, _reuse_arena):")
        if self.packed_offload_plan is not None:
            reuse_lines[0] = ("def reused(args, *, _reuse_arena, _offload):" if self.consume_inputs else
                              "def reused(*args, _reuse_arena, _offload):")
        reuse_source = (reuse_lines[0] + "\n    try:\n" +
                        "\n".join("    " + line for line in "\n".join(reuse_lines[1:]).splitlines()) +
                        "\n    finally:\n        _reuse_arena.finish()\n")
        reuse_source = scope_source(reuse_source)
        exec(compile(reuse_source, "<reuse_backend_wrapper>", "exec"), namespace)
        artifact = WrapperArtifact(
            fn=namespace["compiled"], observed_fn=namespace["observed"],
            reuse_fn=namespace["reused"] if not self.enable_autodiff and reusable else None,
            reusable_values=tuple(sorted(reusable)),
            source=source,
            kernels=[artifact.kernel for artifact in self.artifacts],
            triton_artifacts=self.triton_artifacts,
            tilelang_artifacts=self.tilelang_artifacts,
            custom_kernels=self.custom_kernels,
            autograd_regions=self.autograd_regions,
            native_autograd_regions=self.native_autograd_regions,
            eager_autograd_ops=eager_autograd_ops,
            deepopt_autograd_ops=deepopt_autograd,
            deepopt_programs=self.deepopt_programs,
            kernel_routes=tuple(kernel_routes),
            joint_autograd_regions=self.joint_autograd_regions,
            legacy_backward_operations=tuple(legacy_backward),
            final_schedule=self.schedule.evidence(),
            logical_lifetimes=self.runtime_memory_manager.lifetime_plan(),
            scratch_lifetimes=self.runtime_memory_manager.scratch_plan(),
            boundary_metadata=tuple({"index": index, "phase": self.diagnostic_phase,
                "nodes": tuple({"operation": getattr(node, "op", type(node).__name__),
                    "module": getattr(node, "module_path", None),
                    "source_node": getattr(node, "source_node", None),
                    "inputs": tuple(value.name for value in node.inputs()),
                    "outputs": tuple(value.name for value in node.outputs()),
                    "representations": tuple(getattr(value.meta, "representation", None) for value in node.outputs())}
                    for node in scheduled_nodes(item))}
                for index, item in enumerate(self.schedule.items)),
        )
        for region in self.joint_autograd_regions:
            for child in (region.forward_artifact, region.backward_artifact):
                artifact.triton_artifacts.extend(child.triton_artifacts)
                artifact.tilelang_artifacts.extend(child.tilelang_artifacts)
                artifact.custom_kernels.extend(child.custom_kernels)
                artifact.kernel_routes += child.kernel_routes
                artifact.scratch_lifetimes += child.scratch_lifetimes
        artifact.fn._deepopt_scratch_lifetimes = artifact.scratch_lifetimes
        return artifact
