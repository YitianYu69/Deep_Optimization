from __future__ import annotations

from dataclasses import dataclass, fields, replace
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.fx as fx
from torch import is_grad_enabled

from .codegen.wrapper import WrapperCodegen
from .effects import analyze_fx_effects, compile_with_inductor
from .lowering import GraphLowering
from .registry import BackendRegistry
from .scheduler import FusionGroup, Scheduler, SinglePointwiseOp, external_values
from .reporting import CompilationReport, GraphReport, SAVED_STATE_SCHEMA_VERSION
from .ir import find_tensor_values, model_state_origin


@dataclass(frozen=True)
class BackendConfig:
    """Static, first-order graph compiler options; independent of FP8 policy."""

    block_size: int = 1024
    max_pointwise_fusion_length: int = 128
    cache_dir: str | Path | None = None
    debug: bool = False
    enable_aot_autodiff: bool = True
    use_native_autodiff: bool = True
    require_full_aot: bool = False
    collect_saved_state_metadata: bool = False
    mutation_backend: Literal["native", "inductor", "eager", "error"] = "native"
    capture_deepopt: bool = True
    fuse_deepopt_activations: bool = False
    fusion_phases: tuple[str, ...] | str = ("forward", "backward")
    share_compatible_packs: bool | None = None
    fuse_packed_multiply: bool | None = None
    qualify_native_group_fusion: bool = False
    fuse_norm_cat_body: bool = False
    donate_fresh_backward_add: bool = False
    native_conv_backward_order: Literal["source", "dw_first"] = "source"
    cuda_conv_dx_geometry: bool = False
    native_conv_workspace_reuse: Literal["off", "tiled_dx", "whole_batch"] = "off"
    native_conv_dx_batch_size: int = 1
    fuse_norm_activation_residuals: bool = False
    norm_activation_fusion_stage: Literal["off", "forward", "forward_backward"] = "off"
    fuse_swiglu_residuals: bool = False
    swiglu_fusion_stage: Literal["off", "activation_pack", "forward", "forward_backward"] = "off"
    bf16_gemm_backend: str = "native"
    bf16_decode: str = "materialized"
    bf16_tile: int = 64
    bf16_num_stages: int = 3
    bf16_backward_order: str = "dx_first"
    kernel_backend: str = "triton"
    kernel_policy: Any = None
    bf16_policy: Any = None
    activation_memory: Any = None
    # Static admission of pointwise partitions. "advisory" records predictions
    # without changing execution; "enforce" returns losing partitions to their
    # eager callable. Predictions are estimates, never measurements.
    cost_model: str = "advisory"
    # Explicit streaming bandwidth in bytes per second. Omitted means "read the
    # declared device properties"; an unavailable answer leaves regions
    # unestimated instead of guessing.
    cost_model_bandwidth_bytes_per_s: float | None = None
    cost_model_admit_margin: float = 0.10
    packed_activation_offload: Any = None
    packed_offload_selection_mode: Literal["peak", "all_eligible", "regional_order"] = "peak"
    packed_offload_peak_profile: Any = None

    def __post_init__(self) -> None:
        if self.packed_offload_selection_mode not in ("peak", "all_eligible", "regional_order"):
            raise ValueError("packed_offload_selection_mode must be peak, all_eligible or regional_order")
        from .model_offload_plan import resolve_peak_profile

        object.__setattr__(self, "packed_offload_peak_profile", resolve_peak_profile(self.packed_offload_peak_profile))
        from deepopt.kernel_policy import resolve_kernel_policy
        from deepopt._bf16_policy import resolve_bf16_policy
        from deepopt._activation_memory import (activation_memory_compiler_options,
                                               resolve_activation_memory, validate_activation_memory_execution)
        from deepopt._packed_offload import validate_packed_offload_execution

        object.__setattr__(self, "packed_activation_offload", validate_packed_offload_execution(
            self.packed_activation_offload, compiler_options={
                "enable_aot_autodiff": self.enable_aot_autodiff, "use_native_autodiff": self.use_native_autodiff}))
        if (self.packed_activation_offload.enabled and not self.packed_activation_offload.physical_host_budget
                and (self.packed_offload_selection_mode != "peak" or self.packed_offload_peak_profile is not None)):
            raise ValueError("Whole-model selection and peak profiles require packed offload descriptor version 2")

        memory_options = {key: getattr(self, key) for key in ("fuse_packed_multiply", "share_compatible_packs")
                          if getattr(self, key) is not None}
        object.__setattr__(self, "_activation_memory_explicit_options", tuple(memory_options.items()))
        memory = resolve_activation_memory(self.activation_memory, compiler_options=memory_options)
        memory_options = activation_memory_compiler_options(memory, memory_options)
        object.__setattr__(self, "activation_memory", memory)
        for key in ("fuse_packed_multiply", "share_compatible_packs"):
            object.__setattr__(self, key, memory_options.get(key, False))

        if self.bf16_policy is not None:
            object.__setattr__(self, "bf16_policy", resolve_bf16_policy(self.bf16_policy))

        if self.bf16_gemm_backend not in ("native", "generated"):
            raise ValueError("bf16_gemm_backend must be native or generated")
        if isinstance(self.fusion_phases, str):
            aliases = {"both": ("forward", "backward"), "none": (),
                       "forward": ("forward",), "backward": ("backward",)}
            if self.fusion_phases not in aliases:
                raise ValueError("fusion_phases must be forward, backward, both, or none")
            object.__setattr__(self, "fusion_phases", aliases[self.fusion_phases])
        if (not isinstance(self.fusion_phases, (tuple, list))
                or any(phase not in ("forward", "backward") for phase in self.fusion_phases)
                or len(set(self.fusion_phases)) != len(self.fusion_phases)):
            raise ValueError("fusion_phases must contain unique forward/backward phases")
        object.__setattr__(self, "fusion_phases", tuple(self.fusion_phases))
        resolve_activation_memory(memory, compiler_options={
            "fusion_phases": self.fusion_phases, "enable_aot_autodiff": self.enable_aot_autodiff,
            "use_native_autodiff": self.use_native_autodiff})
        if self.bf16_decode not in ("materialized", "tiled", "panel"):
            raise ValueError("bf16_decode must be materialized, tiled, or panel")
        if self.bf16_decode in ("tiled", "panel") and self.bf16_gemm_backend != "generated":
            raise ValueError(f"bf16_decode={self.bf16_decode} requires bf16_gemm_backend=generated")
        if self.bf16_backward_order not in ("dx_first", "dw_first"):
            raise ValueError("bf16_backward_order must be dx_first or dw_first")
        if type(self.native_conv_backward_order) is not str or self.native_conv_backward_order not in ("source", "dw_first"):
            raise ValueError("native_conv_backward_order must be source or dw_first")
        if (type(self.native_conv_workspace_reuse) is not str
                or self.native_conv_workspace_reuse not in ("off", "tiled_dx", "whole_batch")):
            raise ValueError("native_conv_workspace_reuse must be off, tiled_dx, or whole_batch")
        if type(self.native_conv_dx_batch_size) is not int or self.native_conv_dx_batch_size < 1:
            raise ValueError("native_conv_dx_batch_size must be a positive integer")
        for flag, stage, choices in (
                ("fuse_norm_activation_residuals", "norm_activation_fusion_stage", ("off", "forward", "forward_backward")),
                ("fuse_swiglu_residuals", "swiglu_fusion_stage", ("off", "activation_pack", "forward", "forward_backward"))):
            value = getattr(self, stage)
            if type(getattr(self, flag)) is not bool or type(value) is not str or value not in choices:
                raise ValueError(f"Invalid typed native training fusion controls: {flag}/{stage}")
            if getattr(self, flag) != (value != "off"):
                raise ValueError(f"{flag} must match the enabled/off value of {stage}")
            if value != "off" and not (self.enable_aot_autodiff and self.use_native_autodiff):
                raise ValueError("Training residual fusion requires DeepOpt native AutoDiff")
            required = {"forward", "backward"} if value == "forward_backward" else {"forward"}
            if value != "off" and not required.issubset(self.fusion_phases):
                raise ValueError(f"{stage} conflicts with the selected global fusion_phases")
        if type(self.bf16_tile) is not int or self.bf16_tile not in (32, 64, 128):
            raise ValueError("bf16_tile must be 32, 64, or 128")
        if type(self.bf16_num_stages) is not int or self.bf16_num_stages not in (1, 2, 3, 4, 5):
            raise ValueError("bf16_num_stages must be between 1 and 5")
        if self.kernel_backend not in {"triton", "tilelang", "mixed"}:
            raise ValueError("kernel_backend must be triton, tilelang, or mixed")
        policy = resolve_kernel_policy(self.kernel_policy if self.kernel_policy is not None else self.kernel_backend)
        if policy.backend != self.kernel_backend:
            raise ValueError("kernel_backend conflicts with the supplied kernel_policy")
        validate_activation_memory_execution(memory, kernel_policy=policy)
        if self.bf16_decode == "panel" and policy.route("gemm", "linear.dw") != "triton":
            raise ValueError("bf16_decode=panel requires a Triton linear.dw GEMM route")
        if (
            type(self.block_size) is not int
            or self.block_size < 1
            or self.block_size & (self.block_size - 1)
        ):
            raise ValueError("block_size must be a positive power of two")
        if (
            type(self.max_pointwise_fusion_length) is not int
            or self.max_pointwise_fusion_length < 1
        ):
            raise ValueError("max_pointwise_fusion_length must be at least one")
        if self.mutation_backend not in {"native", "inductor", "eager", "error"}:
            raise ValueError(
                "mutation_backend must be 'native', 'inductor', 'eager', or 'error'"
            )
        if self.cost_model not in {"off", "advisory", "enforce"}:
            raise ValueError("cost_model must be off, advisory, or enforce")
        if self.cost_model_bandwidth_bytes_per_s is not None and (
            type(self.cost_model_bandwidth_bytes_per_s) not in (int, float)
            or not self.cost_model_bandwidth_bytes_per_s > 0
        ):
            raise ValueError("cost_model_bandwidth_bytes_per_s must be a positive number or None")
        if type(self.cost_model_admit_margin) not in (int, float) or not (
            0.0 <= self.cost_model_admit_margin < 1.0
        ):
            raise ValueError("cost_model_admit_margin must be in [0, 1)")
        for name in (
            "debug",
            "enable_aot_autodiff",
            "use_native_autodiff",
            "require_full_aot",
            "collect_saved_state_metadata",
            "capture_deepopt",
            "fuse_deepopt_activations",
            "share_compatible_packs",
            "fuse_packed_multiply",
            "qualify_native_group_fusion",
            "fuse_norm_cat_body",
            "donate_fresh_backward_add",
            "cuda_conv_dx_geometry",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.require_full_aot and not self.enable_aot_autodiff:
            raise ValueError("require_full_aot requires enable_aot_autodiff")
        if self.require_full_aot and not self.use_native_autodiff:
            raise ValueError("require_full_aot requires use_native_autodiff; legacy training programs do not qualify")
        if self.require_full_aot and self.cost_model == "enforce":
            # Declining a differentiable partition returns it to eager autograd,
            # which is exactly what strict ownership forbids.
            raise ValueError(
                "require_full_aot cannot use cost_model='enforce'; "
                "use 'advisory' to record predictions without changing ownership"
            )
        if (self.fuse_norm_cat_body or self.donate_fresh_backward_add
                or self.native_conv_backward_order != "source" or self.cuda_conv_dx_geometry
                or self.native_conv_workspace_reuse != "off") and not (
            self.enable_aot_autodiff and self.use_native_autodiff
        ):
            raise ValueError("Native memory optimizations require DeepOpt native AutoDiff")
        if self.cuda_conv_dx_geometry and self.donate_fresh_backward_add:
            raise ValueError("CUDA convolution geometry and backward-add donation require a separately qualified ownership composition")
        if self.native_conv_workspace_reuse != "off" and (self.donate_fresh_backward_add
                or self.native_conv_backward_order != "source" or self.cuda_conv_dx_geometry):
            raise ValueError("Private convolution workspace requires separately qualified composition with order, geometry, or donation controls")
        if self.cache_dir is not None and (
            not isinstance(self.cache_dir, (str, Path))
            or not str(self.cache_dir).strip()
        ):
            raise ValueError("cache_dir must be a nonempty path or None")

    @classmethod
    def from_options(cls, options: Mapping | None = None, *, mode: str | None = None):
        if mode not in (None, "default"):
            raise ValueError(
                "The deepopt compiler supports mode='default'; use explicit backend options"
            )
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("DeepOpt compiler options must be a mapping")
        raw = dict(options or {})
        unknown = set(raw) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown DeepOpt compiler options: {sorted(unknown)}")
        return cls(**raw)

    @property
    def reduction_scratch_bytes(self):
        return self.activation_memory.reduction_scratch_bytes


def compile_fx(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any],
    *,
    registry: BackendRegistry | None = None,
    config: BackendConfig | None = None,
    report: CompilationReport | None = None,
):
    """Compile the supported FX subset and return a callable.

    Unsupported operators stay on their original PyTorch callable path.
    Supported semantic regions use this project's ``AOTAutoDiff`` before kernel
    scheduling. Its derived backward IR and explicit saved-state exports run
    through one thin autograd bridge per region. BF16/FP8, compression, geometry,
    attention and spatial VJPs lower to shared numerical kernels or native
    library calls; they never delegate derivation to ``torch.autograd.grad``.
    Strict ownership rejects legacy custom backward/eager escapes. An explicit
    ``use_native_autodiff=False`` migration control retains the former programs.

    This compiler supports one device per graph, first-order gradients,
    and the existing static-shape pointwise subset. It does not manage
    distributed process groups or communication.
    """
    config = config or BackendConfig()
    from deepopt.kernel_policy import resolve_kernel_policy

    kernel_policy = resolve_kernel_policy(
        config.kernel_policy if config.kernel_policy is not None else config.kernel_backend,
    )
    from deepopt._activation_memory import (current_activation_memory, invocation_activation_memory, resolve_activation_memory,
                                           validate_activation_memory_execution)
    from torch.utils._pytree import tree_leaves

    invocation_devices = {value.device.type for value in tree_leaves(example_inputs)
                          if isinstance(value, torch.Tensor)} or {"cpu"}
    for device in invocation_devices:
        from deepopt._packed_offload import validate_packed_offload_execution

        validate_packed_offload_execution(config.packed_activation_offload, device_type=device)
        memory = invocation_activation_memory(config.activation_memory, device)
        resolve_activation_memory(memory, compiler_options={**dict(config._activation_memory_explicit_options),
            "fusion_phases": config.fusion_phases, "enable_aot_autodiff": config.enable_aot_autodiff,
            "use_native_autodiff": config.use_native_autodiff})
        validate_activation_memory_execution(memory, kernel_policy=kernel_policy)
    effective_options = {
        item.name: (
            kernel_policy.as_dict() if item.name == "kernel_policy" else
            config.bf16_policy.as_dict() if item.name == "bf16_policy" and config.bf16_policy is not None else
            config.activation_memory.as_dict() if item.name == "activation_memory" else
            config.packed_activation_offload.as_dict() if item.name == "packed_activation_offload" else
            config.packed_offload_peak_profile.as_dict() if item.name == "packed_offload_peak_profile"
                and config.packed_offload_peak_profile is not None else
            str(config.cache_dir) if item.name == "cache_dir" and config.cache_dir is not None else
            getattr(config, item.name)
        )
        for item in fields(config)
    }
    started = perf_counter()
    compile_registry = (registry if registry is not None else BackendRegistry()).copy()

    if config.enable_aot_autodiff and config.use_native_autodiff:
        from .nonstrict_padding import normalize_nonstrict_zero_replication_padding

        gm = normalize_nonstrict_zero_replication_padding(gm)
    gm.graph.lint()
    from .effects import snapshot_stochastic_depth

    effective_options["stochastic_depth_snapshots"] = snapshot_stochastic_depth(gm)
    effects = analyze_fx_effects(gm)
    from dataclasses import asdict
    effective_options["write_rewrites"] = tuple(asdict(rewrite) for rewrite in effects.write_rewrites)
    analyzed = perf_counter()
    if effects.needs_functionalization:
        if config.debug:
            print(
                f"Mutation nodes {effects.mutation_nodes!r}; "
                f"using {config.mutation_backend} backend"
            )
        if config.mutation_backend == "error":
            raise NotImplementedError(
                f"FX graph writes through aliases at {effects.mutation_nodes!r}"
            )
        if config.mutation_backend == "eager":
            if config.require_full_aot:
                raise ValueError(
                    "require_full_aot cannot delegate a mutated graph to eager execution"
                )
            if report is not None:
                report.graphs.append(
                    GraphReport(
                        len(list(gm.graph.nodes)),
                        reason="explicit eager mutation policy",
                        effective_compiler_options=effective_options,
                    )
                )
            return gm.forward
        if config.mutation_backend == "inductor":
            from .operations import is_deepopt_target

            if any(
                is_deepopt_target(
                    gm.get_submodule(node.target)
                    if node.op == "call_module"
                    else node.target
                )
                for node in gm.graph.nodes
                if node.op.startswith("call_")
            ):
                raise ValueError(
                    "Prepared DeepOpt operations require mutation_backend='native' or 'eager'"
                )
            if config.require_full_aot:
                raise ValueError(
                    "require_full_aot cannot delegate a graph to another compiler"
                )
            if report is not None:
                report.graphs.append(
                    GraphReport(
                        len(list(gm.graph.nodes)),
                        reason="explicit Inductor mutation policy",
                        effective_compiler_options=effective_options,
                    )
                )
            return compile_with_inductor(gm, tuple(example_inputs))

    lowering = GraphLowering(
        gm,
        example_inputs,
        registry=compile_registry,
        functionalize_nodes=frozenset(effects.functionalize_nodes),
        write_rewrites=effects.write_rewrites,
        native_autodiff=config.enable_aot_autodiff and config.use_native_autodiff,
    )
    ir_graph = lowering.run()
    from deepopt._bf16_policy import current_bf16_policy

    devices = {str(value.meta.device).split(":")[0] for node in ir_graph.nodes
               for value in (*node.inputs(), *node.outputs()) if value.meta is not None and value.meta.device is not None}
    bf16_policies = {device: current_bf16_policy(device) for device in devices}
    effective_options["bf16_policies"] = {device: policy.as_dict() for device, policy in bf16_policies.items()}
    from deepopt.opt._recipe import bf16_recipe

    effective_options["bf16_recipes"] = {device: bf16_recipe(policy) for device, policy in bf16_policies.items()}
    effective_options["activation_memory_plans"] = {
        device: (config.activation_memory if not config.activation_memory.is_default else
                 current_activation_memory(device)).as_dict() for device in devices}
    for node in ir_graph.nodes:
        device = next((str(value.meta.device).split(":")[0] for value in (*node.inputs(), *node.outputs())
                       if value.meta is not None and value.meta.device is not None), "cpu")
        node.bf16_policy = current_bf16_policy(device)
        node.activation_memory = (config.activation_memory if not config.activation_memory.is_default else
                                  current_activation_memory(device))
        plan = getattr(node, "plan", None) or getattr(getattr(node, "module", None), "plan", None)
        if plan is not None and node.inputs():
            from types import SimpleNamespace
            from deepopt.opt.precision.bf16 import invocation_bf16_policy

            metadata = node.inputs()[0].meta
            node.bf16_policy = invocation_bf16_policy(plan, SimpleNamespace(
                dtype=metadata.dtype, device=torch.device(metadata.device)))
    lowered_at = perf_counter()
    unsafe_values = frozenset(
        value
        for node in gm.graph.nodes
        if node.name in effects.unsafe_nodes and node in lowering.value_map
        for value in find_tensor_values(lowering.value_map[node])
    )
    from .native_operations import materialize_cached_bf16_casts

    cached_cast_aliases = materialize_cached_bf16_casts(ir_graph, example_inputs, gm, unsafe_values)
    effective_options["cached_autocast_parameter_aliases"] = cached_cast_aliases
    # Cast sharing can introduce nodes after policy annotation. They inherit
    # their source's immutable selection rather than lacking report identity.
    for node in ir_graph.nodes:
        if not hasattr(node, "activation_memory"):
            node.activation_memory = next((getattr(value.producer, "activation_memory")
                for value in node.inputs() if hasattr(value.producer, "activation_memory")), config.activation_memory)
    lowered_nodes = tuple(ir_graph.nodes)
    from .linear import _fusion_signature, fuse_activations, linear_module
    from .ir import PointwiseNode

    fusion_candidates = []
    # Unsupported prepared recipes retain their original executor. They still
    # need an explicit fusion decision rather than disappearing from the report.
    for fx_node in gm.graph.nodes:
        lowering_reason = lowering.fallback_reasons.get(fx_node.name)
        if lowering_reason is None:
            continue
        target = lowering.context.resolve_target(fx_node)
        module = linear_module(target)
        if module is None:
            continue
        inputs = list(find_tensor_values(lowering.context.map_arg(fx_node.args)))
        value = inputs[0] if inputs else None
        producer = getattr(value, "producer", None)
        discovered = isinstance(producer, PointwiseNode) and producer.op in ("gelu", "silu")
        source = producer.args[0] if discovered and producer.args else value
        output = lowering.value_map.get(fx_node)
        meta = getattr(output, "meta", None)
        fusion_candidates.append({
            "module_path": fx_node.target if fx_node.op == "call_module" else getattr(target, "path", None),
            "node_name": fx_node.name,
            "phase": "train" if module.training else "eval",
            "module_training": bool(module.training),
            "training": bool(meta is not None and meta.requires_grad and is_grad_enabled()),
            "requested": config.fuse_deepopt_activations,
            "effective_enabled": config.fuse_deepopt_activations,
            "discovered": discovered,
            "eligible": False if config.fuse_deepopt_activations else None,
            "selected": False,
            "status": "rejected" if config.fuse_deepopt_activations else "disabled",
            "reason": "unsupported_lowering" if config.fuse_deepopt_activations else "disabled",
            "reason_detail": lowering_reason,
            "activation": {
                "kind": producer.op,
                "approximate": producer.kwargs.get("approximate", "none"),
            } if discovered else None,
            "input_signature": _fusion_signature(source),
            "activation_output_signature": _fusion_signature(value),
            "selected_schedule": None,
            "observed_cuda_launches": None,
        })
    joint_programs = []
    if config.enable_aot_autodiff and config.use_native_autodiff and is_grad_enabled():
        from .autodiff.joint import partition

        ir_graph, joint_programs = partition(ir_graph, compile_registry, config, kernel_policy, unsafe_values)
        fusion_candidates.extend(candidate for program in joint_programs for candidate in program.fusion_candidates)
    fusion_patterns = fuse_activations(
        ir_graph, enabled=config.fuse_deepopt_activations, unsafe_values=unsafe_values,
        kernel_policy=kernel_policy, diagnostics=fusion_candidates,
    )
    from .native_linear import fuse_native

    fusion_patterns += fuse_native(
        ir_graph, enabled=config.fuse_deepopt_activations, unsafe_values=unsafe_values,
        config=config, kernel_policy=kernel_policy, diagnostics=fusion_candidates,
    )
    fusion_patterns += tuple(pattern for program in joint_programs for pattern in program.fusion_patterns)
    memory_only_regions = sum(candidate.get("selected", False) and candidate.get("kernel_fusion") is False
                              for candidate in fusion_candidates)
    fused_region_count = len(fusion_patterns) - memory_only_regions
    if fused_region_count < 0:
        raise ValueError("Optimization report has more memory-only selections than transformed regions")
    schedule = Scheduler(
        ir_graph,
        registry=compile_registry,
        max_pointwise_fusion_length=config.max_pointwise_fusion_length,
        unsafe_values=unsafe_values,
        kernel_policy=kernel_policy,
    ).run()
    from . import heuristic as cost_model

    bandwidth = (None if config.cost_model_bandwidth_bytes_per_s is None
                 else float(config.cost_model_bandwidth_bytes_per_s))
    if config.cost_model != "off":
        schedule, cost_estimates = cost_model.apply(
            schedule, mode=config.cost_model, bandwidth_bytes_per_s=bandwidth,
            admit_margin=config.cost_model_admit_margin,
        )
    else:
        cost_estimates = ()
    # cost_model stays a scalar above; its parameters are recorded separately so
    # the option round-trips as declared.
    effective_options["cost_model_parameters"] = {
        "bandwidth_bytes_per_s": bandwidth,
        "admit_margin": config.cost_model_admit_margin,
    }
    scheduled_at = perf_counter()
    if not config.enable_aot_autodiff and any(
        value.meta is not None and value.meta.requires_grad
        for item in schedule.items
        if isinstance(item, (FusionGroup, SinglePointwiseOp))
        for value in external_values(
            item.nodes if isinstance(item, FusionGroup) else [item.node]
        )
    ):
        raise ValueError(
            "Differentiable generated regions require enable_aot_autodiff=True"
        )
    artifact = WrapperCodegen(
        gm,
        schedule,
        registry=compile_registry,
        cache_dir=config.cache_dir,
        block_size=config.block_size,
        enable_autodiff=config.enable_aot_autodiff,
        require_full_aot=config.require_full_aot,
        unsafe_values=unsafe_values,
        kernel_policy=kernel_policy,
        bf16_options={name: getattr(config, name) for name in (
            "bf16_gemm_backend", "bf16_decode", "bf16_tile", "bf16_num_stages", "bf16_backward_order",
        )},
    ).run()
    generated_at = perf_counter()

    # Keep generated evidence inspectable without writing noisy files or
    # printing into a training/serving process by default.
    artifact.fn.backend_artifact = artifact
    from .linear import DeepOptLinearNode
    from .native_linear import NativeLinearNode
    from .native_stateful import NativeStatefulNode
    from .native_attention import AttentionNode
    from .native_operations import OperationNode

    lowered_deepopt = tuple(
        f"{node.outputs()[0].name}: {node.op}"
        for node in lowered_nodes if isinstance(node, (DeepOptLinearNode, NativeLinearNode, NativeStatefulNode, AttentionNode, OperationNode))
    )
    schedules_by_node = {
        program.node_name: {
            "forward": program.forward_steps,
            "backward": program.backward_steps if program.training else (),
            "saved_components": program.saved_components,
        }
        for program in artifact.deepopt_programs
    }
    for candidate in fusion_candidates:
        if candidate["selected"]:
            candidate["selected_schedule"] = schedules_by_node.get(candidate["node_name"])
    graph_report = GraphReport(
        operations=sum(
            isinstance(node, fx.Node) and node.op.startswith("call_")
            for node in gm.graph.nodes
        ),
        triton_regions=len(artifact.triton_artifacts),
        tilelang_regions=len(artifact.tilelang_artifacts),
        kernel_backend=kernel_policy.backend,
        kernel_policy=kernel_policy.as_dict(),
        kernel_routes=artifact.kernel_routes,
        final_schedule=artifact.final_schedule,
        logical_lifetimes=artifact.logical_lifetimes,
        effective_compiler_options=effective_options,
        precision_contracts=tuple({
            "operation": getattr(node, "op", type(node).__name__),
            "bf16_policy": node.bf16_policy.as_dict(),
            "activation_memory": node.activation_memory.as_dict(),
            "outputs": tuple({"name": value.name, "shape": value.meta.shape,
                "dtype": str(value.meta.dtype), "storage_dtype": str(value.meta.storage_dtype or value.meta.dtype),
                "provenance": value.provenance, "representation": value.meta.representation}
                for value in node.outputs() if value.meta is not None),
        } for node in lowered_nodes),
        fusion_candidates=tuple(fusion_candidates),
        pack_sharing_decisions=tuple(decision for region in artifact.joint_autograd_regions
                                    for decision in region.pack_sharing_decisions),
        pointwise_cost_estimates=tuple(cost_estimates),
        static_regions={
            # Predicted counts, valid in both advisory and enforce modes.
            "predicted_declined_pointwise_regions": sum(
                estimate["decision"] == "declined" for estimate in cost_estimates
            ),
            "predicted_admitted_pointwise_regions": sum(
                estimate["decision"] == "admitted" for estimate in cost_estimates
            ),
            "unestimated_pointwise_regions": sum(
                estimate["decision"] == "unestimated" for estimate in cost_estimates
            ),
            # Applied only when enforce actually returned the region to eager.
            "declined_pointwise_regions": sum(
                estimate.get("applied", False) for estimate in cost_estimates
            ),
            "standalone_pointwise_regions": sum(
                isinstance(item, SinglePointwiseOp) or
                isinstance(item, FusionGroup) and len(item.nodes) == 1
                for item in schedule.items
            ),
            "multi_operation_pointwise_regions": sum(
                isinstance(item, FusionGroup) and len(item.nodes) > 1
                for item in schedule.items
            ),
            "linear_launch_programs": len(artifact.deepopt_programs),
            "bf16_linear_programs": sum(getattr(p, "precision", None) == "bf16" for p in artifact.deepopt_programs),
            "generated_bf16_programs": sum(getattr(p, "generated_kernels", False) for p in artifact.deepopt_programs),
            "native_saved_state_programs": sum(getattr(p, "precision", None) == "native" for p in artifact.deepopt_programs),
            "fused_preparation_regions": fused_region_count,
            "memory_donation_regions": sum(candidate.get("selected", False)
                and candidate.get("transformation") == "buffer_donation" for candidate in fusion_candidates),
            "compiler_autograd_regions": len(artifact.autograd_regions) + len(artifact.native_autograd_regions) + len(artifact.joint_autograd_regions),
        },
        custom_kernels=len(artifact.custom_kernels),
        compiler_autograd_regions=len(artifact.autograd_regions)
        + len(artifact.native_autograd_regions) + len(artifact.joint_autograd_regions),
        native_autodiff_regions=len(artifact.joint_autograd_regions),
        native_autodiff_operations=sum(len(region.rules) for region in artifact.joint_autograd_regions),
        differentiation_rules=tuple(rule for region in artifact.joint_autograd_regions for rule in region.rules),
        legacy_backward_operations=artifact.legacy_backward_operations,
        native_autodiff_graphs=tuple({
            "saved_state_schema_version": (4 if region.offload_plan is not None and region.offload_plan.bundles
                                            else SAVED_STATE_SCHEMA_VERSION),
            **({"packed_activation_offload": {"transport_envelope_version": region.offload_plan.policy.version,
                                             "plan": region.offload_plan.as_dict(),
                                             **({"model_scope": region.model_offload_plan}
                                                if hasattr(region, "model_offload_plan") else {})}}
               if region.offload_plan is not None and (region.offload_plan.bundles
                   or hasattr(region, "model_offload_plan")) else {}),
            "runtime_saved_state": region.saved_state_metadata,
            "model_state_sources": tuple({"source": origin[0],
                "provenance": value.provenance, "shape": origin[1].shape,
                "dtype": str(origin[1].dtype), "device": str(origin[1].device)}
                for value in region.inputs if (origin := model_state_origin(value)) is not None),
            "bf16_policy": region.context.bf16_policy.as_dict(),
            "activation_memory": region.context.activation_memory.as_dict(),
            "differentiation_owner": "deepopt_native_autodiff",
            "forward_operations": tuple(getattr(node, "op", type(node).__name__) for node in region.graph.nodes),
            "backward_operations": tuple(getattr(node, "op", type(node).__name__) for node in region.trace.graph.nodes),
            "actual_execution_order": region.execution_order,
            "pack_sharing_decisions": region.pack_sharing_decisions,
            "saved_state": tuple({"name": value.name, "shape": value.meta.shape,
                "dtype": str(value.meta.dtype) if value.meta.dtype is not None else None,
                "device": str(value.meta.device) if value.meta.device is not None else None,
                "presence": getattr(value.meta, "presence", "required"),
                "representation": value.meta.representation,
                **({"stride": value.meta.stride} if region.offload_plan is not None and region.offload_plan.bundles else {})}
                for value in region.trace.saved_tensors),
            "saved_value_bindings": region.trace.saved_value_specs,
            "saved_activation_recipe": (region.context.saved_activation_policy.recipe
                if region.context.saved_activation_policy is not None else ()),
            "saved_activation_enabled": bool(region.context.saved_activation_policy and region.context.saved_activation_policy.enabled),
            "semantic_state_contracts": tuple({"operation": spec.op,
                "requested_gradients": tuple(getattr(spec, "needs", ())) or tuple(
                    metadata is not None and metadata.requires_grad for metadata in
                    getattr(spec, "metas", (getattr(spec, "input_meta", None),
                        getattr(spec, "weight_meta", None), getattr(spec, "bias_meta", None)))),
                "state_names": tuple(getattr(spec, "names", getattr(spec, "state_names", ()))),
            } for spec in region.specifications),
            "gradient_targets": tuple(region.inputs[int(target.key)].name for target in region.trace.grad_targets),
            "backward_orders": region.backward_orders,
            "final_forward_schedule": region.forward_artifact.final_schedule,
            "final_backward_schedule": region.backward_artifact.final_schedule,
            "forward_logical_lifetimes": region.forward_artifact.logical_lifetimes,
            "backward_logical_lifetimes": region.backward_artifact.logical_lifetimes,
            "forward_scratch_lifetimes": getattr(region.forward_artifact, "scratch_lifetimes", ()),
            "backward_scratch_lifetimes": getattr(region.backward_artifact, "scratch_lifetimes", ()),
        } for region in artifact.joint_autograd_regions),
        deepopt_operations=artifact.deepopt_autograd_ops + lowered_deepopt,
        lowered_deepopt_operations=lowered_deepopt,
        compiled_deepopt_backward_regions=sum(program.training for program in artifact.deepopt_programs)
        + sum(rule["operation"] in {"autodiff_sdpa", "autodiff_spatial"}
              for region in artifact.joint_autograd_regions for rule in region.rules),
        unsupported_deepopt_operations=tuple(
            f"{name}: {reason}" for name, reason in lowering.fallback_reasons.items()
        ),
        fused_deepopt_regions=fused_region_count,
        memory_optimization_regions=memory_only_regions,
        fusion_patterns=fusion_patterns,
        program_schedules=tuple({
            "module_path": getattr(program, "module_path", None),
            "node_name": getattr(program, "node_name", None),
            "module_training": getattr(program, "module_training", None),
            "training": program.training,
            "forward": program.forward_steps,
            "backward": program.backward_steps if program.training else (),
            "saved_components": program.saved_components,
            "activation": program.activation,
            "activation_saved_state": getattr(program, "activation_state", None),
            "precision": getattr(program, "precision", "fp8"),
            "implementation": getattr(program, "execution", "existing_fp8_kernels"),
            "decode": getattr(program, "decode", None),
            "scratch_requirements": getattr(program, "scratch_requirements", ()),
            "backward_order": getattr(program, "backward_order", None),
            "epilogue": getattr(program, "epilogue", None),
            "residual_epilogue": getattr(program, "residual", False),
            "kernel_routes": program.kernel_routes,
            "kernel_policy_fingerprint": program.kernel_policy_fingerprint,
            "profile_ranges": getattr(program, "profile_ranges", False),
        } for program in artifact.deepopt_programs),
        compile_time_ms={
            "registry_and_effects": (analyzed - started) * 1000,
            "lowering": (lowered_at - analyzed) * 1000,
            "scheduling": (scheduled_at - lowered_at) * 1000,
            "codegen": (generated_at - scheduled_at) * 1000,
        },
        eager_autograd_operations=artifact.eager_autograd_ops,
        native_operations=tuple(
            str(node.target) for node in gm.graph.nodes if node.op.startswith("call_")
        ),
        reason="static contiguous dense pointwise fusion; other operations retain their declared executor",
    )
    artifact.fn.compilation_report = graph_report
    if report is not None:
        report.graphs.append(graph_report)
    if config.debug:
        ir_graph.dump()
        schedule.dump()
        print("\n========== GENERATED WRAPPER ==========")
        print(artifact.source)
        print("========================================\n")
    from .operations import module_contracts, module_contracts_changed, numerical_scope

    scope = deepcopy(numerical_scope())
    contracts = module_contracts(gm)
    raw = artifact.fn

    def cast_aliases_changed():
        for paths in cached_cast_aliases:
            values = []
            for path in paths:
                value = gm
                for component in path.split("."):
                    value = getattr(value, component)
                values.append(value)
            if any(value is not values[0] for value in values[1:]):
                return True
        return False

    def guarded(*args):
        # An opaque call reads the live policy. If its output precision may
        # have changed, downstream kernels cannot reuse old static metadata.
        # Select the original FX operations before executing any compiled work.
        reason = (
            "numerical scope changed"
            if numerical_scope() != scope
            else "native autocast parameter alias changed"
            if cast_aliases_changed()
            else "module configuration changed"
            if module_contracts_changed(contracts)
            else None
        )
        if reason is not None:
            if config.require_full_aot:
                raise ValueError(
                    f"{reason} under require_full_aot; recompile the graph"
                )
            if report is not None:
                report.fallback(reason)
            return gm.forward(*args)
        return raw(*args)

    guarded.backend_artifact = artifact
    guarded.compilation_report = graph_report
    return guarded


def do_backend(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any],
    *,
    registry: BackendRegistry | None = None,
    config: BackendConfig | None = None,
    options: Mapping | None = None,
    mode: str | None = None,
    report: CompilationReport | None = None,
):
    """Alias suitable as a ``torch.compile(backend=...)`` entry point."""
    selected = BackendConfig.from_options(options, mode=mode)
    if config is not None and options:
        changes = dict(options)
        explicit = dict(config._activation_memory_explicit_options)
        for name in ("fuse_packed_multiply", "share_compatible_packs"):
            if name not in changes and name not in explicit:
                # Reapply the selected recipe without turning a resolved
                # default into a new explicit legacy-option contradiction.
                changes[name] = None
        selected = replace(config, **changes)
    elif config is not None:
        selected = config
    return compile_fx(
        gm, example_inputs, registry=registry, config=selected, report=report
    )


def make_backend(config: BackendConfig, report: CompilationReport):
    """Bind a report to one model without process-global diagnostic state."""

    def backend(gm, example_inputs, **kwargs):
        return do_backend(gm, example_inputs, config=config, report=report, **kwargs)

    return backend
