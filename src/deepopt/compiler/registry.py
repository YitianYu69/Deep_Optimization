"""Extension points for custom IR operators.

An extension describes the complete path for one operator: matching an FX node,
lowering it to a user supplied IR node, rendering its per-element expression in
Triton, and deciding whether it can participate in fusion.

Extensions are trusted Python code. In particular, ``triton_codegen`` returns
source text that is compiled as part of a generated Triton module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from torch.fx import GraphModule, Node

from .ir import IRNode


FxMatcher = Callable[[Node, GraphModule], bool]
Lowering = Callable[[Any, Node], IRNode]
TritonCodegen = Callable[[IRNode, Sequence[str], Mapping[str, str]], str]
FusionPredicate = Callable[[IRNode, IRNode], bool]


@dataclass(frozen=True)
class KernelCodegenContext:
    """Compile-time options supplied to a custom standalone kernel compiler."""

    backend: str = "triton"
    block_size: int = 1024
    cache_dir: str | None = None
    kernel_policy: Any = None
    bf16_gemm_backend: str = "native"
    bf16_decode: str = "materialized"
    bf16_tile: int = 64
    bf16_num_stages: int = 3
    bf16_backward_order: str = "dx_first"
    fuse_deepopt_activations: bool = False
    fusion_phases: tuple[str, ...] = ("forward", "backward")
    share_compatible_packs: bool = False
    fuse_packed_multiply: bool = False
    qualify_native_group_fusion: bool = False
    fuse_norm_cat_body: bool = False
    donate_fresh_backward_add: bool = False
    native_conv_backward_order: str = "source"
    cuda_conv_dx_geometry: bool = False
    native_conv_workspace_reuse: str = "off"
    native_conv_dx_batch_size: int = 1
    fuse_norm_activation_residuals: bool = False
    norm_activation_fusion_stage: str = "off"
    fuse_swiglu_residuals: bool = False
    swiglu_fusion_stage: str = "off"
    max_pointwise_fusion_length: int = 128
    callsite_occurrence: int = 0
    saved_activation_policy: Any = None
    bf16_policy: Any = None
    activation_memory: Any = None
    packed_activation_offload: Any = None
    packed_offload_selection_mode: str = "peak"
    packed_offload_peak_profile: Any = None
    collect_saved_state_metadata: bool = False


KernelCodegen = Callable[[IRNode, KernelCodegenContext], Callable[..., Any]]
NativeFallback = Callable[[IRNode, tuple[Any, ...], Mapping[str, Any]], Any]


@dataclass(frozen=True)
class OperatorExtension:
    """Implementation of a custom operator with explicit tensor results.

    ``lowering`` receives a :class:`LoweringContext` and must call
    ``context.add_node(fx_node, ir_node)`` before returning the node.
    ``triton_codegen`` receives already-rendered positional arguments and
    keyword arguments and returns a Triton expression for one program element.
    ``kernel_codegen`` can compile a standalone callable for operators that do
    not fit the elementwise fusion model. Fusion is only safe for pure
    operators, so impure extensions are never scheduled into Triton kernels.
    ``native_fallback(node, args, kwargs)`` owns ordinary autograd when a
    standalone kernel is used in a differentiable graph without training
    codegen. `training_codegen` returns an executable that owns its complete
    forward/saved-state/backward contract. Without either, standalone code
    generation is restricted to non-differentiable calls. Multi-result kernels
    return a tuple in the order declared by `IRNode.outputs()`.
    Fused pointwise extensions instead register an explicit backward rule.
    """

    name: str
    node_type: type[IRNode]
    matcher: FxMatcher
    lowering: Lowering
    triton_codegen: TritonCodegen | None = None
    can_fuse: FusionPredicate | None = None
    kernel_codegen: KernelCodegen | None = None
    pure: bool = True
    native_fallback: NativeFallback | None = None
    training_codegen: KernelCodegen | None = None
    tilelang_codegen: TritonCodegen | None = None
    decomposes: bool = False


class BackendRegistry:
    """Per-backend registry; pass an instance to ``compile_fx`` to extend it."""

    def __init__(self) -> None:
        self._extensions: dict[str, OperatorExtension] = {}
        from .autodiff.backward_rules import BWD_RULES_REGISTRY

        self._backward_rules = dict(BWD_RULES_REGISTRY)
        self._multi_output_rules: set[str] = set()
        from .linear import extension

        self.register(extension())
        from .native_linear import extension as native_extension

        self.register(native_extension())
        from .native_stateful import extension as stateful_extension, exp_extension, storage_primitive_extension

        self.register(stateful_extension())
        self.register(exp_extension())
        self.register(storage_primitive_extension())
        from .autodiff.residual_primitives import nearest2d_extension

        self.register(nearest2d_extension())
        from .native_attention import extension as attention_extension

        self.register(attention_extension())
        from .native_operations import extension as spatial_extension

        self.register(spatial_extension())
        from .native_mha import extension as mha_extension

        self.register(mha_extension())
        from deepopt.operations import extension as registered_operation_extension

        operation, backward = registered_operation_extension()
        self.register(operation)
        self.register_backward_rule(operation.name, backward)

    def register_backward_rule(self, name: str, rule: Callable, *, multi_output: bool = False) -> None:
        """Register an extension's first-order derivative on this registry only."""
        if not isinstance(name, str) or not name.isidentifier() or not callable(rule):
            raise ValueError("A backward rule requires an identifier and callable")
        if name in self._backward_rules:
            raise KeyError(f"Backward rule {name!r} is already registered")
        self._backward_rules[name] = rule
        if multi_output:
            self._multi_output_rules.add(name)

    def multi_output_backward_rule(self, name: str) -> bool:
        """Whether a VJP accepts cotangents aligned with all node results."""
        return name in self._multi_output_rules

    def backward_rule(self, name: str):
        try:
            return self._backward_rules[name]
        except KeyError as exc:
            raise NotImplementedError(
                f"No backward rule registered for {name!r}"
            ) from exc

    def has_backward_rule(self, name: str) -> bool:
        return name in self._backward_rules

    def register(self, extension: OperatorExtension) -> None:
        if not isinstance(extension, OperatorExtension):
            raise TypeError("extension must be an OperatorExtension")
        if type(extension.decomposes) is not bool:
            raise TypeError("extension.decomposes must be a boolean")
        if not extension.name or not extension.name.isidentifier():
            raise ValueError("extension names must be non-empty Python identifiers")
        if extension.name in self._extensions:
            raise KeyError(
                f"Operator extension {extension.name!r} is already registered"
            )
        if not isinstance(extension.node_type, type) or not issubclass(
            extension.node_type, IRNode
        ):
            raise TypeError(f"{extension.name!r}.node_type must inherit IRNode")
        for callback_name in ("matcher", "lowering"):
            if not callable(getattr(extension, callback_name)):
                raise TypeError(f"{extension.name!r}.{callback_name} must be callable")
        for callback_name in (
            "triton_codegen",
            "tilelang_codegen",
            "can_fuse",
            "kernel_codegen",
            "native_fallback",
            "training_codegen",
        ):
            callback = getattr(extension, callback_name)
            if callback is not None and not callable(callback):
                raise TypeError(f"{extension.name!r}.{callback_name} must be callable")
        self._extensions[extension.name] = extension

    def copy(self) -> BackendRegistry:
        clone = BackendRegistry()
        clone._extensions.update(self._extensions)
        clone._backward_rules = dict(self._backward_rules)
        clone._multi_output_rules = set(self._multi_output_rules)
        return clone

    def match_fx(
        self, node: Node, graph_module: GraphModule
    ) -> OperatorExtension | None:
        matches = [
            extension
            for extension in self._extensions.values()
            if extension.matcher(node, graph_module)
        ]
        if len(matches) > 1:
            names = ", ".join(extension.name for extension in matches)
            raise ValueError(
                f"FX node {node.name!r} matches multiple backend extensions: {names}"
            )
        return matches[0] if matches else None

    def for_node(self, node: IRNode) -> OperatorExtension | None:
        matches = [
            extension
            for extension in self._extensions.values()
            if isinstance(node, extension.node_type)
        ]
        if not matches:
            return None
        # Prefer the most specific registered node type when extensions use
        # inheritance, and reject ambiguous registrations instead of guessing.
        matches.sort(key=lambda ext: type(node).mro().index(ext.node_type))
        if len(matches) > 1:
            first_depth = type(node).mro().index(matches[0].node_type)
            second_depth = type(node).mro().index(matches[1].node_type)
            if first_depth == second_depth:
                raise TypeError(
                    f"Ambiguous operator extensions for {type(node).__name__}"
                )
        return matches[0]

    def __iter__(self):
        return iter(self._extensions.values())

    def __len__(self) -> int:
        return len(self._extensions)
