"""Public orchestration for FX-based model transformation."""

from copy import deepcopy
from dataclasses import dataclass, replace

from torch import fx, nn

from .do_nn.functional import validate_fp8_plan
from .fx.matcher import match_graph
from .fx.registry import ConversionRegistry, default_registry
from .fx.rewriter import rewrite_graph
from .fx.tracer import DOTracer


@dataclass(frozen=True)
class TransformReport:
    trace_mode: str
    original_type: str
    converted_modules: tuple[str, ...]
    function_calls: tuple[str, ...]
    untraced_modules: tuple[str, ...] = ()
    fallback_reason: str | None = None


class TransformationTraceError(RuntimeError):
    """The model could not be traced with the requested strategy"""



def _trace_full(
    model: nn.Module, 
    registry: ConversionRegistry, 
    *, 
    concrete_args: dict[str, object] | None = None
) -> fx.GraphModule:

    tracer = DOTracer(extra_leaf_types=registry.leaf_types)

    try:
        graph = tracer.trace(model, concrete_args=concrete_args)
    except Exception as exc:
        raise TransformationTraceError(
            f"Full FX tracing failed for {type(model).__qualname__}"
        ) from exc

    graph_module = fx.GraphModule(model, graph)

    # Keep original aliases, unused state, and registration order. Optimizer
    # checkpoints associate state with parameter iteration order, whereas FX
    # registers referenced attributes in execution order. Append FX constants.
    for attribute in ("_modules", "_parameters", "_buffers"):
        original = getattr(model, attribute)
        traced = getattr(graph_module, attribute)
        setattr(graph_module, attribute, {
            **original,
            **{name: value for name, value in traced.items() if name not in original},
        })
    graph_module._non_persistent_buffers_set.update(model._non_persistent_buffers_set)
    graph_module.training = model.training

    return graph_module



def _transform_submodules(
    model: nn.Module,
    registry: ConversionRegistry,
    *,
    plan: dict,
) -> TransformReport:
    replacements: dict[int, nn.Module] = {}
    visited = {id(model)}

    # Keep original objects alive while their identities are cache keys.
    original_modules = list(model.modules())

    converted_modules: list[str] = []
    function_calls: list[str] = []
    untraced_modules = ["<root>"]

    def visit(module: nn.Module, path: str) -> None:
        module_id = id(module)

        if module_id in visited or module_id in replacements:
            return
        visited.add(module_id)

        if registry.is_converted(module):
            return
        
        rule = registry.lookup(module)
        if rule is not None:
            replacements[module_id] = rule.converter(module, plan=plan)
            converted_modules.append(path)
            return
        
        try:
            graph_module = (
                module
                if isinstance(module, fx.GraphModule)
                else _trace_full(module, registry)
            )
        except TransformationTraceError:
            untraced_modules.append(path)
            graph_module = None
        
        if graph_module is not None:
            matches = match_graph(graph_module, registry)

            if matches:
                converted_modules.extend(
                    f"{path}.{node.target}"
                    for node in matches
                    if node.op == "call_module"
                )
                function_calls.extend(
                    f"{path}.{node.name}"
                    for node in matches
                    if node.op == "call_function"
                )

                replacements[module_id] = rewrite_graph(
                    graph_module,
                    matches,
                    registry,
                    plan=plan,
                    replacements=replacements,
                )

                # A successful trace may omit registered children that were
                # not called. Visit those too; the cache skips converted ones.

        for name, child in list(module._modules.items()):
            if child is not None:
                visit(child, f"{path}.{name}")
    
    for name, child in list(model._modules.items()):
        if child is not None:
            visit(child, name)
    
    # Install replacements at every registered reference, including aliases
    # outside the particular subgraph that discovered a shared module.
    parents = original_modules + [
        parent
        for replacement in replacements.values()
        for parent in replacement.modules()
    ]

    for parent in parents:
        for name, child in list(parent._modules.items()):
            replacement = replacements.get(id(child))
            if replacement is not None:
                parent._modules[name] = replacement

    return TransformReport(
        trace_mode="modules",
        original_type=(
            f"{type(model).__module__}.{type(model).__qualname__}"
        ),
        converted_modules=tuple(converted_modules),
        function_calls=tuple(function_calls),
        untraced_modules=tuple(untraced_modules),
    )
        



def transform_model(
    model: nn.Module,
    *,
    plan: dict,
    trace_mode: str = "full",
    inplace: bool = True,
    registry: ConversionRegistry | None = None,
    concrete_args: dict[str, object] | None = None,
) -> tuple[nn.Module, TransformReport]:
    """Replace supported calls with DO operations and report the result.

    ``full`` returns a GraphModule; ``modules`` keeps the root class and
    transforms its children. ``auto`` attempts full tracing and falls back to
    modules on a tracing failure. Untraced regions can contain unchanged ops.

    ``inplace=True`` shares parameters and may replace source children.
    ``inplace=False`` transforms a deep copy. Always use the returned model.
    ``concrete_args`` requires successful full tracing.
    """

    if trace_mode not in ("full", "modules", "auto"):
        raise ValueError("trace_mode must be 'full', 'modules', or 'auto'")
    if trace_mode == "modules" and concrete_args is not None:
        raise ValueError("concrete_args applies only to the full tracing")

    working_plan = validate_fp8_plan(plan)
    working_model = model if inplace else deepcopy(model)
    registry = default_registry() if registry is None else registry

    if trace_mode == "modules":
        report = _transform_submodules(
            working_model, registry,
            plan=working_plan,
        )
        return working_model, report

    if isinstance(working_model, fx.GraphModule):
        if concrete_args is not None:
            raise ValueError("concrete_args requires tracing an ordinary nn.Module")
        graph_module = working_model
    else:
        try:
            graph_module = _trace_full(working_model, registry, concrete_args=concrete_args)
        except TransformationTraceError as exc:
            if trace_mode != "auto" or concrete_args is not None:
                raise
            report = _transform_submodules(
                working_model,
                registry,
                plan=working_plan,
            )
            cause = exc.__cause__ or exc

            return working_model, replace(
                report,
                fallback_reason=f"{type(cause).__name__}: {cause}"
            )

    matches = match_graph(graph_module, registry)

    report = TransformReport(
        trace_mode="full",
        original_type=(
            f"{type(model).__module__}.{type(model).__qualname__}"
        ),
        converted_modules=tuple(
            node.target for node in matches if node.op == "call_module"
        ),
        function_calls=tuple(
            node.name for node in matches if node.op == "call_function"
        ),
    )

    graph_module = rewrite_graph(graph_module, matches, registry, plan=working_plan)
    return graph_module, report
