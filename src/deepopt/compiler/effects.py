"""Conservative FX alias and mutation analysis for backend selection."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import inspect
from typing import Any

import torch.fx as fx
import torch


_VIEW_NAMES = frozenset(
    {
        "view",
        "reshape",
        "flatten",
        "transpose",
        "permute",
        "t",
        "select",
        "narrow",
        "slice",
        "squeeze",
        "unsqueeze",
        "expand",
        "as_strided",
        "detach",
        "unbind",
        "split",
        "chunk",
        "getitem",
    }
)
_INPLACE_NAMES = frozenset(
    {
        "iadd",
        "isub",
        "imul",
        "imatmul",
        "itruediv",
        "ifloordiv",
        "ipow",
        "imod",
        "iand",
        "ior",
        "ixor",
        "ilshift",
        "irshift",
        "setitem",
        "__setitem__",
    }
)


@dataclass(frozen=True)
class WriteRewrite:
    node: str
    destination: str
    operation: str
    destination_dtype: str | None
    destination_stride: tuple | None
    reason: str = "private_single_use_destination"


@dataclass(frozen=True)
class EffectReport:
    alias_nodes: tuple[str, ...]
    mutation_nodes: tuple[str, ...]
    unsafe_nodes: tuple[str, ...] = ()
    functionalize_nodes: tuple[str, ...] = ()
    write_rewrites: tuple[WriteRewrite, ...] = ()

    @property
    def needs_functionalization(self) -> bool:
        return bool(self.mutation_nodes)


def _name(target: Any) -> str:
    return str(getattr(target, "__name__", target)).split(".")[0]


def _schema_effects(target: Any) -> tuple[bool, bool]:
    schema = getattr(target, "_schema", None)
    if schema is None:
        return False, False
    arguments = getattr(schema, "arguments", ())
    returns = getattr(schema, "returns", ())
    mutates = any(
        bool(getattr(getattr(arg, "alias_info", None), "is_write", False))
        for arg in arguments
    )
    aliases = any(getattr(ret, "alias_info", None) is not None for ret in returns)
    return mutates, aliases


def _python_call_mutates(target: Any, node: fx.Node) -> bool:
    """Catch Python API flags that are absent from an ATen operator schema."""
    if node.op != "call_function":
        return False
    if getattr(target, "_schema", None) is not None:
        return False
    try:
        bound = inspect.signature(target).bind_partial(*node.args, **node.kwargs)
    except (TypeError, ValueError):
        return False
    inplace = bound.arguments.get("inplace", False)
    out = bound.arguments.get("out")
    if (inplace is not False and inplace is not None) or out is not None:
        return True
    # Functional normalization can update caller-owned running statistics
    # without an ``inplace`` flag in its Python signature.
    name = getattr(target, "__name__", "")
    if name in {"batch_norm", "instance_norm"}:
        training = bound.arguments.get(
            "training" if name == "batch_norm" else "use_input_stats",
            name == "instance_norm",
        )
        has_running_state = any(
            bound.arguments.get(key) is not None
            for key in ("running_mean", "running_var")
        )
        return training is not False and has_running_state
    return False


def _written_inputs(node: fx.Node, target: Any) -> tuple[fx.Node, ...]:
    from .operations import is_deepopt_target, has_semantic_fresh_output, activation_call

    activation = activation_call(target, node.args, node.kwargs)
    if activation is not None:
        return (activation[1],) if activation[2] is True and isinstance(activation[1], fx.Node) else ()

    if is_deepopt_target(target) and has_semantic_fresh_output(target):
        return ()
    if getattr(target, "_deepopt_compiler_operation", False):
        from torch.utils._pytree import tree_leaves

        values = list(tree_leaves(node.args[3])) if len(node.args) > 3 else []
        if getattr(target, "writes_input", False) and node.args:
            values.extend(tree_leaves(node.args[0]))
        return tuple(value for value in values if isinstance(value, fx.Node))
    schema = getattr(target, "_schema", None)
    if schema is not None:
        written = []
        for index, argument in enumerate(schema.arguments):
            if not bool(
                getattr(getattr(argument, "alias_info", None), "is_write", False)
            ):
                continue
            value = (
                node.args[index]
                if index < len(node.args)
                else node.kwargs.get(argument.name)
            )
            if isinstance(value, fx.Node):
                written.append(value)
        if written:
            return tuple(written)
    if node.op == "call_function" and getattr(target, "__name__", "") in {
        "batch_norm",
        "instance_norm",
    }:
        try:
            bound = inspect.signature(target).bind_partial(*node.args, **node.kwargs)
            state = tuple(
                value
                for key in ("running_mean", "running_var")
                if isinstance((value := bound.arguments.get(key)), fx.Node)
            )
            if state:
                return state
        except (TypeError, ValueError):
            pass
    out = node.kwargs.get("out")
    if isinstance(out, fx.Node):
        return (out,)
    first = next((arg for arg in node.args if isinstance(arg, fx.Node)), None)
    return (first,) if first is not None else ()


def _private_pointwise_write(node: fx.Node, target: Any, aliases: set[str], gm, fresh_writes=()) -> bool:
    """An in-place ReLU/add on a fresh, single-use result has no visible write.

    The replacement must never write a placeholder, parameter, view, or value
    that another FX node can observe. It is then safe to use an out-of-place
    ReLU with the same value in the backend IR.
    """
    from .operations import activation_call, has_semantic_fresh_output
    import operator

    activation = activation_call(target, node.args, node.kwargs)
    source = activation[1] if activation else node.args[0] if node.args else None
    if not isinstance(source, fx.Node):
        return False
    name = _name(target)
    dropout = (target is torch.nn.functional.dropout and node.kwargs.get("inplace", node.args[3] if len(node.args) > 3 else False) is True
               or type(target) is torch.nn.Dropout and target.inplace)
    is_supported = (
        name in {"add_", "iadd", "__iadd__"}
        or activation is not None and activation[2] is True
        or dropout
    )
    if not is_supported or source.op not in {
        "call_function",
        "call_method",
        "call_module",
    }:
        return False
    positions = {value: index for index, value in enumerate(gm.graph.nodes)}
    # A classifier's final flatten may be a private view of a fresh pool
    # result. Prove sole ownership along that whole chain, rather than
    # treating arbitrary views or caller inputs as writable intermediates.
    owner = source
    private_views = set()
    if dropout:
        while ((owner.op == "call_method" and owner.target in {"view", "reshape", "flatten"})
               or owner.op == "call_function" and owner.target in {torch.flatten, torch.reshape}):
            if not owner.args or not isinstance(owner.args[0], fx.Node):
                return False
            parent = owner.args[0]
            if set(parent.users) != {owner}:
                return False
            private_views.update((owner.name, parent.name))
            owner = parent
    if (source.name in aliases - private_views or node not in source.users
            or any(user.op == "output" for user in source.users if user is not node)
            or any(positions[user] < positions[node] for user in source.users if user is not node)):
        return False
    from torch.utils._pytree import tree_leaves

    if sum(value is source for value in tree_leaves((node.args, node.kwargs))) != 1:
        return False
    producer = gm.get_submodule(owner.target) if owner.op == "call_module" else owner.target
    fresh = has_semantic_fresh_output(producer) or type(producer) in (
        torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.LayerNorm, torch.nn.GroupNorm)
    fresh |= producer in (operator.add, operator.sub, operator.mul, operator.truediv, torch.add,
                          torch.sub, torch.mul, torch.div, torch.clone, torch.relu, torch.nn.functional.adaptive_avg_pool2d)
    fresh |= owner.op == "call_method" and producer in {"add", "sub", "mul", "div", "clone"}
    fresh |= source.name in fresh_writes
    if not fresh:
        return False
    if "out" in node.kwargs:
        return False
    if node.kwargs.get("alpha", 1) != 1:
        return False
    return True


def _aot_safe_norm_write(target: Any) -> bool:
    # The native AOT region snapshots running statistics before its forward
    # updates them. A write to these buffers does not make the activation
    # output unsafe to wrap.
    return (
        target is torch.nn.functional.batch_norm
        or target is torch.nn.functional.instance_norm
    )


def analyze_fx_effects(gm: fx.GraphModule) -> EffectReport:
    # Discover the complete alias graph before granting any write proof.
    aliases = [node.name for node in gm.graph.nodes if node.op in {"call_function", "call_method"}
               and (_schema_effects(node.target)[1] or _name(node.target) in _VIEW_NAMES)]
    alias_bases = {source.name for node in gm.graph.nodes if node.name in aliases for source in node.all_input_nodes}
    mutations: list[str] = []
    functionalized: list[str] = []
    rewrites = []
    aot_safe_mutations: set[str] = set()
    for node in gm.graph.nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        target = (
            gm.get_submodule(node.target) if node.op == "call_module" else node.target
        )
        name = (
            _name(target) if node.op != "call_module" else type(target).__name__.lower()
        )
        mutates, alias = _schema_effects(target)
        from .operations import is_deepopt_target

        deepopt = is_deepopt_target(target)
        if deepopt:
            from .operations import has_semantic_fresh_output

            if has_semantic_fresh_output(target):
                aot_safe_mutations.add(node.name)
        from .operations import activation_call

        activation = activation_call(target, node.args, node.kwargs)
        if activation is not None and activation[2] is False:
            aot_safe_mutations.add(node.name)
        mutates |= deepopt
        if node.op == "call_module":
            mutates |= bool(getattr(target, "inplace", False))
        else:
            mutates |= (
                name.endswith("_")
                or name.startswith("__i")
                and name.endswith("__")
                or name in _INPLACE_NAMES
            )
            mutates |= bool(node.kwargs.get("inplace", False)) or "out" in node.kwargs
            mutates |= _python_call_mutates(target, node)
            alias |= name in _VIEW_NAMES
        if alias and node.name not in aliases:
            aliases.append(node.name)
        if mutates:
            mutations.append(node.name)
            if _private_pointwise_write(node, target, set(aliases) | alias_bases, gm, functionalized):
                functionalized.append(node.name)
                source = activation[1] if activation else node.args[0]
                metadata = source.meta.get("example_value", source.meta.get("val", source.meta.get("tensor_meta")))
                stride = getattr(metadata, "stride", None)
                operation = (activation[0] if activation else "dropout" if
                             target is torch.nn.functional.dropout or type(target) is torch.nn.Dropout else "add")
                rewrites.append(WriteRewrite(node.name, source.name, operation,
                    str(metadata.dtype) if metadata is not None else None,
                    tuple(stride() if callable(stride) else stride) if stride is not None else None))
            else:
                if _aot_safe_norm_write(target):
                    aot_safe_mutations.add(node.name)
    # Only the actual written operand and its aliases are unsafe. Reader
    # outputs are marked unsafe separately, without recursively treating them
    # as aliases. The former global closure propagated one ReLU write across
    # every later residual block in ResNet.
    alias_set = set(aliases)
    mutation_set = set(mutations) - set(functionalized)
    unsafe = mutation_set - aot_safe_mutations
    nodes = tuple(gm.graph.nodes)
    alias_edges: dict[str, set[str]] = {}
    for node in nodes:
        if node.name in alias_set:
            for source in node.all_input_nodes:
                alias_edges.setdefault(node.name, set()).add(source.name)
                alias_edges.setdefault(source.name, set()).add(node.name)
    for index, node in enumerate(nodes):
        if node.name not in mutation_set:
            continue
        target = (
            gm.get_submodule(node.target) if node.op == "call_module" else node.target
        )
        written = {value.name for value in _written_inputs(node, target)}
        alias_group = set(written)
        pending = list(written)
        while pending:
            for related in alias_edges.get(pending.pop(), ()):
                if related not in alias_group:
                    alias_group.add(related)
                    pending.append(related)
        unsafe.update(alias_group)
        for earlier in nodes[:index]:
            if earlier.name not in alias_group and any(
                source.name in alias_group for source in earlier.all_input_nodes
            ):
                unsafe.add(earlier.name)
    return EffectReport(
        tuple(aliases), tuple(mutations), tuple(sorted(unsafe)), tuple(functionalized), tuple(rewrites)
    )


def compile_with_inductor(gm: fx.GraphModule, example_inputs: tuple[Any, ...]):
    """Use the registered Inductor backend for graphs that need functionalization."""
    from torch._dynamo.backends.registry import lookup_backend

    compiler = lookup_backend("inductor")
    # Inductor may rewrite its input GraphModule, including the output node.
    # Clone the graph while retaining the original module parameters by
    # reference so direct compile_fx callers can still execute their gm.
    inductor_gm = fx.GraphModule(gm, copy.deepcopy(gm.graph))
    return compiler(inductor_gm, list(example_inputs))


def stochastic_depth_snapshot(mask, survival_probability):
    """The graph proof owns this small, exact, post-sampling allocation."""
    return mask.clone()


def snapshot_stochastic_depth(gm):
    """Expose private empty/bernoulli_/div_ masks as immutable coefficients.

    No random call or write is removed/replayed. Unknown users, escapes,
    aliases and writes after consumption refuse this narrowly owned pattern.
    """
    import operator

    nodes = tuple(gm.graph.nodes)
    positions = {node: index for index, node in enumerate(nodes)}
    decisions = []
    for draw in nodes:
        if draw.op != "call_method" or draw.target != "bernoulli_" or len(draw.args) != 2 or draw.kwargs:
            continue
        root, survival = draw.args
        if (not isinstance(root, fx.Node) or root.op != "call_function" or root.target is not torch.empty
                or type(survival) not in (int, float) or not 0 <= survival <= 1):
            continue
        metadata = draw.meta.get("example_value", draw.meta.get("val", draw.meta.get("tensor_meta")))
        if (metadata is None or metadata.requires_grad or metadata.dtype not in (torch.bfloat16, torch.float32)
                or len(metadata.shape) < 1 or any(size != 1 for size in metadata.shape[1:])):
            continue
        aliases, writers, consumers, invalid = {root, draw}, {draw}, set(), False
        changed = True
        while changed:
            changed = False
            for alias in tuple(aliases):
                for user in alias.users:
                    if user in writers:
                        continue
                    if (user.op == "call_method" and user.target == "div_" and user.args[0] in aliases
                            and len(user.args) == 2 and user.args[1] == survival and survival > 0 and not user.kwargs):
                        writers.add(user)
                        if user not in aliases:
                            aliases.add(user)
                            changed = True
                    elif ((user.op == "call_method" and user.target in ("mul", "__mul__"))
                          or user.op == "call_function" and user.target in (operator.mul, torch.mul)):
                        consumers.add(user)
                    else:
                        invalid = True
        if invalid or not consumers or len(writers) > 2:
            continue
        last_write = max(positions[node] for node in writers)
        first_consumer = min(consumers, key=positions.get)
        if positions[first_consumer] <= last_write:
            continue
        with gm.graph.inserting_before(first_consumer):
            snapshot = gm.graph.call_function(stochastic_depth_snapshot, (draw, survival))
            snapshot.meta = dict(draw.meta)
            snapshot.meta["deepopt_exact_mask"] = dict(survival_probability=survival,
                mask_shape=tuple(metadata.shape), scaled=len(writers) == 2)
        for consumer in consumers:
            consumer.args = fx.map_arg(consumer.args, lambda value: snapshot if value in aliases else value)
            consumer.kwargs = fx.map_arg(consumer.kwargs, lambda value: snapshot if value in aliases else value)
        decisions.append(dict(snapshot=snapshot.name, source=draw.name, survival_probability=survival,
                              consumers=tuple(sorted(node.name for node in consumers))))
    if decisions:
        gm.graph.lint()
        gm.recompile()
    return tuple(decisions)
