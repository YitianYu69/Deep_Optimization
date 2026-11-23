"""Expose only PyTorch's known no-op replication leaf to native AutoDiff.

Dynamo's deterministic CUDA ``F.pad`` path hides replication padding behind
``flat_apply_capture``. The native compiler does not accept arbitrary leaves.
This pass recognizes the exact upstream closures and static zero-pad ABI,
then restores an ordinary padding node with the existing metadata-only VJP.
"""

from copy import copy
import operator
from types import CodeType, FunctionType

import torch
from torch import fx
from torch.utils._pytree import LeafSpec, TreeSpec, tree_flatten


def _nested_code(function, name):
    if type(function) is not FunctionType:
        return None
    return next((value for value in function.__code__.co_consts
                 if isinstance(value, CodeType) and value.co_name == name), None)


def _cell(function, name):
    if type(function) is not FunctionType or function.__closure__ is None:
        return None
    try:
        index = function.__code__.co_freevars.index(name)
        return function.__closure__[index].cell_contents
    except (ValueError, IndexError):
        return None


def _attribute(gm, node):
    if (not isinstance(node, fx.Node) or node.op != "get_attr"
            or type(node.target) is not str or "." in node.target):
        return None
    # Dynamo registers both constants directly on its GraphModule. Avoid
    # traversing arbitrary objects or evaluating user-defined properties.
    return vars(gm).get(node.target)


def _same_spec(actual, expected):
    if type(actual) is not type(expected) or type(actual) not in (TreeSpec, LeafSpec):
        return False
    children = getattr(type(actual), "children", None)
    actual_children = children(actual) if children is not None else actual.children_specs
    expected_children = children(expected) if children is not None else expected.children_specs
    if actual.type is not expected.type or type(actual_children) is not list:
        return False
    if expected.context is None:
        if actual.context is not None:
            return False
    elif not (type(actual.context) is list and not actual.context and not expected.context):
        return False
    return (len(actual_children) == len(expected_children)
            and all(_same_spec(left, right) for left, right in zip(actual_children, expected_children)))


def _zero_replication_leaf(gm, node):
    try:
        from torch._decomp.decompositions import _replication_pad
        from torch._dynamo.decorators import nonstrict_trace
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
        from torch._higher_order_ops.flat_apply import flat_apply, is_valid_output, to_graphable
        from torch._higher_order_ops.invoke_leaf_function import _LeafCallable
    except ImportError:
        return None

    factory = getattr(TorchInGraphFunctionVariable, "_call_nonstrict_traceable_function", None)
    if (type(factory) is not FunctionType or node.op != "call_function" or type(node.target) is not FunctionType
            or node.target.__code__ is not _nested_code(factory, "flat_apply_capture")
            or node.kwargs or len(node.args) < 3
            or _cell(node.target, "flat_apply") is not flat_apply
            or _cell(node.target, "is_valid_output") is not is_valid_output
            or _cell(node.target, "to_graphable") is not to_graphable
            or type(_cell(node.target, "captured_spec")) is not LeafSpec):
        return None
    leaf = _attribute(gm, node.args[0])
    if type(leaf) is not _LeafCallable:
        return None
    patched = getattr(leaf, "_fn", None)
    if type(patched) is not FunctionType or patched.__code__ is not _nested_code(factory, "patched_fn"):
        return None
    wrapped = _cell(patched, "fn")
    if wrapped is not _replication_pad and (type(wrapped) is not FunctionType
            or wrapped.__code__ is not _nested_code(nonstrict_trace, "wrapped")
            or getattr(wrapped, "__wrapped__", None) is not _replication_pad
            or _cell(wrapped, "traceable_fn") is not _replication_pad):
        return None

    source, *pads = node.args[2:]
    if (not isinstance(source, fx.Node) or len(pads) not in (2, 4, 6)
            or not all(type(value) is int and value == 0 for value in pads)):
        return None
    # Only the positional (tensor, tuple/list of pads) input TreeSpec belongs
    # to the F.pad branch. Never unflatten or evaluate an unknown PyTree.
    spec = _attribute(gm, node.args[1])
    expected_specs = (tree_flatten(((0, tuple(pads)), {}))[1],
                      tree_flatten(((0, list(pads)), {}))[1])
    if not any(_same_spec(spec, expected) for expected in expected_specs) or len(node.users) != 1:
        return None
    output, = node.users
    if (output.op != "call_function" or output.target is not operator.getitem
            or output.args != (node, 0) or output.kwargs):
        return None
    return source, tuple(pads), output


def normalize_nonstrict_zero_replication_padding(gm):
    """Return a rewritten copy only for the trusted static no-op leaf ABI."""
    if not any(_zero_replication_leaf(gm, node) is not None for node in gm.graph.nodes):
        return gm
    graph = fx.Graph()
    copied = {}
    for node in gm.graph.nodes:
        copied[node] = graph.node_copy(node, lambda source: copied[source])
    result = fx.GraphModule(gm, graph)
    result.meta = copy(gm.meta)
    for node in tuple(result.graph.nodes):
        matched = _zero_replication_leaf(result, node)
        if matched is None:
            continue
        source, pads, output = matched
        with result.graph.inserting_before(node):
            replacement = result.graph.call_function(torch.nn.functional.pad, (source, pads, "replicate", None))
            replacement.meta = copy(output.meta)
        output.replace_all_uses_with(replacement)
        result.graph.erase_node(output)
        attributes = node.args[:2]
        result.graph.erase_node(node)
        for attribute in attributes:
            if not attribute.users:
                result.graph.erase_node(attribute)
    result.graph.lint()
    result.recompile()
    return result
