from ..ir import (
    ExternKernelNode,
    IRGraph,
    TensorValue,
    TensorMeta,
    InputNode,
    OutputNode,
    register_uses,
)
from .tracer import SymbolicTensor
from .register_ops import get_trace_op_node

from typing import Any
from numbers import Number


def _tree_map(fn, value):
    if isinstance(value, tuple):
        return tuple(_tree_map(fn, x) for x in value)
    if isinstance(value, list):
        return [_tree_map(fn, x) for x in value]
    if isinstance(value, dict):
        return {k: _tree_map(fn, v) for k, v in value.items()}
    return fn(value)


class GraphBuilder:
    def __init__(self, graph: IRGraph, *, value_prefix: str = "v"):
        self.graph = graph
        self.value_prefix = value_prefix
        self.value_index = 0

    def new_value(self, meta: TensorMeta, *, name: str | None = None) -> TensorValue:
        if name is None:
            name = f"{self.value_prefix}{self.value_index}"
        self.value_index += 1
        return TensorValue(name=name, meta=meta)

    def input(
        self, index: int, meta: TensorMeta, *, name: str | None = None
    ) -> SymbolicTensor:
        value = self.new_value(meta, name=name)
        node = InputNode(index=index, value=value)
        value.producer = node
        self.graph.add(node)
        return SymbolicTensor(value=value, builder=self)

    def output(self, result: Any) -> OutputNode:
        ir_result = _tree_map(self.unwrap, result)
        node = OutputNode(result=ir_result)
        register_uses(node)
        self.graph.add(node)
        return node

    def wrap(self, value: Any):
        if isinstance(value, SymbolicTensor):
            if value.builder is not self:
                raise ValueError(
                    "Cannot mix SymbolicTensor objects from different GraphBuilder instance"
                )
            return value
        if isinstance(value, TensorValue):
            return SymbolicTensor(value, self)
        if isinstance(value, Number):
            return value

        raise TypeError(f"Cannot wrap value of type {type(value)}")

    def unwrap(self, value: Any):
        if isinstance(value, SymbolicTensor):
            if value.builder is not self:
                raise ValueError(
                    "Cannot emit an op with a SymbolicTensor from another GraphBuilder"
                )
            return value.value
        if isinstance(value, TensorValue):
            return value
        return value

    def emit(self, op_name, *args, **kwargs):
        op_definition = get_trace_op_node(op_name)
        ir_args = _tree_map(self.unwrap, args)
        ir_kwargs = _tree_map(self.unwrap, kwargs)

        out_meta = op_definition.infer_meta(ir_args, ir_kwargs)
        out_value = self.new_value(out_meta)

        node = op_definition.node_type(
            op=op_definition.name,
            args=ir_args,
            kwargs=ir_kwargs,
            value=out_value,
            target=op_definition.target,
        )

        out_value.producer = node
        register_uses(node)
        self.graph.add(node)
        return SymbolicTensor(value=out_value, builder=self)

    def emit_native(
        self,
        op_name: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output_meta: TensorMeta | None,
    ) -> SymbolicTensor:
        """Add a callable PyTorch operation to an AOT graph."""
        ir_args = _tree_map(self.unwrap, args)
        ir_kwargs = _tree_map(self.unwrap, kwargs)
        output = (
            self.new_value(output_meta)
            if output_meta is not None
            else TensorValue(name=f"{self.value_prefix}{self.value_index}", meta=None)
        )
        self.value_index += 1
        self.graph.add(
            ExternKernelNode(
                op=op_name,
                target=target,
                args=ir_args,
                kwargs=ir_kwargs,
                value=output,
            )
        )
        return SymbolicTensor(value=output, builder=self)

    def add(
        self,
        lhs: SymbolicTensor | TensorValue | Number,
        rhs: SymbolicTensor | TensorValue | Number,
    ):
        return self.emit("add", lhs, rhs)

    def sub(
        self,
        lhs: SymbolicTensor | TensorValue | Number,
        rhs: SymbolicTensor | TensorValue | Number,
    ):
        return self.emit("sub", lhs, rhs)

    def mul(
        self,
        lhs: SymbolicTensor | TensorValue | Number,
        rhs: SymbolicTensor | TensorValue | Number,
    ):
        return self.emit("mul", lhs, rhs)

    def div(
        self,
        lhs: SymbolicTensor | TensorValue | Number,
        rhs: SymbolicTensor | TensorValue | Number,
    ):
        return self.emit("div", lhs, rhs)

    def neg(self, value: SymbolicTensor | TensorValue | Number):
        return self.emit("neg", value)
