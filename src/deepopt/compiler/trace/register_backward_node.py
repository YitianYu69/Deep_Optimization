"""Compatibility re-exports for the backward rule registry."""

from ..autodiff.backward_rules import BWD_RULES_REGISTRY, register_backward_rule


BACKWARD_Node_REGISTRY = BWD_RULES_REGISTRY
register_backward_node = register_backward_rule

__all__ = ["BACKWARD_Node_REGISTRY", "register_backward_node", "register_backward_rule"]
