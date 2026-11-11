"""Forward numerical state; only requested backward tensors are retained."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerNormState:
    mean: object
    inverse: object
    packed: object = None
