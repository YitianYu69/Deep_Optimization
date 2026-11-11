"""Transient pooling backward operands, separate from public integer outputs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PoolState:
    routing: object = None
    coefficient: object = None
    factor: object = None
    public_indices: object = None
    samples: object = None
    coefficient_layout: str = "last"
