"""Serializable progress owned by the training lifecycle."""

from dataclasses import dataclass


@dataclass
class TrainingState:
    completed_epochs: int = 0
    optimizer_steps: int = 0
    best_score: float | None = None
    best_epoch: int = 0
    monitor: str | None = None
    monitor_mode: str | None = None
