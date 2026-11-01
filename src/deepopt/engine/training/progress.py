"""Serializable progress within an epoch; no file I/O or batch replay."""

from dataclasses import dataclass
from copy import deepcopy

from torchmetrics import Metric

from deepopt.engine.reporting.metric_state import snapshot_metrics, restore_metrics


def type_name(value) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


@dataclass
class EpochProgress:
    metrics: Metric
    updates_at_start: int
    batches: int = 0
    samples: int | None = 0
    seconds: float = 0.0
    # Keep epoch-start logging stable when an update-level schedule resumes.
    learning_rates: list | None = None

    def state_dict(self) -> dict:
        return {"updates_at_start": self.updates_at_start, "batches": self.batches,
                "samples": self.samples, "seconds": self.seconds,
                "learning_rates": deepcopy(self.learning_rates),
                "metrics": snapshot_metrics(self.metrics)}

    @classmethod
    def restore(cls, state: dict, metric: Metric, device):
        restore_metrics(metric, state["metrics"], device)
        return cls(metric, state["updates_at_start"], state["batches"], state["samples"], state["seconds"],
                   deepcopy(state.get("learning_rates")))
