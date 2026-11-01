"""Public memory APIs, with estimates, storage statistics and planning separated."""

from importlib import import_module

__all__ = ["MemoryEstimationError", "TrainingMemoryEstimate", "estimate_training_memory",
           "ModelMemorySummary", "summarize_model_memory", "log_model_memory", "prepare_batch_size"]

_EXPORTS = {
    "MemoryEstimationError": ".estimation",
    "TrainingMemoryEstimate": ".estimation",
    "estimate_training_memory": ".estimation",
    "ModelMemorySummary": ".statistics",
    "summarize_model_memory": ".statistics",
    "log_model_memory": ".statistics",
    "prepare_batch_size": ".planning",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
