"""Compatibility imports; storage accounting lives in deepopt.memory.statistics."""

from .memory.statistics import ModelMemorySummary, log_model_memory, summarize_model_memory

__all__ = ["ModelMemorySummary", "log_model_memory", "summarize_model_memory"]
