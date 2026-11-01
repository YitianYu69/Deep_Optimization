"""Keep former engine import paths and serialized type names working.

Aliases point to the same module objects, so existing imports and patches do
not create duplicate implementations. New internal code uses the categorized
paths; this table is only for compatibility with existing recipes/checkpoints.
"""

from importlib import import_module
import sys


MODULE_ALIASES = {
    "trainer": "deepopt.engine.training.trainer",
    "loops": "deepopt.engine.training.loops",
    "steps": "deepopt.engine.training.steps",
    "state": "deepopt.engine.training.state",
    "progress": "deepopt.engine.training.progress",
    "runtime": "deepopt.engine.execution.runtime",
    "batch": "deepopt.engine.execution.batch",
    "precision": "deepopt.engine.execution.precision",
    "optimizer_step": "deepopt.engine.execution.optimizer_step",
    "optimization": "deepopt.engine.execution.optimization",
    "metrics": "deepopt.engine.reporting.metrics",
    "metric_lifecycle": "deepopt.engine.reporting.metric_lifecycle",
    "metric_state": "deepopt.engine.reporting.metric_state",
    "checkpoint": "deepopt.engine.checkpointing.checkpoint",
    "strategy_state": "deepopt.engine.checkpointing.strategy_state",
    "batch_size": "deepopt.memory.planning",
}


def register_aliases():
    package = sys.modules["deepopt.engine"]
    for old_name, target in MODULE_ALIASES.items():
        module = import_module(target)
        alias = f"deepopt.engine.{old_name}"
        sys.modules[alias] = module
        setattr(package, old_name, module)
        # Step checkpoints record MetricAccumulator's qualified class name.
        # Keep the same identity for all definitions from relocated modules.
        for value in vars(module).values():
            if isinstance(value, type) and value.__module__ == target:
                value.__module__ = alias
