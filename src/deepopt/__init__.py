"""Training entry points; importing configuration does not load PyTorch."""

from importlib import import_module


__all__ = ["Trainer", "TrainingRuntime", "TrainingModule", "MultiOptimizer", "FiniteCheck", "TensorEvent",
           "LossResult", "LossReport", "Checkpoint", "EveryNSteps",
           "initialize_distributed", "shutdown_distributed", "is_main_process", "DistributedContext", "TrainingCore", "DDPCore",
           "DiagnosticPolicy", "ExecutionMode", "CUDAGraphCore", "cuda_graph_core", "ScheduleController"]

_MODULES = {"Trainer": ".engine", "TrainingRuntime": ".engine",
            "ScheduleController": ".scheduling",
            "CUDAGraphCore": ".cuda_graph", "cuda_graph_core": ".cuda_graph",
            "TrainingModule": ".module", "MultiOptimizer": ".optimizers",
            "FiniteCheck": ".observers", "TensorEvent": ".observers",
            "LossResult": ".tasks.loss", "LossReport": ".tasks.loss",
            "Checkpoint": ".checkpoint", "EveryNSteps": ".callbacks",
            "initialize_distributed": ".distributed._process", "shutdown_distributed": ".distributed._process",
            "DistributedContext": ".distributed._process",
            "DiagnosticPolicy": ".engine.execution.policy", "ExecutionMode": ".engine.execution.policy",
            "is_main_process": ".distributed._process", "TrainingCore": ".engine.core", "DDPCore": ".distributed.ddp"}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
