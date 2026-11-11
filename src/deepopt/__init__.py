"""Training entry points; importing configuration does not load PyTorch."""

from importlib import import_module


__all__ = ["Trainer", "TrainingRuntime", "TrainingModule", "MultiOptimizer", "FiniteCheck", "TensorEvent",
           "LossResult", "LossReport", "Checkpoint", "EveryNSteps",
           "initialize_distributed", "shutdown_distributed", "is_main_process", "DistributedContext", "TrainingCore", "DDPCore",
           "DiagnosticPolicy", "ExecutionMode", "CUDAGraphCore", "cuda_graph_core", "ScheduleController",
           "autocast", "FP8Config", "DelayedScalingConfig", "ActivationCompressionConfig", "fp8_scaling_scope", "fp8_checkpoint", "ConfigError", "ExperimentConfig", "TrainingConfig", "RuntimeConfig", "CompileConfig",
           "load_config", "get_config", "reset_config", "config_from_dict", "load_preset", "list_presets",
           "load_config_template", "write_config_template"]

_MODULES = {"Trainer": ".engine", "TrainingRuntime": ".engine",
            "autocast": ".opt._autocast", "FP8Config": ".config",
            "DelayedScalingConfig": ".config", "fp8_scaling_scope": ".opt.scaling.delayed",
            "ActivationCompressionConfig": ".config",
            "fp8_checkpoint": ".opt.scaling.delayed",
            "ConfigError": ".config", "ExperimentConfig": ".config", "TrainingConfig": ".config",
            "RuntimeConfig": ".config", "CompileConfig": ".config",
            "load_config": ".config", "config_from_dict": ".config",
            "get_config": ".config", "reset_config": ".config",
            "load_config_template": ".config", "write_config_template": ".config",
            "load_preset": ".configs", "list_presets": ".configs",
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
