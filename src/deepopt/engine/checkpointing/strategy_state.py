"""Checkpoint data for replacement policies; never serialize strategy objects."""

from copy import deepcopy
import math

from deepopt.engine.training.progress import type_name
from deepopt.engine.execution.runtime import TrainingRuntime
from deepopt.engine.strategies import ClipGradNorm, ExplicitObjective


def _strategies(trainer):
    return {"loss": trainer.loss_strategy, "gradients": trainer.runtime.gradient_strategy}


def _primitive_config(value):
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_primitive_config(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive_config(item) for key, item in value.items())
    return False


def _description(strategy):
    if strategy is None:
        return None
    config = strategy.get_config()
    if not isinstance(config, dict) or not _primitive_config(config):
        raise ValueError("Strategy configuration must be a dictionary of finite primitive values")
    return {"type": type_name(strategy), "config": config}


def snapshot_strategies(trainer) -> dict:
    result = {}
    for name, strategy in _strategies(trainer).items():
        description = _description(strategy)
        if description is None:
            result[name] = None
        else:
            state = strategy.state_dict()
            if not isinstance(state, dict):
                raise ValueError("Strategy.state_dict() must return a dictionary")
            result[name] = {**description, "state": state}
    return deepcopy(result)


def validate_strategies(saved, trainer) -> None:
    if saved is None:
        # Format 4 predates pluggable strategies. Its existing trainer settings
        # describe exactly the default loss policy and optional norm clipping.
        gradient = trainer.runtime.gradient_strategy
        legacy_gradient = (gradient is None if trainer.grad_clip_norm is None else
                           type(gradient) is ClipGradNorm and gradient.max_norm == trainer.grad_clip_norm)
        if (type(trainer.runtime) is not TrainingRuntime or type(trainer.loss_strategy) is not ExplicitObjective
                or trainer.loss_strategy.get_config() != ExplicitObjective().get_config()
                or not legacy_gradient):
            raise ValueError("Checkpoint has no strategy state; resume with the original default strategies")
        return
    strategies = _strategies(trainer)
    if not isinstance(saved, dict) or set(saved) != set(strategies):
        raise ValueError("Checkpoint strategy names differ")
    for name, strategy in strategies.items():
        expected, actual = _description(strategy), saved[name]
        if expected is None and actual is None:
            continue
        if (expected is None or not isinstance(actual, dict) or not isinstance(actual.get("state"), dict)
                or {key: actual.get(key) for key in ("type", "config")} != expected):
            raise ValueError(f"Resume requires the same {name} strategy type and configuration")


def restore_strategies(saved, trainer) -> None:
    if saved is not None:
        for name, strategy in _strategies(trainer).items():
            if strategy is not None:
                strategy.load_state_dict(deepcopy(saved[name]["state"]))
