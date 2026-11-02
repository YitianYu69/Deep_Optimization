"""Torch-free schedule configuration and stable type identities."""

from copy import deepcopy
from dataclasses import fields
import json

from .base import Schedule
from .curves import Constant, LinearWarmup, LinearDecay, CosineDecay, ExponentialDecay, Piecewise, SequentialSchedule


TYPES = {"constant": Constant, "linear_warmup": LinearWarmup, "linear_decay": LinearDecay,
         "cosine_decay": CosineDecay, "exponential_decay": ExponentialDecay,
         "piecewise": Piecewise, "sequential": SequentialSchedule}


def serializable(value):
    # Roundtrip copies mutable metadata, rejects NaN, and normalizes tuples.
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Schedule metadata/state must be finite JSON-serializable data") from exc


def describe(schedule):
    if not isinstance(schedule, Schedule):
        raise TypeError("Schedules must inherit Schedule")
    name = next((name for name, cls in TYPES.items() if type(schedule) is cls),
                type(schedule).__module__ + "." + type(schedule).__qualname__)
    return {"type": name, "config": serializable(schedule.get_config()), "interval": schedule.interval}


def schedule_from_config(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("type"), str) or raw["type"] not in TYPES:
        raise ValueError("Unknown schedule type; choose " + ", ".join(TYPES))
    cls = TYPES[raw["type"]]
    options = deepcopy({k: v for k, v in raw.items() if k != "type"})
    unknown = options.keys() - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError("Unknown schedule fields: " + ", ".join(sorted(unknown)))
    if cls is SequentialSchedule:
        if not isinstance(options.get("schedules"), (list, tuple)):
            raise ValueError("sequential.schedules must be a list of schedule configurations")
        options["schedules"] = tuple(schedule_from_config(child) for child in options["schedules"])
    try:
        return cls(**options)
    except TypeError as exc:
        raise ValueError(f"Invalid {raw['type']} schedule configuration: {exc}") from exc


def config_for(schedule):
    config = serializable(schedule.get_config())
    if type(schedule) is SequentialSchedule:
        config["schedules"] = [config_for(child) for child in schedule.schedules]
    name = next((name for name, cls in TYPES.items() if type(schedule) is cls), None)
    if name is None:
        raise ValueError("Custom schedules are supplied through Python, not TOML")
    return {"type": name, **config}


def normalize_schedules(raw):
    if not isinstance(raw, dict) or any(not isinstance(name, str) or not name.strip() for name in raw):
        raise ValueError("schedules must be a table with nonempty coefficient names")
    return {name: config_for(schedule_from_config(config)) for name, config in raw.items()}


def stateless(schedule):
    if type(schedule) is SequentialSchedule:
        return all(stateless(child) for child in schedule.schedules)
    return type(schedule).state_dict is Schedule.state_dict
