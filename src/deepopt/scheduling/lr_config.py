"""Torch-free validation for native PyTorch LR policies."""

from .base import finite, position


def normalize_lr_schedule(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("training.lr_schedule must be a table")
    kind = raw.get("type")
    fields = {"constant": {}, "step_decay": {"step_size": None, "gamma": .1},
              "exponential_decay": {"gamma": None}, "linear_warmup": {"warmup_steps": None, "start_factor": .1},
              "cosine_decay": {"total_steps": None, "eta_min": 0.},
              "warmup_cosine": {"warmup_steps": None, "total_steps": None, "start_factor": .1, "eta_min": 0.},
              "piecewise": {"milestones": None, "gamma": .1}}
    if not isinstance(kind, str) or kind not in fields:
        raise ValueError("Unknown LR schedule type; choose " + ", ".join(fields))
    if set(raw) - {"type", *fields[kind]}:
        raise ValueError("Unknown LR schedule fields")
    config = {"type": kind, **fields[kind], **raw}
    for key in ("step_size", "warmup_steps", "total_steps"):
        if key in config:
            position(config[key], key)
    for key in ("gamma", "start_factor", "eta_min"):
        if key in config:
            config[key] = finite(config[key], key)
            if key == "eta_min":
                if config[key] < 0:
                    raise ValueError("eta_min must be nonnegative")
            elif not 0 < config[key] <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
    if kind == "cosine_decay" and config["total_steps"] < 2:
        raise ValueError("Cosine LR scheduling requires total_steps >= 2")
    if kind == "warmup_cosine" and config["total_steps"] <= config["warmup_steps"]:
        raise ValueError("warmup_cosine requires total_steps > warmup_steps")
    if kind == "piecewise":
        milestones = config["milestones"]
        if not isinstance(milestones, (list, tuple)):
            raise ValueError("LR milestones must be a list")
        for milestone in milestones:
            position(milestone, "milestone")
        if any(a >= b for a, b in zip(milestones, milestones[1:])):
            raise ValueError("LR milestones must be strictly increasing without duplicates")
        config["milestones"] = list(milestones)
    return config
