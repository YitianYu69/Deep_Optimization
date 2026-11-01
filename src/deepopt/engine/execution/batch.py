"""Transfer nested tensor batches without prescribing a dataset's structure."""

from collections import defaultdict
from collections.abc import Mapping
from copy import copy
from dataclasses import fields, is_dataclass

import torch


def move_to_device(value, device, *, non_blocking: bool = False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        items = {key: move_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
        if type(value) is dict:
            return items
        try:
            if isinstance(value, defaultdict):
                moved = type(value)(value.default_factory, items)
            else:
                moved = type(value)(items)
            # A constructor can accept a dictionary as metadata instead of
            # entries (ConfigParser does this). Keep the type only when it
            # preserves the transferred data, including tensor identities.
            if (isinstance(moved, Mapping) and len(moved) == len(items)
                    and all(key in moved and moved[key] is item for key, item in items.items())):
                return moved
            return items
        except (TypeError, ValueError):
            # Custom mappings may require unrelated constructor arguments.
            # A plain mapping preserves their data without mutating the source.
            return items
    if is_dataclass(value) and not isinstance(value, type):
        # Copy declared fields without rerunning constructors or post-init code.
        # object.__setattr__ also supports frozen and slotted dataclass batches.
        moved = copy(value)
        if moved is value:
            raise TypeError("A dataclass batch must support copying without modifying the original")
        for field in fields(value):
            if hasattr(value, field.name):
                object.__setattr__(moved, field.name, move_to_device(getattr(value, field.name), device, non_blocking=non_blocking))
        return moved
    if isinstance(value, tuple):
        items = [move_to_device(item, device, non_blocking=non_blocking) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value
