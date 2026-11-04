"""Validated built-in recipes available from source, wheels, and zip imports."""

from dataclasses import asdict
from importlib.resources import files
import re
import tomllib

from deepopt.config import ConfigError, ExperimentConfig, config_from_dict


__all__ = ["list_presets", "load_preset"]


def list_presets() -> tuple[str, ...]:
    """Return the sorted names of packaged TOML presets, without extensions."""
    return tuple(sorted(resource.name[:-5] for resource in files(__package__).iterdir()
                        if resource.is_file() and resource.name.endswith(".toml")))


def load_preset(name: str = "baseline", *, as_dict: bool = False) -> ExperimentConfig | dict:
    """Read a packaged preset and return validated settings or their plain mapping.

    Names are case-sensitive stems such as ``fp8``; ``fp8.toml`` is also accepted.
    Each call returns independent configuration data and does not import PyTorch.
    """
    if not isinstance(name, str):
        raise ConfigError("Preset name must be a simple name such as 'baseline' or 'fp8.toml'")
    stem = name.removesuffix(".toml")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", stem) is None:
        raise ConfigError("Preset name must be a simple name without directories")
    resource = files(__package__).joinpath(stem + ".toml")
    if not resource.is_file():
        raise ConfigError(f"Unknown built-in preset {name!r}; available presets: {', '.join(list_presets())}")
    with resource.open("rb") as stream:
        settings = config_from_dict(tomllib.load(stream))
    return asdict(settings) if as_dict else settings
