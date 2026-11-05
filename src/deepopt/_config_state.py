"""Process-local storage for explicitly loaded configuration defaults."""

from copy import deepcopy
from threading import RLock


_lock = RLock()
_active_config = None


def install_config(config) -> None:
    """Install an owned, already validated configuration in one atomic assignment."""
    global _active_config
    owned = deepcopy(config)
    with _lock:
        _active_config = owned


def get_config():
    with _lock:
        return deepcopy(_active_config)


def reset_config() -> None:
    global _active_config
    with _lock:
        _active_config = None
