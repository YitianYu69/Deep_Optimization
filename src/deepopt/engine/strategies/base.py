"""Small replacement policies with explicit checkpoint configuration and state."""

from abc import ABC, abstractmethod


class Strategy(ABC):
    """Keep configuration separate from mutable training state.

    Configuration must contain comparable primitive values (no tensors).
    State may also contain tensors, as in an optimizer state_dict. Checkpoints
    store data only; callers construct their strategy before restoring it.
    """

    @abstractmethod
    def get_config(self) -> dict:
        """Return every setting that must agree when resuming training."""

    def state_dict(self) -> dict:
        """Stateless policies need not override these two state methods."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError(f"{type(self).__name__} does not accept mutable state")
