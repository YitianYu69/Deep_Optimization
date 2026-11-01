"""Optional TorchData adapter preserving exhausted step-checkpoint boundaries."""

from importlib.metadata import version

try:
    from torchdata.stateful_dataloader import StatefulDataLoader as _TorchDataLoader
except ImportError as exc:
    raise ImportError("Worker-state resume needs TorchData; install with pip install 'deep-optimization[workers]'") from exc


class StatefulDataLoader(_TorchDataLoader):
    """TorchData worker/prefetch recovery with DeepOpt's epoch-boundary contract.

    Keep batch size and worker count fixed for within-epoch restore. For random
    transforms, the dataset's state_dict/load_state_dict must include its RNGs;
    TorchData collects and restores that state in each worker. Use in_order=True.
    Arbitrary dataset state must be compatible with weights_only=True checkpoints.
    """
    supports_batch_size_change = False

    def __init__(self, *args, **kwargs):
        if kwargs.get("in_order", True) is not True:
            raise ValueError("StatefulDataLoader requires in_order=True for deterministic checkpoint order")
        super().__init__(*args, **kwargs)
        self._deepopt_resume = False

    def __iter__(self):
        if self._deepopt_resume:
            self._deepopt_resume = False
            # TorchData's public __iter__ restarts a restored finished iterator.
            # DeepOpt must see that exhausted iterator once to finalize the saved
            # epoch (including a final partial accumulation window) without replay.
            if self._initial_iter_for_state_dict:
                self._initial_iter_for_state_dict = False
            else:
                self._iterator = self._get_iterator()
            return self._iterator
        return super().__iter__()

    def _settings(self):
        return {"batch_size": getattr(self.batch_sampler, "batch_size", self.batch_size),
                "num_workers": self.num_workers, "drop_last": self.drop_last,
                "sampler": f"{type(self.sampler).__module__}.{type(self.sampler).__qualname__}"}

    def state_dict(self):
        return {"deepopt_format": 1, "torchdata": version("torchdata"),
                "settings": self._settings(), "loader": super().state_dict()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("deepopt_format") != 1:
            raise ValueError("Invalid DeepOpt StatefulDataLoader snapshot")
        if state.get("torchdata") != version("torchdata"):
            raise ValueError("Worker resume requires the same TorchData version")
        if state.get("settings") != self._settings():
            raise ValueError("Worker step resume requires the same batch size, worker count, sampler, and drop_last")
        if not isinstance(state.get("loader"), dict):
            raise ValueError("Invalid TorchData loader snapshot")
        super().load_state_dict(state["loader"])
        self._deepopt_resume = True

    def reset_epoch(self):
        if getattr(self, "_iterator", None) is not None:
            raise ValueError("Reconstruct a worker loader before restoring an epoch checkpoint")
        self.next_iter_state = None
        self._deepopt_resume = False

    def set_batch_size(self, size):
        if type(size) is not int or size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self._iterator is not None or self.next_iter_state is not None or self._deepopt_resume:
            raise ValueError("Resize a worker loader only before iteration; within-epoch rebatching is unsupported")
        if self.batch_sampler is None or self.batch_size is None:
            raise ValueError("Custom batch samplers must define their own resizing contract")
        self.batch_sampler.batch_size = size
        object.__setattr__(self, "batch_size", size)
