"""Rank-local terminal progress, separate from callbacks and checkpoint state."""

from typing import Literal

from torch.utils.data import DataLoader, IterableDataset


ProgressBarMode = bool | Literal["step", "epoch"]


def progress_bar_mode(value: ProgressBarMode) -> Literal["step", "epoch"] | None:
    if type(value) is bool:
        return "step" if value else None
    if isinstance(value, str) and value in ("step", "epoch"):
        return value
    raise ValueError("progress_bar must be a boolean or 'step' or 'epoch'")


def validate_progress_bar_every_steps(value):
    if type(value) is not int or value < 1:
        raise ValueError("progress_bar_every_steps must be a positive integer")


def _metric_postfix(phase, metrics):
    return {f"{phase}/{name}": value for name, value in metrics.items()
            if name not in {"samples", "batches", "seconds", "samples_per_second"}}


def _batch_count(loader):
    # Do not probe arbitrary iterables or fetch a batch to invent an ETA.
    if isinstance(loader, (list, tuple)) or (
        isinstance(loader, DataLoader) and not isinstance(loader.dataset, IterableDataset)
    ):
        try:
            return len(loader)
        except (TypeError, NotImplementedError, OverflowError):
            pass
    return None


class TrainingProgress:
    """Render host counters and metric snapshots supplied by the training loop."""

    def __init__(self, mode, *, epochs, completed_epochs, every_steps=1):
        from tqdm.auto import tqdm

        self.mode, self.epochs = mode, epochs
        self.every_steps = every_steps
        self._tqdm, self.bar = tqdm, None
        self._pending_steps, self._training_metrics = 0, {}
        if mode == "epoch":
            self.bar = tqdm(total=epochs, initial=completed_epochs,
                            desc="Training", unit="epoch", dynamic_ncols=True)

    def start_epoch(self, loader, progress, *, epoch, optimizer_steps, accumulation_steps, resuming):
        if self.mode != "step":
            return
        self._epoch = epoch
        self._pending_steps, self._training_metrics = 0, {}
        self._optimizer_step = optimizer_steps
        initial = optimizer_steps - progress.updates_at_start
        remaining = getattr(loader, "remaining_batches", None)
        if resuming and callable(remaining):
            batches = remaining()
        else:
            batches = _batch_count(loader)
            if resuming and batches is not None:
                batches = max(0, batches - progress.batches)
        total = (None if batches is None else
                 initial + (batches + accumulation_steps - 1) // accumulation_steps)
        self.bar = self._tqdm(total=total, initial=initial,
                              desc=f"Training epoch {self._epoch}/{self.epochs}",
                              unit="step", dynamic_ncols=True)

    def update_step(self, optimizer_step, *, metrics=None):
        if self.mode == "step" and self.bar is not None:
            self._optimizer_step = optimizer_step
            self._pending_steps += 1
            if metrics is not None:
                self._training_metrics = metrics
            if self._pending_steps >= self.every_steps:
                self._flush_steps()

    def _flush_steps(self):
        if self.mode == "step" and self.bar is not None and self._pending_steps:
            postfix = _metric_postfix("train", self._training_metrics)
            postfix["optimizer_step"] = self._optimizer_step
            self.bar.set_postfix(postfix, refresh=False)
            self.bar.update(self._pending_steps)
            self._pending_steps = 0

    def phase(self, name, *, metrics=None):
        if self.mode == "step" and self.bar is not None:
            if metrics is not None:
                self._training_metrics = metrics
            self._flush_steps()
            self.bar.set_description(f"{name} epoch {self._epoch}/{self.epochs}")

    def finish_epoch(self, row):
        if self.bar is None:
            return
        self._flush_steps()
        postfix = {**_metric_postfix("train", row["train"]), **_metric_postfix("val", row["validation"])}
        if self.mode == "step":
            postfix["optimizer_step"] = row["optimizer_steps"]
            self.bar.set_description(f"Training epoch {row['epoch']}/{self.epochs}", refresh=False)
        self.bar.set_postfix(postfix, refresh=False)
        if self.mode == "epoch":
            self.bar.update(1)
        else:
            self.close()

    def close(self):
        if self.bar is not None:
            try:
                self._flush_steps()
                self.bar.close()
            finally:
                self.bar = None
