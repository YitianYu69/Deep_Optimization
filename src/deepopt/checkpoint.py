"""Recipe-level checkpoint callback; history and paths stay outside Trainer."""

from copy import deepcopy
from collections import deque
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import tempfile

import torch
from deepopt.distributed import _process as distributed

from deepopt.memory.planning import loader_batch_size
from deepopt.engine.checkpointing.checkpoint import (read_checkpoint, read_checkpoint_progress, require_step_loader,
                                restore_checkpoint, save_checkpoint)


def _loader_generators(train_loader, validation_loader):
    """Follow public sampler links without iterating data or inspecting datasets."""
    generators, seen_generators, seen_owners = {}, set(), set()
    owners = deque()
    for phase, loader in (("train", train_loader), ("validation", validation_loader)):
        for suffix, owner in (("", loader), (".sampler", getattr(loader, "sampler", None)),
                              (".batch_sampler", getattr(loader, "batch_sampler", None))):
            owners.append((phase + suffix, owner))
    # Visit the original public locations first, preserving existing keys for
    # shared generators. Then follow BatchSampler/SkipBatchSampler chains.
    while owners:
        name, owner = owners.popleft()
        if owner is None or id(owner) in seen_owners:
            continue
        seen_owners.add(id(owner))
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator) and id(generator) not in seen_generators:
            generators[name] = generator
            seen_generators.add(id(generator))
        for attribute in ("sampler", "batch_sampler"):
            child = getattr(owner, attribute, None)
            if child is not None:
                owners.append((name + "." + attribute, child))
    return generators


class Checkpoint:
    """Pass one callback to fit(callbacks=[Checkpoint(directory)]).

    Keeps last.pt and (when monitored) best.pt at completed epochs. Opt in to
    last_step.pt with every_steps=N; N counts optimizer updates, not batches.
    Owns completed-epoch history and discovers ordinary loader RNG generators.
    Explicit generators can include additional recipe-owned torch.Generators.
    Attach from epoch zero, or call restore() before attaching to a resumed run.
    """

    def __init__(self, directory, *, every_steps: int = 0, config=None, generators=None):
        if type(every_steps) is not int or every_steps < 0:
            raise ValueError("every_steps must be a nonnegative integer; 0 disables step checkpoints")
        self.directory = Path(directory)
        self.every_steps = every_steps
        self.config = asdict(config) if is_dataclass(config) else deepcopy(config)
        self._explicit_generators = generators
        self.generators = {}
        self.history = []
        self._trainer = None
        self._train_loader = None

    def _bind(self, trainer, train_loader, validation_loader):
        self._trainer, self._train_loader = trainer, train_loader
        self.generators = (_loader_generators(train_loader, validation_loader)
                           if self._explicit_generators is None else dict(self._explicit_generators))

    def on_fit_start(self, trainer, train_loader, validation_loader):
        if len(self.history) != trainer.state.completed_epochs:
            raise ValueError("Attach Checkpoint before the first fit(), or call checkpoint.restore() "
                             "to recover its history before continuing")
        if self._trainer is not None and self._trainer is not trainer:
            raise ValueError("Use a new Checkpoint callback for another trainer, or call restore() on it")
        if self._trainer is None and any((self.directory / name).exists()
                                        for name in ("last.pt", "last_step.pt", "best.pt")):
            raise ValueError("Checkpoint directory already contains a run; call restore() or use a new directory")
        if self.every_steps:
            require_step_loader(train_loader)
        distributed.rank_zero_call(lambda: self.directory.mkdir(parents=True, exist_ok=True))
        self._bind(trainer, train_loader, validation_loader)

    def _save(self, trainer, name, *, kind="epoch"):
        save_checkpoint(self.directory / name, trainer, {} if self.config is None else self.config,
                        generators=self.generators, history=self.history,
                        kind=kind, train_loader=self._train_loader)

    def _write_history(self):
        distributed.rank_zero_call(self._write_history_local)

    def _write_history_local(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             delete=False, suffix=".tmp") as file:
                temporary = Path(file.name)
                for row in self.history:
                    file.write(json.dumps(row) + "\n")
            os.replace(temporary, self.directory / "metrics.jsonl")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def __call__(self, trainer, row):
        if trainer is not self._trainer or len(self.history) + 1 != trainer.state.completed_epochs:
            raise ValueError("Checkpoint callback received an unexpected trainer or epoch")
        self.history.append(deepcopy(row))
        self._save(trainer, "last.pt")
        if trainer.state.monitor is not None and trainer.state.best_epoch == trainer.state.completed_epochs:
            self._save(trainer, "best.pt")
        self._write_history()

    def should_run_step(self, optimizer_step: int) -> bool:
        return self.every_steps > 0 and optimizer_step % self.every_steps == 0

    def on_optimizer_step(self, trainer, event):
        if self.should_run_step(event["optimizer_step"]):
            self._save(trainer, "last_step.pt", kind="step")

    def restore(self, trainer, train_loader, validation_loader=None, *, path=None,
                batch_size_mode="manual", sample_batch_fn=None, memory_budget_bytes=None):
        """Restore into caller-built components; retain the saved history internally.

        By default choose the newest last.pt/last_step.pt by saved progress,
        preferring the completed epoch when update counts tie. No Python model
        reconstruction or loss inference is performed. Explicit paths must be
        last.pt or last_step.pt in this callback's directory.
        """
        candidates = []
        for name in ("last.pt", "last_step.pt"):
            candidate = self.directory / name
            if candidate.exists():
                kind, progress = read_checkpoint_progress(candidate)
                expected = "epoch" if name == "last.pt" else "step"
                if kind != expected:
                    raise ValueError("Checkpoint kind does not match its filename")
                candidates.append((candidate, progress))
        if not candidates:
            raise FileNotFoundError(f"No last.pt or last_step.pt in {self.directory}")
        newest = max(candidates, key=lambda entry: entry[1])
        selected = newest
        if path is not None:
            requested_path = Path(path)
            if requested_path.parent == Path("."):
                requested_path = self.directory / requested_path
            matching = [entry for entry in candidates if entry[0].resolve() == requested_path.resolve()]
            if not matching:
                raise ValueError("Restore a last.pt or last_step.pt from the callback's directory")
            selected = matching[0]
            if selected[1] < newest[1]:
                raise ValueError("Selected checkpoint is older than available progress; restore the newest checkpoint")
        payload = read_checkpoint(selected[0])
        config = deepcopy(payload["config"] if self.config is None else self.config)
        requested_size = loader_batch_size(train_loader)
        if self.config is None and requested_size is not None and "batch_size" in config.get("training", {}):
            # Saved metadata is not a new batch-size request. The caller's
            # reconstructed loader supplies that request when config is omitted.
            config["training"]["batch_size"] = requested_size
        generators = (_loader_generators(train_loader, validation_loader)
                      if self._explicit_generators is None else dict(self._explicit_generators))
        history = restore_checkpoint(payload, trainer, config, generators=generators,
                                     train_loader=train_loader, validation_loader=validation_loader,
                                     batch_size_mode=batch_size_mode, sample_batch_fn=sample_batch_fn,
                                     memory_budget_bytes=memory_budget_bytes)
        selected_size = loader_batch_size(train_loader)
        if selected_size is not None and "batch_size" in config.get("training", {}):
            config["training"]["batch_size"] = selected_size
        self.config = config
        self.history = history
        self._bind(trainer, train_loader, validation_loader)
        # Repair a best-file/history write interrupted after the durable last.pt.
        if (payload["kind"] == "epoch" and trainer.state.monitor is not None
                and trainer.state.best_epoch == trainer.state.completed_epochs and self.history):
            self._save(trainer, "best.pt")
        self._write_history()
        return trainer.state
