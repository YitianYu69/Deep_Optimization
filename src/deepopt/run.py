"""Run-directory management and the train / validate / checkpoint sequence."""

from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import tempfile

from .config import config_from_dict
from ._config_resolution import _UNSET, ResolvedConfig, config_scope, resolve_config
from deepopt.engine.checkpointing.checkpoint import environment, read_checkpoint, require_step_loader, restore_checkpoint, save_checkpoint
from deepopt.engine.checkpointing.checkpoint import restore_schedules_for_evaluation
from deepopt.memory.planning import apply_batch_size_plan, plan_batch_size
from .baseline import baseline_sample_batch, build_baseline_experiment
from .logger import init_logger, logging_session, log_output
from .callbacks import EveryNSteps
from .distributed import _process as distributed


logger = init_logger(__name__)


def _write_text(path: Path, text: str) -> None:
    if not distributed.is_main_process():
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False, suffix=".tmp") as file:
            temporary = Path(file.name)
            file.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append_json_record(path: Path, record: dict) -> None:
    if not distributed.is_main_process():
        return
    previous = path.read_text() if path.exists() else ""
    if previous and not previous.endswith("\n"):
        previous += "\n"
    _write_text(path, previous + json.dumps(record) + "\n")


def train(config=None, *, output=None, resume=None, epochs=None, batch_size=None,
          batch_size_mode="auto", sample_batch_fn=None, memory_budget_bytes=None,
          checkpoint_every_steps=None, progress_bar=True, experiment_factory=build_baseline_experiment,
          execution_mode=None) -> Path:
    """Run a configured experiment with artifacts and optional memory planning.

    config accepts an ExperimentConfig, a dictionary, a TOML path, or builtin:NAME.
    batch_size overrides the configured size. Auto may adjust it before fresh
    or resumed training; manual keeps it fixed. Progress writes to stderr, while
    completed epoch records are JSON on stdout.
    """
    if type(progress_bar) is not bool:
        raise ValueError("progress_bar must be a boolean")
    if batch_size_mode not in ("auto", "manual"):
        raise ValueError("batch_size_mode must be auto or manual")
    if sample_batch_fn is not None and not callable(sample_batch_fn):
        raise TypeError("sample_batch_fn must be callable")
    if memory_budget_bytes is not None and (type(memory_budget_bytes) is not int or memory_budget_bytes < 1):
        raise ValueError("memory_budget_bytes must be a positive integer")
    payload = read_checkpoint(resume) if resume is not None else None
    training_overrides = {key: value for key, value in {
        "epochs": epochs, "batch_size": batch_size,
        "checkpoint_every_steps": checkpoint_every_steps,
    }.items() if value is not None}
    overrides = {"training": training_overrides} if training_overrides else {}
    if execution_mode is not None:
        overrides["runtime"] = {"execution_mode": execution_mode}
    snapshot = resolve_config(config, overrides=overrides,
                              base=ResolvedConfig(payload["config"]) if payload is not None else _UNSET)
    if not snapshot:
        raise ValueError("A new run requires --config or deepopt.load_config()")
    snapshot = resolve_config(snapshot, require_complete=True)
    config = config_from_dict(snapshot)
    if sample_batch_fn is None and experiment_factory is build_baseline_experiment:
        sample_batch_fn = partial(baseline_sample_batch, config)
    if payload is not None:
        completed = payload["state"]["completed_epochs"]
        if config.training.epochs < completed:
            raise ValueError("Total epochs cannot be less than the number already completed in the checkpoint")
        if config.training.epochs == completed and payload["kind"] == "step":
            raise ValueError("Total epochs must exceed completed epochs to finish a step checkpoint's active epoch")
    if resume is not None:
        if output is not None:
            raise ValueError("Resume continues in the checkpoint directory; omit --output")
        if Path(resume).name not in ("last.pt", "last_step.pt"):
            raise ValueError("Resume from last.pt or last_step.pt; best.pt is for evaluation")
        expected_kind = "step" if Path(resume).name == "last_step.pt" else "epoch"
        if payload["kind"] != expected_kind:
            raise ValueError("Checkpoint kind does not match the resume filename")
        run_dir = Path(resume).resolve().parent
        if payload["kind"] == "step" and (run_dir / "last.pt").exists():
            last_epoch = read_checkpoint(run_dir / "last.pt")
            if last_epoch["state"]["optimizer_steps"] > payload["state"]["optimizer_steps"]:
                raise ValueError("last_step.pt is older than last.pt; resume last.pt to retain newer progress")
    else:
        if output is None:
            raise ValueError("A new run requires --output (a new directory)")
        run_dir = Path(output).resolve()
        if run_dir.exists():
            raise ValueError(f"Output directory already exists: {run_dir}; use --resume or a new path")

    with logging_session(config, directory=None, failure_directory=run_dir.parent / "logs") as session:
        with config_scope(snapshot):
            trainer, train_loader, validation_loader, generators = experiment_factory(config)
        if config.training.checkpoint_every_steps and (payload is None or config.training.epochs > completed):
            require_step_loader(train_loader)
        if validation_loader is None:
            logger.info("No validation loader: best-checkpoint selection is disabled")
        history = []
        if payload is not None:
            skip_plan = batch_size_mode == "auto" and config.training.epochs == completed
            history = restore_checkpoint(payload, trainer, config, generators=generators,
                                         train_loader=train_loader, validation_loader=validation_loader,
                                         batch_size_mode="manual" if skip_plan else batch_size_mode,
                                         sample_batch_fn=sample_batch_fn,
                                         memory_budget_bytes=memory_budget_bytes)
            adjustment = getattr(trainer, "resume_adjustment", None)
            if adjustment is not None and skip_plan:
                adjustment.update(mode=batch_size_mode, status="skipped",
                                  reason="Requested epoch budget is already complete; no training memory plan is needed")
                logger.warning("Automatic batch-size planning skipped: %s", adjustment["reason"])
        else:
            adjustment = plan_batch_size(trainer, requested_batch_size=config.training.batch_size,
                                         mode=batch_size_mode, sample_batch_fn=sample_batch_fn,
                                         memory_budget_bytes=memory_budget_bytes)
            apply_batch_size_plan(trainer, train_loader, validation_loader, adjustment)
            distributed.rank_zero_call(lambda: run_dir.mkdir(parents=True, exist_ok=False))
        if adjustment is not None:
            config = replace(config, training=replace(config.training, batch_size=adjustment["new_batch_size"]))
            config = config_from_dict(asdict(config))
        session.bind(run_dir / "logs", config=config)
        logger.info("%s run in %s", "Resuming" if payload is not None else "Starting", run_dir)
        if adjustment is not None:
            record = {**adjustment, "event": "resume" if payload is not None else "start"}
            _append_json_record(run_dir / "batch_size.jsonl", record)
            if payload is not None:
                _append_json_record(run_dir / "resume.jsonl", record)

        def save(path, *, kind="epoch"):
            save_checkpoint(path, trainer, config, generators=generators, history=history,
                            kind=kind, train_loader=train_loader)

        def write_history():
            _write_text(run_dir / "metrics.jsonl", "".join(json.dumps(row) + "\n" for row in history))

        # last.pt is authoritative after interruption. Repair logs, and repair a
        # best-file write interrupted after last.pt when that last epoch was best.
        if history and payload["kind"] == "epoch" and trainer.state.best_epoch == trainer.state.completed_epochs:
            save(run_dir / "best.pt")
        write_history()
        _write_text(run_dir / "config.json", json.dumps(asdict(config), indent=2) + "\n")
        _write_text(run_dir / "environment.json", json.dumps(
            {**environment(), "device": str(trainer.device)}, indent=2) + "\n")

        # Finishing artifact repair must not require an unwanted extra training
        # epoch when last.pt already contains the requested completed epoch.
        if trainer.state.completed_epochs == config.training.epochs:
            logger.info("Run saved to %s", run_dir)
            return run_dir

        def record_epoch(trainer, row):
            history.append(row)
            save(run_dir / "last.pt")
            if trainer.state.best_epoch == trainer.state.completed_epochs:
                save(run_dir / "best.pt")
            write_history()
            log_output(json.dumps(row))

        def record_step(trainer, event):
            save(run_dir / "last_step.pt", kind="step")

        trainer.fit(train_loader, validation_loader, epochs=config.training.epochs,
                    monitor=config.training.monitor if validation_loader is not None else None,
                    monitor_mode=config.training.monitor_mode,
                    callbacks=[record_epoch],
                    step_callbacks=[EveryNSteps(record_step, config.training.checkpoint_every_steps)]
                    if config.training.checkpoint_every_steps else [],
                    progress_bar=progress_bar)
        logger.info("Run saved to %s", run_dir)
        return run_dir


def evaluate(checkpoint, *, device=None, experiment_factory=build_baseline_experiment) -> dict:
    payload = read_checkpoint(checkpoint)
    config = config_from_dict(payload["config"])
    if device is not None:
        config = replace(config, runtime=replace(config.runtime, device=device, precision="fp32"))
        config = config_from_dict(asdict(config))
    with logging_session(config, Path(checkpoint).resolve().parent / "logs"):
        with config_scope(resolve_config(config)):
            trainer, _, validation_loader, _ = experiment_factory(config)
        if validation_loader is None:
            raise ValueError("Evaluation requires a validation loader from experiment_factory")
        trainer.model.load_state_dict(payload["model"])
        restore_schedules_for_evaluation(payload, trainer)
        result = {"checkpoint": str(Path(checkpoint).resolve()),
                  "epoch": payload["state"]["completed_epochs"],
                  "device": str(trainer.device), "validation": trainer.evaluate(validation_loader)}
        logger.info("Evaluation: %s", json.dumps(result))
        return result
