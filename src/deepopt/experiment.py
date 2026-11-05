"""Connect independently defined components; no training or model logic here."""

from dataclasses import asdict
from collections.abc import Mapping
from pathlib import Path

from .config import ExperimentConfig, config_from_dict
from ._config_resolution import config_scope, resolve_config
from .data.loaders import build_loaders
from .engine import Trainer
from .optimizer_factory import build_optimizer
from .scheduling.lr import build_scheduler
from .scheduling import ScheduleController
from .runtime import resolve_device, seed_everything
from .distributed._process import initialize_distributed, rank, world_size, collective_errors, require_equal
from .engine.execution.policy import DiagnosticPolicy
from .opt.policy import resolve_fp8_config


def build_experiment(config: ExperimentConfig | Mapping | str | Path | None = None, *, model_factory, dataset_factory,
                     loader_factory=build_loaders,
                     optimizer_factory=build_optimizer, scheduler_factory=build_scheduler,
                     task_factory=None, metric_factory=None, trainer_factory=Trainer,
                     loss_fn=None, step_fn=None, validation_step_fn=None, metrics=None, metric_inputs=None,
                     schedules=None, step_factory=None):
    """Wire caller-owned model_factory() and dataset_factory() components.

    The dataset factory returns the actual training/validation datasets and
    binds its own sources, split rules, transforms, and optional application
    settings. No sample counts, feature dimensions, or task metadata are read
    here. The default loader helper is for finite map-style datasets; provide
    loader_factory for streams, custom collation, or different batching.
    Loss/step functions and metric instances use the same interface as Trainer;
    a Task or TrainingModule is optional. Omitted options do not override a
    caller's partially configured trainer_factory.
    Dictionaries use the same schema as ExperimentConfig and are normalized
    before any factories run; custom factories still receive typed settings.
    A TOML path or builtin:NAME reference may be supplied directly as well.
    """
    snapshot = resolve_config(config, require_complete=True)
    resolved = config_from_dict(snapshot)
    with config_scope(snapshot):
        return _build_experiment(
            resolved, model_factory=model_factory, dataset_factory=dataset_factory,
            loader_factory=loader_factory, optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory, task_factory=task_factory,
            metric_factory=metric_factory, trainer_factory=trainer_factory,
            loss_fn=loss_fn, step_fn=step_fn, validation_step_fn=validation_step_fn,
            metrics=metrics, metric_inputs=metric_inputs, schedules=schedules, step_factory=step_factory)


def _build_experiment(config, *, model_factory, dataset_factory, loader_factory,
                      optimizer_factory, scheduler_factory, task_factory, metric_factory,
                      trainer_factory, loss_fn, step_fn, validation_step_fn, metrics,
                      metric_inputs, schedules, step_factory):
    seed_everything(config.seed, config.runtime.deterministic)
    context = initialize_distributed(config.runtime.device)
    device = resolve_device(str(context.device))
    configured = ScheduleController.from_config(config.schedules) if config.schedules else None
    if schedules is not None and not isinstance(schedules, ScheduleController):
        raise TypeError("schedules must be a ScheduleController")
    if schedules is not None and configured is not None and schedules.get_config() != configured.get_config():
        raise ValueError("Supplied coefficient schedules differ from configuration")
    schedules = schedules if schedules is not None else configured
    if step_factory is not None:
        if step_fn is not None or loss_fn is not None:
            raise ValueError("Choose step_factory or step_fn/loss_fn")
        step_fn = step_factory(schedules)
        if not callable(step_fn):
            raise TypeError("step_factory(schedules) must return a step_fn(model, batch)")
    train_data, validation_data = dataset_factory()
    loader_options = ({"resumable": config.training.checkpoint_every_steps > 0}
                      if loader_factory is build_loaders else {})
    train_loader, validation_loader, generators = loader_factory(
        train_data, validation_data, batch_size=config.training.batch_size, seed=config.seed, **loader_options)
    model = model_factory().to(device)
    from .engine.execution.configuration import prepare_execution_model
    snapshot = resolve_config(config)
    with collective_errors("FP8 configuration"):
        fp8 = resolve_fp8_config(model, config.runtime.fp8, config=snapshot)
    require_equal(asdict(fp8), "FP8 configuration")
    with collective_errors("FP8 model preparation"):
        model, fp8, _ = prepare_execution_model(model, snapshot, fp8)
    optimizer = optimizer_factory(model, config.training)
    scheduler = scheduler_factory(optimizer, config.training)
    options = {name: value for name, value in {
        "loss_fn": loss_fn, "step_fn": step_fn, "validation_step_fn": validation_step_fn,
        "metrics": metrics, "metric_inputs": metric_inputs,
    }.items() if value is not None}
    if config.training.scheduler_interval != "epoch":
        options["scheduler_interval"] = config.training.scheduler_interval
    if config.runtime.execution_mode is not None:
        options["execution_mode"] = config.runtime.execution_mode
    if config.runtime.diagnostics is not None:
        options["diagnostics"] = DiagnosticPolicy.from_dict(config.runtime.diagnostics)
    if config.runtime.mode_schedule is not None:
        options["mode_schedule"] = {int(step): mode for step, mode in config.runtime.mode_schedule.items()}
    if config.runtime.non_blocking:
        options["non_blocking"] = True
    if fp8.enabled:
        options["fp8"] = fp8
    if config.experiment_identity:
        options["experiment_identity"] = config.experiment_identity
    if schedules is not None:
        options["schedules"] = schedules
    trainer = trainer_factory(
        model, optimizer, device=device, precision=config.runtime.precision,
        checks=config.runtime.checks, accumulation_steps=config.training.accumulation_steps,
        grad_clip_norm=config.training.grad_clip_norm,
        task=task_factory() if task_factory is not None else None,
        scheduler=scheduler, metric_factory=metric_factory, **options,
    )
    if world_size() > 1:
        seed_everything(config.seed + rank(), config.runtime.deterministic)
    return trainer, train_loader, validation_loader, generators
