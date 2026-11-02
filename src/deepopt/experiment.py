"""Connect independently defined components; no training or model logic here."""

from .config import ExperimentConfig
from .data.loaders import build_loaders
from .engine import Trainer
from .optim import build_optimizer, build_scheduler
from .runtime import resolve_device, seed_everything
from .distributed._process import initialize_distributed, rank, world_size
from .engine.execution.policy import DiagnosticPolicy
from .engine.execution.compilation import compile_model


def build_experiment(config: ExperimentConfig, *, model_factory, dataset_factory,
                     loader_factory=build_loaders,
                     optimizer_factory=build_optimizer, scheduler_factory=build_scheduler,
                     task_factory=None, metric_factory=None, trainer_factory=Trainer,
                     loss_fn=None, step_fn=None, validation_step_fn=None, metrics=None, metric_inputs=None):
    """Wire caller-owned model_factory() and dataset_factory() components.

    The dataset factory returns the actual training/validation datasets and
    binds its own sources, split rules, transforms, and optional application
    settings. No sample counts, feature dimensions, or task metadata are read
    here. The default loader helper is for finite map-style datasets; provide
    loader_factory for streams, custom collation, or different batching.
    Loss/step functions and metric instances use the same interface as Trainer;
    a Task or TrainingModule is optional. Omitted options do not override a
    caller's partially configured trainer_factory.
    """
    seed_everything(config.seed, config.runtime.deterministic)
    context = initialize_distributed(config.runtime.device)
    device = resolve_device(str(context.device))
    train_data, validation_data = dataset_factory()
    loader_options = ({"resumable": config.training.checkpoint_every_steps > 0}
                      if loader_factory is build_loaders else {})
    train_loader, validation_loader, generators = loader_factory(
        train_data, validation_data, batch_size=config.training.batch_size, seed=config.seed, **loader_options)
    model = model_factory().to(device)
    compile_model(model, config.runtime.compile)
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
        policy = DiagnosticPolicy.for_mode(config.runtime.diagnostics.get("mode", "balanced")).as_dict()
        options["diagnostics"] = DiagnosticPolicy(**{**policy, **config.runtime.diagnostics})
    if config.runtime.mode_schedule is not None:
        options["mode_schedule"] = {int(step): mode for step, mode in config.runtime.mode_schedule.items()}
    if config.runtime.non_blocking:
        options["non_blocking"] = True
    if config.experiment_identity:
        options["experiment_identity"] = config.experiment_identity
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
