"""High-level orchestration built on the same runtime as manual loops."""

from collections.abc import Callable
from copy import deepcopy
from contextlib import nullcontext
import inspect
from math import isfinite

import torch
from torch import nn
from torchmetrics import Metric
from deepopt.distributed import _process as distributed
from deepopt._config_resolution import _UNSET, ResolvedConfig, config_scope, resolve_config
from deepopt.config import ema_config_from_dict
from deepopt.engine.execution.configuration import execution_config, prepare_execution_model, runtime_signature

from deepopt.tasks.base import Task
from deepopt.logger import init_logger
from deepopt.memory.statistics import log_model_memory, summarize_model_memory
from deepopt.engine.training.loops import EpochRunner
from deepopt.engine.training.progress import EpochProgress
from deepopt.engine.execution.runtime import TrainingRuntime
from deepopt.engine.training.state import TrainingState
from deepopt.engine.training.steps import StepAdapter
from deepopt.engine.training.ema import ExponentialMovingAverage
from deepopt.engine.strategies import ExplicitObjective, GradientStrategy, LossStrategy


logger = init_logger("deepopt.engine.trainer")

_DEFAULT_MONITOR = object()


class Trainer:
    """Control epochs, validation, scheduler timing, and callback boundaries.

    Pass loss_fn for (inputs, targets) batches, or step_fn(model, batch) for
    arbitrary computation. No subclass is required. Existing Tasks and
    TrainingModule hooks remain available. Strategies own loss normalization
    and gradient processing. For arbitrary backward/update order, own the loop
    and call TrainingRuntime directly instead of overriding trainer internals.
    fp8=True (or an enabled FP8Config) prepares supported operations through
    either core; use trainer.model and runtime.transformation_report afterward.
    """

    def __init__(self, model: nn.Module | TrainingRuntime, optimizer=None, *,
                 device: str | torch.device | None = None, precision: str | None = None,
                 accumulation_steps=_UNSET, grad_clip_norm=_UNSET,
                 task: Task | None = None, scheduler=_UNSET, scheduler_interval=_UNSET,
                 metric_factory: Callable[[], Metric] | None = None,
                 loss_strategy: LossStrategy | None = None, gradient_strategy: GradientStrategy | None = None,
                 loss_fn=None, step_fn=None, validation_step_fn=None, metrics=None, metric_inputs=None,
                 checks=None, ddp_options=_UNSET, execution_mode=_UNSET, diagnostics=_UNSET, mode_schedule=_UNSET,
                 non_blocking=None, experiment_identity=_UNSET, schedules=_UNSET, fp8=_UNSET, config=None):
        with distributed.collective_errors("trainer configuration"):
            explicit_training = {"accumulation_steps": accumulation_steps,
                                 "scheduler_interval": scheduler_interval}
            if scheduler is None:
                explicit_training.update(lr_schedule=None, lr_gamma=1.0)
            if isinstance(model, TrainingRuntime):
                base = resolve_config()
                base["runtime"] = model.config.get("runtime", {})
                base.setdefault("training", {})["grad_clip_norm"] = model.grad_clip_norm
                self._config_snapshot = resolve_config(config, base=base,
                                                       overrides={"training": explicit_training})
                if config is not None and runtime_signature(self._config_snapshot) != runtime_signature(model.config):
                    raise ValueError("Configure runtime-owned settings on the supplied runtime")
            else:
                self._config_snapshot = execution_config(
                    config, device=device if device is not None else _UNSET,
                    precision=precision if precision is not None else _UNSET,
                    checks=checks if checks is not None else _UNSET, grad_clip_norm=grad_clip_norm,
                    execution_mode=execution_mode, diagnostics=diagnostics, mode_schedule=mode_schedule,
                    non_blocking=non_blocking if non_blocking is not None else _UNSET,
                    fp8=fp8, model=model, ddp_options=ddp_options, training_overrides=explicit_training)
            training = self._config_snapshot.get("training", {})
            scheduler_was_explicit = scheduler is not _UNSET
            accumulation_steps = (training.get("accumulation_steps", 1)
                                  if accumulation_steps is _UNSET else accumulation_steps)
            scheduler_interval = (training.get("scheduler_interval", "epoch")
                                  if scheduler_interval is _UNSET else scheduler_interval)
            if experiment_identity is _UNSET:
                experiment_identity = self._config_snapshot.get("experiment_identity", {})
            if schedules is _UNSET:
                from deepopt.scheduling import ScheduleController
                definitions = self._config_snapshot.get("schedules", {})
                schedules = ScheduleController.from_config(definitions) if definitions else None
            if metric_factory is not None and (metrics is not None or metric_inputs is not None):
                raise ValueError("Choose metrics/metric_inputs or metric_factory, not both")
            if scheduler_interval not in ("epoch", "step"):
                raise ValueError("scheduler_interval must be epoch or step")
            if isinstance(model, TrainingRuntime):
                if (any(value is not None for value in (optimizer, device, precision, gradient_strategy, checks, non_blocking))
                        or any(value is not _UNSET and value is not None for value in
                               (grad_clip_norm, ddp_options, execution_mode, diagnostics, mode_schedule, fp8))):
                    raise ValueError("Configure optimizer, device, precision, gradients, and checks on the supplied runtime")
                self.runtime = model
        if not isinstance(model, TrainingRuntime):
            runtime_settings = self._config_snapshot.get("runtime", {})
            context = distributed.initialize_distributed(runtime_settings.get("device", "cpu"))
            with distributed.collective_errors("model device"):
                if optimizer is None:
                    if "learning_rate" not in training:
                        raise ValueError("Trainer requires an optimizer or configured training.learning_rate")
                    model = model.to(context.device)
                else:
                    if any(parameter.device != context.device for parameter in model.parameters()):
                        raise ValueError("Move the model to the training device before constructing its optimizer")
            with distributed.collective_errors("model preparation"):
                model, prepared_fp8, _ = prepare_execution_model(model, self._config_snapshot, fp8)
            with config_scope(self._config_snapshot):
                with distributed.collective_errors("optimizer and scheduler setup"):
                    if optimizer is None:
                        from deepopt.optimizer_factory import build_optimizer
                        optimizer = build_optimizer(model, self._config_snapshot)
                    if scheduler is _UNSET and (training.get("lr_schedule") is not None or training.get("lr_gamma", 1.0) != 1.0):
                        from deepopt.scheduling.lr import build_scheduler
                        scheduler = build_scheduler(optimizer, self._config_snapshot)
                self.runtime = TrainingRuntime(
                    model, optimizer, grad_clip_norm=grad_clip_norm, gradient_strategy=gradient_strategy,
                    checks=checks if checks is not None else _UNSET, ddp_options=ddp_options,
                    fp8=prepared_fp8, config=self._config_snapshot)
        with self.runtime.core.check_errors("trainer components"):
            self._config_snapshot["runtime"] = self.runtime.config.get("runtime", {})
            self._config_snapshot.setdefault("training", {}).update({
                "accumulation_steps": accumulation_steps, "scheduler_interval": scheduler_interval,
                "grad_clip_norm": self.runtime.grad_clip_norm,
            })
            if scheduler is _UNSET:
                if training.get("lr_schedule") is not None or training.get("lr_gamma", 1.0) != 1.0:
                    from deepopt.scheduling.lr import build_scheduler
                    scheduler = build_scheduler(self.runtime.optimizer, self._config_snapshot)
                else:
                    scheduler = None
            if scheduler_was_explicit and scheduler is None:
                self._config_snapshot["training"].update(lr_schedule=None, lr_gamma=1.0)
            if self.runtime.optimizer is None:
                raise ValueError("Trainer requires a runtime with an optimizer")
            self.runtime.require_complete_optimizer()
            if type(accumulation_steps) is not int or accumulation_steps < 1:
                raise ValueError("accumulation_steps must be a positive integer")
            if scheduler is not None and getattr(scheduler, "optimizer", None) is not self.optimizer:
                raise ValueError("Scheduler must belong to the trainer's optimizer")
            if scheduler is not None:
                try:
                    inspect.signature(scheduler.step).bind()
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError("Automatic training requires scheduler.step() with no required arguments; "
                                     "use a manual loop for metric-based or custom scheduling") from exc
            loss_strategy = loss_strategy if loss_strategy is not None else ExplicitObjective()
            if not isinstance(loss_strategy, LossStrategy):
                raise TypeError("loss_strategy must inherit LossStrategy")
            self.scheduler = scheduler
            self.scheduler_interval = scheduler_interval
            self._steps = StepAdapter(self.model, task, loss_fn=loss_fn, step_fn=step_fn,
                                      validation_step_fn=validation_step_fn, metrics=metrics, metric_inputs=metric_inputs)
            self._epochs = EpochRunner(self.runtime, self._steps, accumulation_steps=accumulation_steps,
                                       loss_strategy=loss_strategy, metric_factory=metric_factory)
            ema_settings = ema_config_from_dict(training.get("ema", {}))
            self._config_snapshot.setdefault("training", {})["ema"] = ema_settings
            self.ema = (ExponentialMovingAverage(self.model, ema_settings["decay"])
                        if ema_settings["enabled"] else None)
        from deepopt.engine.checkpointing.identity import validate_identity
        from deepopt.scheduling import ScheduleController
        with self.runtime.core.check_errors("experiment identity"):
            self.experiment_identity = validate_identity({} if experiment_identity is None else experiment_identity)
            self._config_snapshot["experiment_identity"] = deepcopy(self.experiment_identity)
            if schedules is not None and not isinstance(schedules, ScheduleController):
                raise TypeError("schedules must be a ScheduleController")
            if schedules is not None and schedules.completed_update != 0:
                raise ValueError("Use a fresh schedule controller and restore trainer progress from a checkpoint")
            self.schedules = schedules if schedules is not None and len(schedules) else None
            self._config_snapshot["schedules"] = self.schedules._input_config() if self.schedules is not None else {}
            schedule_state = self.schedules.state_dict() if self.schedules is not None else None
        self.runtime.core.require_equal({"accumulation_steps": accumulation_steps, "precision": self.precision,
                                         "ema": ema_settings,
                                         "experiment_identity": self.experiment_identity,
                                         "schedules": schedule_state,
                                         "scheduler_interval": scheduler_interval,
                                         "loss_strategy": (type(loss_strategy).__qualname__, loss_strategy.get_config()),
                                         "gradient_strategy": (type(self.runtime.gradient_strategy).__qualname__,
                                             self.runtime.gradient_strategy.get_config()) if self.runtime.gradient_strategy else None,
                                         "computation": self._steps.checkpoint_config()}, "trainer settings")
        self.state = TrainingState()
        self.at_epoch_boundary = True
        self.at_step_boundary = False
        self.epoch_progress: EpochProgress | None = None
        self._resume_pending = False
        self._stop_requested = False
        self._schedule_report = None
        summary = log_model_memory(self.model, self.optimizer)
        if self.ema is not None:
            logger.info("EMA parameter shadow storage: %.9f GB", self.ema.storage_bytes / 1_000_000_000)
            if summary.estimated_persistent_bytes is not None:
                logger.info("Estimated persistent training storage including EMA: %.9f GB",
                            (summary.estimated_persistent_bytes + self.ema.storage_bytes) / 1_000_000_000)

    @property
    def config(self):
        """Return the construction-time effective configuration as an owned copy."""
        return deepcopy(self._config_snapshot)

    @property
    def model(self):
        return self.runtime.model

    @property
    def optimizer(self):
        return self.runtime.optimizer

    @property
    def device(self):
        return self.runtime.device

    @property
    def precision(self):
        return self.runtime.precision

    @property
    def grad_clip_norm(self):
        return self.runtime.grad_clip_norm

    @property
    def accumulation_steps(self):
        return self._epochs.accumulation_steps

    @property
    def loss_strategy(self):
        return self._epochs.loss_strategy

    @property
    def task(self):
        return self._steps.task

    @task.setter
    def task(self, task):
        self._steps.task = task

    @property
    def metric_factory(self):
        return self._epochs.metric_factory

    @metric_factory.setter
    def metric_factory(self, factory):
        self._epochs.metric_factory = factory

    def new_metrics(self) -> Metric:
        """Create an independent metric collection, including for resume."""
        return self._epochs.new_metrics()

    def request_stop(self) -> None:
        """Callbacks may request a graceful stop after the current epoch."""
        self._stop_requested = True

    @property
    def execution_mode(self):
        return self.runtime.execution_mode

    def set_execution_mode(self, mode):
        with self.runtime.core.check_errors("trainer mode transition"):
            if not (self.at_epoch_boundary or self.at_step_boundary):
                raise RuntimeError("Change execution mode only at an optimizer-update or epoch boundary")
        self.runtime.set_execution_mode(mode)

    def _log_epoch_memory(self):
        if not distributed.is_main_process():
            return
        summary = summarize_model_memory(self.model, self.optimizer)

        def gb(value):
            return "unknown" if value is None else f"{value:.9f} GB"

        logger.info("Epoch %s memory (%s): model=%s optimizer=%s",
                    self.state.completed_epochs, self.device, gb(summary.model_gb), gb(summary.optimizer_gb))
        logger.info("Epoch %s estimated persistent training storage (model + gradients + optimizer): %s",
                    self.state.completed_epochs, gb(summary.estimated_persistent_gb))
        if self.ema is not None:
            logger.info("Epoch %s EMA parameter shadow storage: %.9f GB",
                        self.state.completed_epochs, self.ema.storage_bytes / 1_000_000_000)
            if summary.estimated_persistent_bytes is not None:
                logger.info("Epoch %s estimated persistent training storage including EMA: %.9f GB",
                            self.state.completed_epochs,
                            (summary.estimated_persistent_bytes + self.ema.storage_bytes) / 1_000_000_000)
        if self.device.type == "cuda":
            logger.info("Epoch %s CUDA memory (%s): allocated=%.9f GB reserved=%.9f GB",
                        self.state.completed_epochs, self.device,
                        torch.cuda.memory_allocated(self.device) / 1_000_000_000,
                        torch.cuda.memory_reserved(self.device) / 1_000_000_000)
            logger.info("Epoch %s CUDA peak memory since last reset (%s): allocated=%.9f GB reserved=%.9f GB",
                        self.state.completed_epochs, self.device,
                        torch.cuda.max_memory_allocated(self.device) / 1_000_000_000,
                        torch.cuda.max_memory_reserved(self.device) / 1_000_000_000)

    def fit(self, train_loader, validation_loader=None, *, epochs=_UNSET,
            monitor=_DEFAULT_MONITOR, monitor_mode=_UNSET, callbacks=(), step_callbacks=(),
            progress_bar: bool = True, config=None) -> TrainingState:
        """Control epochs and optional validation; delegate side effects to callbacks.

        Each callback(trainer, epoch_metrics) runs after validation and progress
        updates. It can log, save, or call request_stop(). Omitted fit settings
        inherit the trainer snapshot. A per-call config may change only epochs,
        monitor and monitor_mode. Epochs is the total target count.
        Step callbacks receive (trainer, step_event) after an optimizer update,
        with no pending accumulated gradients. Stop requests remain epoch-based.
        Without a validation loader, validation results are empty and best-score
        selection is disabled. Otherwise the default monitor is validation loss;
        pass monitor=None explicitly to disable selection while still evaluating.
        The optional progress bar reports completed epochs to stderr using only
        their computed metrics. Set progress_bar=False to disable it entirely.
        logging.memory_each_epoch in the trainer configuration enables a fresh
        memory report after each epoch's training and validation.
        """
        with self.runtime.core.check_errors("fit configuration"):
            base = resolve_config(ResolvedConfig(self._config_snapshot))
            explicit_training = {"epochs": epochs, "monitor_mode": monitor_mode}
            if monitor is not _DEFAULT_MONITOR:
                explicit_training["monitor"] = monitor
            selected = resolve_config(config, base=base, overrides={"training": explicit_training})
            if config is not None:
                before, after = deepcopy(base), deepcopy(selected)
                for snapshot in (before, after):
                    for field in ("epochs", "monitor", "monitor_mode"):
                        snapshot.get("training", {}).pop(field, None)
                if before != after:
                    raise ValueError("fit config may only override training.epochs, monitor, and monitor_mode")
            fit_settings = selected.get("training", {})
            memory_each_epoch = selected.get("logging", {}).get("memory_each_epoch", False)
            if epochs is _UNSET:
                epochs = fit_settings.get("epochs")
            if monitor_mode is _UNSET:
                monitor_mode = fit_settings.get("monitor_mode", self.state.monitor_mode or "min")
            if type(epochs) is not int or epochs <= self.state.completed_epochs:
                raise ValueError("Total epochs must exceed the number already completed")
            if monitor_mode not in ("min", "max"):
                raise ValueError("monitor_mode must be min or max")
            if type(progress_bar) is not bool:
                raise ValueError("progress_bar must be a boolean")
            if monitor is _DEFAULT_MONITOR:
                monitor = fit_settings.get("monitor", "loss") if validation_loader is not None else None
            if monitor is not None and (not isinstance(monitor, str) or not monitor):
                raise ValueError("monitor must be a nonempty metric name or None")
            if validation_loader is None and monitor is not None:
                raise ValueError("A monitored metric requires a validation loader; omit monitor or use monitor=None")
            if self.state.monitor_mode is not None and (monitor, monitor_mode) != (self.state.monitor, self.state.monitor_mode):
                raise ValueError("Cannot change the monitored metric within a training run")
            callbacks = tuple(callbacks)
            step_callbacks = tuple(step_callbacks)
        self.state.monitor, self.state.monitor_mode = monitor, monitor_mode
        self.runtime.core.require_equal({"epochs": epochs, "monitor": monitor, "monitor_mode": monitor_mode,
                                         "callbacks": len(callbacks), "step_callbacks": len(step_callbacks),
                                         "memory_each_epoch": memory_each_epoch,
                                         "validation": validation_loader is not None}, "fit settings")
        for callback in callbacks:
            start = getattr(callback, "on_fit_start", None)
            if callable(start):
                start(self, train_loader, validation_loader)
        step_callbacks = tuple(callback.on_optimizer_step for callback in callbacks
                               if callable(getattr(callback, "on_optimizer_step", None))) + step_callbacks
        self._stop_requested = False
        progress = None
        if progress_bar and distributed.is_main_process():
            from tqdm.auto import tqdm

            progress = tqdm(total=epochs, initial=self.state.completed_epochs,
                            desc="Training", unit="epoch", dynamic_ncols=True)
        logger.info("Training started: completed=%s target=%s device=%s precision=%s accumulation=%s",
                    self.state.completed_epochs, epochs, self.device, self.precision, self.accumulation_steps)
        try:
            while self.state.completed_epochs < epochs and not self._stop_requested:
                learning_rates = deepcopy(
                    self.epoch_progress.learning_rates
                    if self._resume_pending and self.epoch_progress.learning_rates is not None else
                    [group.get("lr") for group in self.optimizer.param_groups])
                lr = learning_rates[0] if learning_rates else None
                training = self.train_epoch(train_loader, step_callbacks=step_callbacks)
                validation = self.evaluate(validation_loader) if validation_loader is not None else {}
                if monitor is not None:
                    if monitor not in validation:
                        raise ValueError(f"Validation did not produce monitored metric {monitor!r}; "
                                         "supply explicit loss reporting, choose a reported metric, or use monitor=None")
                    score, best = validation[monitor], self.state.best_score
                    if not isfinite(score):
                        raise ValueError("The monitored metric must be finite for best-checkpoint selection; "
                                         "use monitor=None when intentionally allowing non-finite reports")
                    improved = best is None or (score > best if monitor_mode == "max" else score < best)
                    if improved:
                        self.state.best_score, self.state.best_epoch = score, self.state.completed_epochs
                row = {"epoch": self.state.completed_epochs, "learning_rate": lr,
                       "learning_rates": learning_rates, "optimizer_steps": self.state.optimizer_steps,
                       "train": training, "validation": validation}
                if self.schedules is not None:
                    row.update(self._schedule_report)
                logger.info("Epoch %s/%s: train=%s validation=%s",
                            self.state.completed_epochs, epochs, training, validation)
                if memory_each_epoch:
                    with self.runtime.core.check_errors("epoch memory logging"):
                        self._log_epoch_memory()
                for callback in callbacks:
                    with self.runtime.core.check_errors("epoch callback"):
                        callback(self, row)
                self._stop_requested = bool(distributed.reduce_number(int(self._stop_requested), torch.distributed.ReduceOp.MAX))
                if progress is not None:
                    postfix = {f"{phase}/{name}": value
                               for phase, metrics in (("train", training), ("val", validation))
                               for name, value in metrics.items()
                               if name not in {"samples", "batches", "seconds", "samples_per_second"}}
                    progress.set_postfix(postfix, refresh=False)
                    progress.update(1)
        except BaseException as exc:
            # A failed validation or callback must not silently skip work when
            # fitting again. Recover from the last fully saved checkpoint.
            self.at_epoch_boundary = False
            self.at_step_boundary = False
            try:
                logger.error("Training interrupted at completed epoch %s, optimizer update %s",
                             self.state.completed_epochs, self.state.optimizer_steps)
            except Exception as log_error:
                exc.add_note(f"Could not log training interruption: {log_error}")
            raise
        finally:
            if progress is not None:
                progress.close()
        logger.info("Training %s: completed=%s updates=%s",
                    "stopped" if self._stop_requested else "finished",
                    self.state.completed_epochs, self.state.optimizer_steps)
        return self.state

    def train_epoch(self, loader, *, step_callbacks=()) -> dict:
        if not self.at_epoch_boundary and not self._resume_pending:
            raise RuntimeError("Previous epoch failed; restore the last checkpoint before continuing")
        if not self._resume_pending:
            distributed.set_loader_epoch(loader, self.state.completed_epochs)
        progress = (self.epoch_progress if self._resume_pending else
                    EpochProgress(self.new_metrics(), self.state.optimizer_steps,
                                  learning_rates=deepcopy([group.get("lr") for group in self.optimizer.param_groups])))
        self._resume_pending = False
        self.epoch_progress = progress
        self.at_epoch_boundary = False
        step_callbacks = tuple(step_callbacks)
        self.runtime.core.require_equal(len(step_callbacks), "step callback counts")

        selected_callbacks = []

        def prepare_coefficients():
            self.schedules.prepare_update(self.state.optimizer_steps + 1, epoch=self.state.completed_epochs + 1)
            if progress.scheduled_values_start is None:
                progress.scheduled_values_start = dict(self.schedules.values)

        def prepare_update():
            if not step_callbacks:
                return False
            next_step = self.state.optimizer_steps + 1
            selected_callbacks.clear()
            for callback in step_callbacks:
                # Objects or bound methods may declare a cheap cadence predicate.
                source = getattr(callback, "__self__", callback)
                due = getattr(source, "should_run_step", None)
                if due is None or due(next_step):
                    selected_callbacks.append(callback)
            self.runtime.core.require_equal([index for index, callback in enumerate(step_callbacks)
                                             if callback in selected_callbacks], "step callback cadence")
            return bool(selected_callbacks)

        def on_update(progress):
            self.state.optimizer_steps += 1
            if self.ema is not None:
                with self.runtime.core.check_errors("EMA update"):
                    self.ema.update(self.model)
            if self.schedules is not None:
                self.schedules.complete_update(self.state.optimizer_steps)
            if self.scheduler is not None and self.scheduler_interval == "step":
                self.scheduler.step()
            self.at_step_boundary = True
            try:
                event = {"epoch": self.state.completed_epochs + 1,
                         "optimizer_step": self.state.optimizer_steps,
                         "batches_processed": progress.batches, "samples_processed": progress.samples}
                if self.schedules is not None:
                    event["scheduled_values"] = dict(self.schedules.values)
                for callback in selected_callbacks:
                    with self.runtime.core.check_errors("step callback"):
                        callback(self, event)
            finally:
                self.at_step_boundary = False

        try:
            results = self._epochs.train(loader, progress, on_update=on_update,
                                         synchronize_updates=prepare_update,
                                         on_window_start=prepare_coefficients if self.schedules is not None else None)
        except BaseException:
            if self.schedules is not None:
                self.schedules.abort_update()
            raise
        # A resumed final-step checkpoint may have no new batches this epoch.
        if (self.scheduler is not None and self.scheduler_interval == "epoch"
                and self.state.optimizer_steps > progress.updates_at_start):
            self.scheduler.step()
        self.state.completed_epochs += 1
        if self.schedules is not None:
            self._schedule_report = {"scheduled_values_start": deepcopy(progress.scheduled_values_start),
                                     "scheduled_values": dict(self.schedules.values)}
        self.at_epoch_boundary = True
        self.epoch_progress = None
        return results

    def evaluate(self, loader) -> dict:
        if self.ema is None:
            with self.schedules.read_only() if self.schedules is not None else nullcontext():
                return self._epochs.evaluate(loader)
        context = self.ema.average_parameters(self.model)
        entered = False
        try:
            with self.runtime.core.check_errors("EMA validation setup"):
                context.__enter__()
                entered = True
            with self.schedules.read_only() if self.schedules is not None else nullcontext():
                return self._epochs.evaluate(loader)
        finally:
            with self.runtime.core.check_errors("EMA validation restore"):
                if entered:
                    context.__exit__(None, None, None)
