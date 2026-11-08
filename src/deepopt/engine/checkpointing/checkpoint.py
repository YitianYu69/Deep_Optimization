"""Versioned, atomic checkpoints at epoch and optimizer-update boundaries."""

from dataclasses import asdict, is_dataclass
from copy import deepcopy
from collections.abc import Mapping
import os
from pathlib import Path
import platform
import random
import tempfile
from importlib.metadata import version

import torch
from deepopt.distributed import _process as distributed
from deepopt._config_resolution import _merge, resolve_config
from deepopt.config import compile_config_from_dict, ema_config_from_dict, fp8_config_from_dict

from deepopt.engine.training.state import TrainingState
from deepopt.engine.training.progress import EpochProgress, type_name
from deepopt.engine.strategies.loss import LOSS_SCALING_VERSION
from deepopt.engine.execution.runtime import TrainingRuntime
from deepopt.engine.checkpointing.strategy_state import snapshot_strategies, validate_strategies, restore_strategies
from deepopt.memory.planning import loader_batch_size, plan_batch_size, apply_batch_size_plan
from deepopt.module import TrainingModule
from deepopt.optimizers import MultiOptimizer
from deepopt.opt._recipe import fp8_recipe
from deepopt.logger import init_logger
from .identity import snapshot_identity, validate_identity


logger = init_logger("deepopt.engine.checkpoint")


def _config_batch_size(config):
    training = config.get("training", {})
    return training.get("batch_size") if isinstance(training, Mapping) else None


def _saved_batch_size(payload):
    size = payload.get("batch_size", _config_batch_size(payload["config"]))
    if size is None and payload.get("train_loader") is not None:
        # Legacy built-in loader snapshots already recorded their physical size.
        state = payload["train_loader"].get("state", {})
        if isinstance(state, Mapping) and isinstance(state.get("sampler"), Mapping):
            size = state["sampler"].get("batch_size")
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError("Invalid checkpoint batch size")
    return size


def require_step_loader(loader) -> None:
    """Check the resume interface, independently of the loader's worker count.

    The loader owns correctness: its snapshot must describe the next batch
    consumed by training, accounting for sampler, prefetch, worker and dataset
    state as needed. Restoring an exhausted epoch must yield no more batches
    on the next iteration, so the trainer can finish that epoch exactly once.
    Method availability alone cannot verify those behavioral guarantees.
    """
    if (not callable(getattr(loader, "state_dict", None))
            or not callable(getattr(loader, "load_state_dict", None))):
        raise ValueError("Step checkpoints require a resumable loader with state_dict/load_state_dict; "
                         "provide a loader implementing the step-resume contract or set checkpoint_every_steps=0")


def _config_dict(config) -> dict:
    return asdict(config) if is_dataclass(config) else deepcopy(dict(config))


def _component_types(trainer) -> dict:
    def name(component):
        return f"{type(component).__module__}.{type(component).__qualname__}"
    components = {key: name(getattr(trainer, key)) for key in ("model", "optimizer", "scheduler", "task")}
    if isinstance(trainer.model, TrainingModule):
        components.update(wrapped_model=name(trainer.model.model), module_task=name(trainer.model.task))
    if type(trainer.runtime) is not TrainingRuntime:
        components["runtime"] = name(trainer.runtime)
    return components


def _trainer_settings(trainer) -> dict:
    settings = {**{key: getattr(trainer, key) for key in ("precision", "accumulation_steps", "grad_clip_norm")},
                "loss_scaling": LOSS_SCALING_VERSION}
    if trainer.runtime.distributed:
        settings["ddp"] = trainer.runtime.core.options
    if trainer.runtime.fp8.enabled:
        settings["fp8"] = asdict(trainer.runtime.fp8)
        settings["fp8_recipe"] = fp8_recipe(trainer.runtime.fp8.backend)
    if trainer.ema is not None:
        settings["ema"] = {"enabled": True, "decay": trainer.ema.decay}
    # Earlier checkpoints already used epoch scheduling. Keep that default's
    # metadata unchanged while rejecting a changed per-update schedule.
    if trainer.scheduler_interval != "epoch":
        settings["scheduler_interval"] = trainer.scheduler_interval
    computation = trainer._steps.checkpoint_config()
    if computation is not None:
        settings["computation"] = computation
    return settings


def _optimizer_topology(trainer) -> dict:
    """Bind positional optimizer state to the model parameters it belongs to."""
    parameter_names = {id(parameter): name for name, parameter in trainer.model.named_parameters()}

    def groups(optimizer):
        try:
            return [[parameter_names[id(parameter)] for parameter in group["params"]]
                    for group in optimizer.param_groups]
        except KeyError as exc:
            raise ValueError("Checkpoint optimizer contains a parameter outside the model") from exc

    topology = {"groups": groups(trainer.optimizer)}
    if isinstance(trainer.optimizer, MultiOptimizer):
        topology["children"] = [
            {"name": name, "type": type_name(child), "groups": groups(child)}
            for name, child in trainer.optimizer.optimizers.items()
        ]
    return topology


def _validate_model_state(saved, model) -> None:
    """Reject common incompatible states before load_state_dict can copy tensors."""
    current = model.state_dict()
    if not isinstance(saved, Mapping) or set(saved) != set(current):
        raise ValueError("Checkpoint model state keys differ from the current model")
    for name, expected in current.items():
        actual = saved[name]
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor):
                raise ValueError(f"Checkpoint model tensor {name!r} is invalid")
            # Lazy modules may materialize their parameters during state loading.
            if not torch.nn.parameter.is_lazy(expected) and expected.shape != actual.shape:
                raise ValueError(f"Checkpoint model tensor {name!r} has a different shape")
            if expected.dtype != actual.dtype:
                raise ValueError(f"Checkpoint model tensor {name!r} has a different dtype")


def environment() -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchmetrics": version("torchmetrics"),
            "cuda": torch.version.cuda, "platform": platform.platform()}


def _checkpoint_payload(trainer, config, *, generators, history, kind="epoch", train_loader=None):
    if kind not in ("epoch", "step"):
        raise ValueError("Checkpoint kind must be epoch or step")
    if kind == "epoch" and not trainer.at_epoch_boundary:
        raise ValueError("Checkpoints can only be saved at a completed epoch boundary")
    if kind == "step":
        if not trainer.at_step_boundary or trainer.epoch_progress is None:
            raise ValueError("Step checkpoints require a completed optimizer update boundary")
        if any(parameter.grad is not None for parameter in trainer.model.parameters()):
            raise ValueError("Step checkpoints cannot contain pending gradients")
        require_step_loader(train_loader)
    if len(history) != trainer.state.completed_epochs:
        raise ValueError("Checkpoint history must contain one record per completed epoch")
    saved_config = _config_dict(config)
    configured_ema = ema_config_from_dict(saved_config.get("training", {}).get("ema", {}))
    if (configured_ema["enabled"] != (trainer.ema is not None)
            or (trainer.ema is not None and configured_ema["decay"] != trainer.ema.decay)):
        raise ValueError("Checkpoint configuration must match the trainer's EMA settings")
    runtime_config = saved_config.setdefault("runtime", {})
    actual_fp8 = trainer.runtime.fp8
    if "fp8" in runtime_config:
        configured_fp8 = runtime_config["fp8"]
        if not isinstance(configured_fp8, Mapping):
            raise ValueError("Checkpoint FP8 configuration must be a table")
        configured_fp8 = fp8_config_from_dict(_merge(asdict(actual_fp8), configured_fp8))
        if (configured_fp8.enabled != actual_fp8.enabled
                or (actual_fp8.enabled and configured_fp8 != actual_fp8)):
            raise ValueError("Checkpoint configuration must match the trainer's FP8 settings")
    # Explicit recipe metadata may omit runtime settings. Always retain the
    # actual preparation policy so reconstruction uses the saved computation.
    runtime_config["fp8"] = asdict(actual_fp8)
    batch_size = loader_batch_size(train_loader)
    if batch_size is None:
        batch_size = _config_batch_size(saved_config)
    elif "batch_size" in saved_config.get("training", {}):
        saved_config["training"] = {**saved_config["training"], "batch_size": batch_size}
    payload = {
        "format_version": 5,
        "kind": kind,
        "epoch_progress": trainer.epoch_progress.state_dict() if kind == "step" else None,
        "train_loader": {"type": type_name(train_loader), "state": train_loader.state_dict()} if kind == "step" else None,
        "config": saved_config,
        "experiment_identity": snapshot_identity(saved_config, trainer.experiment_identity),
        "diagnostics": trainer.runtime.diagnostic_state(),
        "schedules": trainer.schedules.state_dict() if trainer.schedules is not None else None,
        "batch_size": batch_size,
        "components": _component_types(trainer),
        "trainer_settings": _trainer_settings(trainer),
        "optimizer_topology": _optimizer_topology(trainer),
        "strategies": snapshot_strategies(trainer),
        "environment": environment(),
        "device": str(trainer.device),
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        "state": asdict(trainer.state),
        "history": history,
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": ([torch.cuda.get_rng_state(trainer.device)] if distributed.world_size() > 1 else
                         torch.cuda.get_rng_state_all()) if trainer.device.type == "cuda" else []},
        "generators": {name: generator.get_state() for name, generator in generators.items()},
    }
    if trainer.schedules is not None:
        trainer.schedules.validate_state_dict(payload["schedules"], completed_update=trainer.state.optimizer_steps)
    if trainer.ema is not None:
        payload["ema"] = trainer.ema.state_dict()
        trainer.ema.validate_state_dict(payload["ema"], completed_updates=trainer.state.optimizer_steps)
    return payload


def save_checkpoint(path, trainer, config=None, *, generators, history, kind="epoch", train_loader=None) -> None:
    """Collect rank-local resume state on all ranks; atomically write on rank zero."""
    with distributed.collective_errors("checkpoint snapshot"):
        if config is None:
            config = trainer.config
        elif isinstance(config, (str, Path)):
            config = resolve_config(config)
        elif isinstance(config, Mapping):
            config = _merge(trainer.config, config)
        payload = _checkpoint_payload(trainer, config, generators=generators, history=history,
                                      kind=kind, train_loader=train_loader)
        if distributed.world_size() > 1:
            local = distributed.cpu_state({key: payload[key] for key in
                                           ("rng", "generators", "train_loader", "epoch_progress", "strategies", "device")})
            local["buffers"] = distributed.cpu_state(dict(trainer.model.named_buffers()))
    if distributed.world_size() > 1:
        if trainer.schedules is not None:
            distributed.require_equal(payload["schedules"], "coefficient schedule checkpoint state")
        states = [None] * distributed.world_size() if distributed.is_main_process() else None
        torch.distributed.gather_object(local, states, dst=0)
        payload["distributed"] = {"world_size": distributed.world_size(), "ranks": states}
    distributed.rank_zero_call(lambda: _write_checkpoint(Path(path), payload, kind))


def _write_checkpoint(path, payload, kind):
    # Replace only after a complete write; a failed write leaves the old file intact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        logger.debug("Saved %s checkpoint: %s", kind, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_checkpoint(path) -> dict:
    # Explicit weights_only also permits our primitive metadata and RNG tensors.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_checkpoint(payload)
    logger.debug("Read %s checkpoint: %s", payload["kind"], path)
    return payload


def read_checkpoint_progress(path) -> tuple[str, tuple[int, int]]:
    """Inspect progress without materializing model/optimizer tensor payloads.

    DeepOpt writes PyTorch's ZIP checkpoint format. Memory mapping avoids
    eagerly reading its storage records; meta placement discards their storage
    mappings while retaining metadata. Only the selected file is loaded on CPU.
    """
    payload = torch.load(path, map_location="meta", mmap=True, weights_only=True)
    _validate_checkpoint(payload)
    state = payload["state"]
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint training progress")
    progress = (state.get("optimizer_steps"), state.get("completed_epochs"))
    if any(type(value) is not int or value < 0 for value in progress):
        raise ValueError("Invalid checkpoint training progress")
    return payload["kind"], progress


def _validate_checkpoint(payload) -> None:
    if (not isinstance(payload, dict) or type(payload.get("format_version")) is not int
            or payload["format_version"] not in (4, 5)):
        raise ValueError("Unsupported checkpoint format; expected deepopt format_version 4 or 5")
    required = {"config", "environment", "device", "model", "optimizer", "scheduler",
                "state", "history", "rng", "generators", "components", "trainer_settings",
                "kind", "epoch_progress", "train_loader"}
    if payload["format_version"] == 5:
        required.add("schedules")
    if not required.issubset(payload):
        raise ValueError("Checkpoint is missing required training state")
    if payload["kind"] not in ("epoch", "step"):
        raise ValueError("Invalid checkpoint kind")
    if payload["kind"] == "step" and (not isinstance(payload["epoch_progress"], dict)
                                        or not isinstance(payload["train_loader"], dict)):
        raise ValueError("Step checkpoint is missing loader or epoch progress")


def _resume_config_for_comparison(config) -> dict:
    """Keep training-affecting metadata while accepting legacy passive defaults.

    Work on an owned copy: comparison must never rewrite a caller's config or
    a checkpoint. Unknown application and extension metadata remains strict.
    """
    settings = deepcopy(config)
    if "training" in settings:
        settings["training"] = {k: v for k, v in settings["training"].items()
                                if k not in ("epochs", "checkpoint_every_steps", "batch_size")
                                and not (k == "scheduler_interval" and v == "epoch")}
        if settings["training"].get("lr_schedule") is None:
            settings["training"].pop("lr_schedule", None)
        ema = ema_config_from_dict(settings["training"].get("ema", {}))
        if ema["enabled"]:
            settings["training"]["ema"] = ema
        else:
            # Existing checkpoints predate EMA; disabled is equivalent.
            settings["training"].pop("ema", None)
    if not settings.get("schedules"):
        settings.pop("schedules", None)
    if not settings.get("loader"):
        settings.pop("loader", None)
    # Logging is passive and can change without changing training state.
    settings.pop("logging", None)
    if "runtime" in settings:
        # Passive diagnostics may be enabled after a resume. Their objects
        # and counters are not part of optimization or checkpoint state.
        settings["runtime"] = {k: v for k, v in settings["runtime"].items()
                               if k not in ("checks", "execution_mode", "diagnostics", "mode_schedule", "non_blocking")}
        # Old checkpoints omit the compiler table. Disabled compilation is
        # equivalent; enabled settings remain strict for reproducibility.
        if not settings["runtime"].get("compile", {}).get("enabled", False):
            settings["runtime"].pop("compile", None)
        if not settings["runtime"].get("fp8", {}).get("enabled", False):
            settings["runtime"].pop("fp8", None)
        else:
            # JSON/TOML grids use lists; runtime policies use tuples. Compare
            # their validated meaning rather than their container spelling.
            settings["runtime"]["fp8"] = asdict(fp8_config_from_dict(settings["runtime"]["fp8"]))
        defaults = {
            "ddp": {"find_unused_parameters": False, "broadcast_buffers": True,
                    "gradient_as_bucket_view": True, "static_graph": False,
                    "validation_sync_buffers": True},
            "cuda_graph": {"device": None, "warmup": 3, "allow_unused_input": False,
                           "autograd": True, "capture_error_mode": "global"},
        }
        for field, values in defaults.items():
            if field in settings["runtime"]:
                settings["runtime"][field] = {key: value for key, value in settings["runtime"][field].items()
                                              if key not in values or value != values[key]}
            if not settings["runtime"].get(field):
                settings["runtime"].pop(field, None)
    if not settings.get("experiment_identity"):
        settings.pop("experiment_identity", None)
    return settings


def _checkpoint_for_rank(payload):
    """Select rank-local state without applying it to the trainer."""
    _validate_checkpoint(payload)
    saved_distributed = payload.get("distributed")
    saved_world = saved_distributed.get("world_size") if isinstance(saved_distributed, Mapping) else 1
    if saved_world != distributed.world_size():
        raise ValueError("Training resume requires the same distributed world size; use evaluate for topology changes")
    rank_buffers = None
    if saved_world > 1:
        ranks = saved_distributed.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != saved_world:
            raise ValueError("Invalid distributed checkpoint rank states")
        local = ranks[distributed.rank()]
        required = {"rng", "generators", "train_loader", "epoch_progress", "strategies", "device", "buffers"}
        if not isinstance(local, Mapping) or not required.issubset(local):
            raise ValueError("Distributed checkpoint is missing rank-local resume state")
        payload = {**payload, **{key: local[key] for key in required if key != "buffers"}}
        rank_buffers = local["buffers"]
    return payload, rank_buffers, saved_world


def _requested_resume_batch_size(payload, trainer, requested, train_loader, *,
                                 batch_size_mode, memory_budget_bytes):
    old_batch_size = _saved_batch_size(payload)
    requested_batch_size = loader_batch_size(train_loader)
    configured_batch_size = _config_batch_size(requested)
    if (requested_batch_size is not None and configured_batch_size is not None
            and requested_batch_size != configured_batch_size):
        raise ValueError("Requested configuration batch size differs from the training loader")
    if requested_batch_size is None:
        requested_batch_size = configured_batch_size
    if requested_batch_size is not None and (type(requested_batch_size) is not int or requested_batch_size < 1):
        raise ValueError("Requested batch size must be a positive integer")
    if (requested_batch_size is None and batch_size_mode == "auto"
            and (trainer.device.type == "cuda" or memory_budget_bytes is not None)):
        raise ValueError("Automatic batch sizing requires a loader exposing its batch_size or an explicit "
                         "training.batch_size setting; use manual mode for custom batching")
    if (train_loader is None and batch_size_mode == "auto"
            and (trainer.device.type == "cuda" or memory_budget_bytes is not None)):
        raise ValueError("Automatic resume sizing requires train_loader so the selected size can be applied; "
                         "use estimate_training_memory separately or select manual mode")
    return old_batch_size, requested_batch_size


def _validate_resume_state(payload, trainer, requested, *, generators, train_loader,
                           old_batch_size, requested_batch_size, saved_world):
    """Check compatibility before any state-loading hooks can mutate objects."""
    if payload["trainer_settings"].get("fp8", {}).get("enabled", False):
        if payload["trainer_settings"].get("fp8_recipe") != fp8_recipe(trainer.runtime.fp8.backend):
            raise ValueError("FP8 numerical recipe or Triton version differs from checkpoint; "
                             "resume with the original library/environment, or load model weights for a new run")
    if _resume_config_for_comparison(payload["config"]) != _resume_config_for_comparison(requested):
        raise ValueError("Resume configuration differs from checkpoint "
                         "(only epochs, batch size, checkpoint cadence, and diagnostic checks may change)")
    saved_settings = dict(payload["trainer_settings"])
    if "fp8" in saved_settings:
        # Older snapshots omit newly added default policy fields. Preserve
        # their meaning while still rejecting an actual execution-policy change.
        saved_settings["fp8"] = asdict(fp8_config_from_dict(saved_settings["fp8"]))
    if payload["components"] != _component_types(trainer) or saved_settings != _trainer_settings(trainer):
        raise ValueError("Resume requires the same component types and trainer settings")
    saved_compile = compile_config_from_dict(payload["config"].get("runtime", {}).get("compile", {}))
    actual_compile = compile_config_from_dict(trainer.config.get("runtime", {}).get("compile", {}))
    if (saved_compile.enabled or actual_compile.enabled) and saved_compile != actual_compile:
        raise ValueError("Resume requires the same model compilation settings")
    if "optimizer_topology" not in payload:
        raise ValueError("Checkpoint has no named optimizer topology; resume with the original code or start a new run")
    if payload["optimizer_topology"] != _optimizer_topology(trainer):
        raise ValueError("Resume requires the same optimizer parameter names, order, and child routing")
    _validate_model_state(payload["model"], trainer.model)
    validate_strategies(payload.get("strategies"), trainer)
    if payload["device"] != str(trainer.device):
        raise ValueError("Resume requires the original device; use evaluate for device changes")
    if payload["environment"]["torch"] != str(torch.__version__):
        raise ValueError("Resume requires the same PyTorch version as the checkpoint")
    if set(payload["generators"]) != set(generators):
        raise ValueError("Checkpoint data generators do not match this experiment")
    if (payload["scheduler"] is None) != (trainer.scheduler is None):
        raise ValueError("Checkpoint scheduler does not match this trainer")
    if trainer.device.type == "cuda" and len(payload["rng"]["cuda"]) != (1 if saved_world > 1 else torch.cuda.device_count()):
        raise ValueError("Resume requires the same number of visible CUDA devices")
    state = TrainingState(**payload["state"])
    _validate_ema_state(payload, trainer, completed_updates=state.optimizer_steps)
    saved_schedules = payload.get("schedules")
    if (saved_schedules is None) != (trainer.schedules is None):
        raise ValueError("Resume requires the same coefficient schedules; old checkpoints cannot acquire new schedules")
    if trainer.schedules is not None:
        trainer.schedules.validate_state_dict(saved_schedules, completed_update=state.optimizer_steps)
        if saved_schedules["epoch"] > state.completed_epochs + (payload["kind"] == "step"):
            raise ValueError("Schedule epoch is ahead of saved training progress")
    history = payload["history"]
    if len(history) != state.completed_epochs:
        raise ValueError("Checkpoint history does not match its completed epoch count")
    if payload["kind"] == "step":
        require_step_loader(train_loader)
        if payload["train_loader"]["type"] != type_name(train_loader):
            raise ValueError("Step resume requires the original loader type")
        if payload["environment"]["torchmetrics"] != version("torchmetrics"):
            raise ValueError("Step resume requires the same TorchMetrics version")
        if (old_batch_size is not None and requested_batch_size is not None
                and old_batch_size != requested_batch_size
                and getattr(train_loader, "supports_batch_size_change", False) is not True):
            raise ValueError("Changing batch size during step resume requires a loader declaring "
                             "supports_batch_size_change=True and preserving its consumed-data position")
    return state


def _restore_loader_progress(payload, trainer, state, train_loader, requested_batch_size):
    """Restore metric/loader state only after the trainer has been invalidated."""
    progress = None
    if payload["kind"] == "step":
        progress = EpochProgress.restore(payload["epoch_progress"], trainer.new_metrics(), trainer.device)
        if trainer.schedules is not None:
            from deepopt.scheduling.base import finite
            start_values = progress.scheduled_values_start
            if not isinstance(start_values, dict) or start_values.keys() != trainer.schedules.values.keys():
                raise ValueError("Invalid saved epoch-start coefficient values")
            for name, value in start_values.items():
                finite(value, name)
        if (type(progress.batches) is not int or progress.batches <= 0
                or (progress.samples is not None and
                    (type(progress.samples) is not int or progress.samples <= 0))
                or type(progress.updates_at_start) is not int
                or not 0 <= progress.updates_at_start < state.optimizer_steps
                or (progress.learning_rates is not None and
                    (not isinstance(progress.learning_rates, list)
                     or len(progress.learning_rates) != len(trainer.optimizer.param_groups)))
                or (trainer.scheduler_interval == "step" and progress.learning_rates is None)):
            raise ValueError("Invalid within-epoch checkpoint progress")
        train_loader.load_state_dict(payload["train_loader"]["state"])
        if (requested_batch_size is not None
                and loader_batch_size(train_loader) != requested_batch_size):
            raise ValueError("The loader restored a different batch size instead of retaining the requested size")
    elif train_loader is not None and callable(getattr(train_loader, "reset_epoch", None)):
        train_loader.reset_epoch()
    return progress


def _restore_training_state(payload, trainer, state, *, rank_buffers, generators, saved_world):
    """Load components, then restore RNG after all component loading hooks."""
    trainer.model.load_state_dict(payload["model"])
    if rank_buffers is not None:
        with torch.no_grad():
            for name, buffer in trainer.model.named_buffers():
                buffer.copy_(rank_buffers[name].to(buffer.device))
    trainer.optimizer.load_state_dict(payload["optimizer"])
    # PyTorch's scheduler-order warning uses a flag not stored by optimizers.
    # A step checkpoint may resume immediately before scheduler.step(), with
    # no new optimizer update in this process. Preserve that prior-step fact.
    trainer.optimizer._opt_called = state.optimizer_steps > 0
    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(payload["scheduler"])
    restore_strategies(payload.get("strategies"), trainer)
    if trainer.schedules is not None:
        trainer.schedules.load_state_dict(payload["schedules"], completed_update=state.optimizer_steps)
    if trainer.ema is not None:
        trainer.ema.load_state_dict(payload["ema"], completed_updates=state.optimizer_steps)
    trainer.state = state
    for name, generator in generators.items():
        generator.set_state(payload["generators"][name])
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch"])
    if trainer.device.type == "cuda":
        if saved_world > 1:
            torch.cuda.set_rng_state(payload["rng"]["cuda"][0], trainer.device)
        else:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])


def restore_checkpoint(payload, trainer, config=None, *, generators, train_loader=None,
                       validation_loader=None, batch_size_mode="auto", sample_batch_fn=None,
                       memory_budget_bytes=None) -> list[dict]:
    """Restore training state, then plan physical batching before continuing.

    Auto uses caller-supplied representative batches and fake tensor execution
    on CUDA (or when an explicit tensor-memory budget is supplied). Manual
    keeps the requested size. Neither policy changes LR, accumulation, or the
    number of previously consumed samples. No real training probes are run.
    """
    payload, rank_buffers, saved_world = _checkpoint_for_rank(payload)
    if batch_size_mode not in ("auto", "manual"):
        raise ValueError("batch_size_mode must be auto or manual")
    saved = payload["config"]
    if config is None:
        requested = saved
    elif isinstance(config, (str, Path)):
        requested = dict(resolve_config(config))
    elif isinstance(config, Mapping):
        requested = _merge(saved, config)
    else:
        requested = _config_dict(config)
    saved_identity = payload.get("experiment_identity", {}).get("required", {})
    if saved_identity != validate_identity(trainer.experiment_identity):
        raise ValueError("Resume requires the same experiment_identity (recipe/data/code versions)")
    old_batch_size, requested_batch_size = _requested_resume_batch_size(
        payload, trainer, requested, train_loader,
        batch_size_mode=batch_size_mode, memory_budget_bytes=memory_budget_bytes,
    )
    state = _validate_resume_state(
        payload, trainer, requested, generators=generators, train_loader=train_loader,
        old_batch_size=old_batch_size, requested_batch_size=requested_batch_size, saved_world=saved_world,
    )
    # Component load_state_dict hooks can fail after mutating their component.
    # Never permit another automatic epoch on a partially restored trainer.
    # A later complete restore is the only operation that re-enables training.
    trainer.at_epoch_boundary = False
    trainer.at_step_boundary = False
    trainer._resume_pending = False
    trainer.epoch_progress = None
    trainer.resume_adjustment = None
    trainer.batch_size_adjustment = None
    trainer.runtime.invalidate()
    progress = _restore_loader_progress(payload, trainer, state, train_loader, requested_batch_size)
    _restore_training_state(payload, trainer, state, rank_buffers=rank_buffers,
                            generators=generators, saved_world=saved_world)
    if requested_batch_size is not None:
        adjustment = plan_batch_size(
            trainer, requested_batch_size=requested_batch_size, mode=batch_size_mode,
            sample_batch_fn=sample_batch_fn, memory_budget_bytes=memory_budget_bytes,
        )
        selected_size = adjustment["new_batch_size"]
        adjustment["old_batch_size"] = old_batch_size
        apply_batch_size_plan(trainer, train_loader, validation_loader, adjustment,
                              step_resume=payload["kind"] == "step")
        adjustment.update(old_batch_size=old_batch_size,
                          validation_batch_size=loader_batch_size(validation_loader),
                          completed_epochs=state.completed_epochs, optimizer_steps=state.optimizer_steps)
        trainer.resume_adjustment = adjustment
        if old_batch_size != selected_size:
            logger.warning("Resume batch size changed from %s to %s (%s mode); "
                           "consumed-data position is retained, LR and accumulation are unchanged",
                           old_batch_size, selected_size, batch_size_mode)
    trainer.runtime.reset_after_restore()
    trainer.runtime.restore_update_count(state.optimizer_steps)
    trainer.epoch_progress = progress
    trainer._resume_pending = progress is not None
    trainer.at_epoch_boundary = progress is None
    trainer.at_step_boundary = False
    return list(payload["history"])


def restore_schedules_for_evaluation(payload, trainer):
    """Restore last applied values without preparing or completing an update."""
    saved = payload.get("schedules")
    if (saved is None) != (trainer.schedules is None):
        raise ValueError("Evaluation requires the checkpoint's coefficient schedules")
    if trainer.schedules is not None:
        trainer.schedules.load_state_dict(saved, completed_update=payload["state"]["optimizer_steps"])


def _validate_ema_state(payload, trainer, *, completed_updates):
    saved = payload.get("ema")
    configured = ema_config_from_dict(payload["config"].get("training", {}).get("ema", {}))
    if (configured["enabled"] != (trainer.ema is not None)
            or (trainer.ema is not None and configured["decay"] != trainer.ema.decay)):
        raise ValueError("EMA settings differ from checkpoint")
    if trainer.ema is None:
        if saved is not None:
            raise ValueError("Checkpoint EMA state requires EMA to be enabled")
    else:
        trainer.ema.validate_state_dict(saved, completed_updates=completed_updates)


def restore_ema_for_evaluation(payload, trainer):
    """Restore validation averages while leaving the checkpoint's live model intact."""
    completed_updates = payload["state"]["optimizer_steps"]
    _validate_ema_state(payload, trainer, completed_updates=completed_updates)
    if trainer.ema is not None:
        trainer.ema.load_state_dict(payload["ema"], completed_updates=completed_updates)
