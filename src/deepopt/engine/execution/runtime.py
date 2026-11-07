"""Optional execution helpers shared by automatic and user-owned loops."""

import torch
from contextlib import ContextDecorator, contextmanager
from contextvars import ContextVar
from deepopt.distributed import _process as distributed_ops
from deepopt.distributed.ddp import DDPCore
from deepopt.engine.core import TrainingCore
from collections.abc import Callable
from math import isfinite
from numbers import Real
from typing import Any, TypeVar
from weakref import proxy, ref
from copy import deepcopy
from dataclasses import asdict

from deepopt._config_resolution import _UNSET, ResolvedConfig
from deepopt.engine.execution.configuration import execution_config

from deepopt.observers import CheckObservers, FiniteCheck, TensorStatistics
from deepopt.engine.execution.policy import DiagnosticPolicy, ExecutionMode, ModeController
from deepopt.engine.execution.batch import move_to_device
from deepopt.engine.execution.optimizer_step import OptimizerStepper
from deepopt.engine.execution.precision import PrecisionPolicy
from deepopt.engine.strategies.gradients import ClipGradNorm, GradientStrategy


Result = TypeVar("Result")
_autocast_entries = ContextVar("deepopt_runtime_autocast_entries", default=())


class _RuntimeAutocast(ContextDecorator):
    """Reusable entries read current scheduling state without owning the runtime."""

    def __init__(self, runtime):
        self._runtime = ref(runtime)

    def __enter__(self):
        runtime = self._runtime()
        if runtime is None:
            raise RuntimeError("The autocast context's runtime no longer exists")
        context = runtime._autocast()
        result = context.__enter__()
        _autocast_entries.set((*_autocast_entries.get(), context))
        return result

    def __exit__(self, *args):
        entries = _autocast_entries.get()
        _autocast_entries.set(entries[:-1])
        return entries[-1].__exit__(*args)


class TrainingRuntime:
    """Provide execution operations without epochs, metrics, or persistence.

    Move the model before constructing its optimizer. backward() leaves the
    loss reduction to the caller. step(gradient_divisor=N) divides accumulated
    gradients by N, validates/clips them, updates the optimizer, and clears its
    gradients. The optimizer may own only part of the model.
    For a scalar mean loss use step() (N=1). For custom update algorithms, call
    the optimizer directly and use only the helpers needed by that algorithm.
    fp8 enables automatic preparation in the chosen core. Use runtime.model
    afterward; autocast() combines its FP8 policy with the configured AMP mode.
    """

    def __init__(self, model, optimizer=None, *, device=_UNSET, precision=_UNSET, grad_clip_norm=_UNSET,
                 gradient_strategy: GradientStrategy | None = None, checks=_UNSET, ddp_options=_UNSET,
                 execution_mode=_UNSET, diagnostics=_UNSET, mode_schedule=_UNSET, non_blocking=_UNSET,
                 fp8=_UNSET, config=None):
        with distributed_ops.collective_errors("runtime configuration"):
            self._config_snapshot = execution_config(
                config, device=device, precision=precision, checks=checks, grad_clip_norm=grad_clip_norm,
                execution_mode=execution_mode, diagnostics=diagnostics, mode_schedule=mode_schedule,
                non_blocking=non_blocking, fp8=fp8, model=model, ddp_options=ddp_options)
            runtime = self._config_snapshot.get("runtime", {})
            training = self._config_snapshot.get("training", {})
            device = runtime.get("device", "cpu")
            precision = runtime.get("precision", "fp32")
            grad_clip_norm = training.get("grad_clip_norm")
            checks = runtime.get("checks", True) if checks is _UNSET else checks
            execution_mode = runtime.get("execution_mode")
            diagnostics = runtime.get("diagnostics")
            configured_modes = runtime.get("mode_schedule")
            mode_schedule = ({int(step): mode for step, mode in configured_modes.items()}
                             if configured_modes is not None else None)
            non_blocking = runtime.get("non_blocking", False)
            if precision not in ("fp32", "bf16"):
                raise ValueError("precision must be fp32 or bf16; fp16 is unsupported")
        with distributed_ops.collective_errors("runtime device"):
            self.distributed_context = distributed_ops.initialize_distributed(device)
            self.device = self.distributed_context.device
            if self.device.type == "cuda" and self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        self.model, self.optimizer = model, optimizer
        with distributed_ops.collective_errors("diagnostic configuration"):
            self.checks = CheckObservers(checks)
            self._checks_spec = checks if type(checks) is bool else self.checks._observers
            self.mode_controller = ModeController(execution_mode, diagnostics, mode_schedule)
            if type(non_blocking) is not bool:
                raise ValueError("non_blocking must be a boolean")
        self._in_compute = self._forward_pending = self._pending_gradients = False
        self._diagnostic_depth = 0
        self.non_blocking = non_blocking
        self.precision = precision
        with distributed_ops.collective_errors("runtime preconditions"):
            if grad_clip_norm is not None and gradient_strategy is not None:
                raise ValueError("Choose grad_clip_norm or gradient_strategy, not both")
            self.grad_clip_norm = grad_clip_norm
            if grad_clip_norm is not None:
                if not 0 < grad_clip_norm < float("inf"):
                    raise ValueError("grad_clip_norm must be positive and finite")
                gradient_strategy = ClipGradNorm(grad_clip_norm)
            if gradient_strategy is not None and not isinstance(gradient_strategy, GradientStrategy):
                raise TypeError("gradient_strategy must inherit GradientStrategy")
            self._gradient_strategy = gradient_strategy
            self._precision = PrecisionPolicy(self.device, precision)
            if any(p.device != self.device for p in model.parameters()):
                raise ValueError("Move the model to the training device before constructing its optimizer")
        if distributed_ops.world_size() > 1:
            self.core = DDPCore(model, self.device, ddp_options, fp8=fp8, config=self._config_snapshot)
        else:
            if ddp_options is not _UNSET and ddp_options:
                raise ValueError("ddp_options requires an initialized multi-rank process group")
            self.core = TrainingCore(model, self.device, fp8=fp8, config=self._config_snapshot)
        self.model = self.core.model
        compilation = getattr(self.model, "_deepopt_compile_config", None)
        if compilation is not None:
            self._config_snapshot.setdefault("runtime", {})["compile"] = deepcopy(compilation)
        self.fp8 = self.core.fp8
        self.transformation_report = self.core.transformation_report
        if self.core.distributed:
            self._config_snapshot.setdefault("runtime", {})["ddp"] = deepcopy(self.core.options)
        self._config_snapshot.setdefault("runtime", {})["fp8"] = asdict(self.fp8)
        self._config_snapshot.setdefault("runtime", {}).update({
            "device": str(device), "precision": precision, "execution_mode": execution_mode,
            "diagnostics": deepcopy(diagnostics), "mode_schedule": deepcopy(configured_modes),
            "non_blocking": non_blocking,
        })
        if type(checks) is bool:
            self._config_snapshot["runtime"]["checks"] = checks
        self._config_snapshot.setdefault("training", {})["grad_clip_norm"] = grad_clip_norm
        self._precision.configure(fp8=self.fp8 if self.fp8.enabled else None,
                                  config=ResolvedConfig(deepcopy(self._config_snapshot)))
        with self.core.check_errors("optimizer setup"):
            self._stepper = (OptimizerStepper(self.model, optimizer, self._precision, gradient_strategy,
                                             checks=self.checks)
                             if optimizer is not None else None)
        self._failed = False
        if self._stepper is not None:
            # The stepper borrows its owner for diagnostics; owning it back
            # would keep native DDP resources alive until cyclic collection.
            self._stepper.checks = proxy(self)
        self._configure_checks()

    @property
    def config(self):
        return deepcopy(self._config_snapshot)

    @property
    def execution_mode(self):
        policy = self.mode_controller.policy
        return policy.mode.value if policy is not None else None

    @property
    def diagnostics(self):
        return self.mode_controller.policy

    def _configure_checks(self):
        checks = self._checks_spec
        # An explicit mode replaces the legacy boolean switch. Custom observer
        # collections still replace the built-ins and are gated by the policy.
        if type(checks) is bool and self.diagnostics is not None:
            checks = [FiniteCheck(kinds={"loss", "gradients", "report", "metrics", "weights", "activation"})]
            if self.diagnostics.mode == ExecutionMode.ANALYZE:
                checks.append(TensorStatistics())
        self.checks = CheckObservers(checks, gate=self.mode_controller.permits)
        self.core.configure_checks(self.checks, self.mode_controller.configuration())

    def _prepare_mode(self):
        if not self._pending_gradients and not self._forward_pending and self.mode_controller.prepare_update():
            self._configure_checks()

    @contextmanager
    def diagnostic_scope(self, name, *, phase="forward"):
        """Apply the current passive-check policy without changing AMP or FP8 formats."""
        self._diagnostic_depth += 1
        try:
            with self.mode_controller.scope(name, phase=phase, legacy_checks=self._checks_spec is not False):
                yield
        finally:
            self._diagnostic_depth -= 1

    def set_execution_mode(self, mode):
        """Change passive diagnostics only at a completed update/epoch boundary."""
        self._require_healthy()
        with self.core.check_errors("execution mode transition"):
            if (self._in_compute or self._diagnostic_depth or self._forward_pending or self._pending_gradients
                    or any(parameter.grad is not None for parameter in self.model.parameters())):
                raise RuntimeError("Change execution mode only at an optimizer-update or epoch boundary")
            policy = DiagnosticPolicy.for_mode(mode)
        self.core.require_equal(policy.as_dict(), "execution mode transition")
        self.mode_controller.base_policy = self.mode_controller.policy = policy
        self.mode_controller.schedule = {}
        self._configure_checks()

    def diagnostic_state(self):
        return {**self.mode_controller.configuration(), "active_mode": self.execution_mode,
                "completed_updates": self.mode_controller.completed_updates}

    def restore_update_count(self, count):
        self.mode_controller.completed_updates = count
        if self.mode_controller.prepare_update():
            self._configure_checks()

    @property
    def distributed(self):
        return self.core.distributed

    def require_complete_optimizer(self) -> None:
        """Require full trainable-parameter coverage for automatic training."""
        if self._stepper is None:
            raise ValueError("Automatic optimization requires an optimizer")
        self._stepper.require_complete_optimizer()

    def _require_healthy(self) -> None:
        self._require_eager()
        if self._failed:
            raise RuntimeError("A previous backward or optimizer update failed; restore a checkpoint "
                               "or reconstruct the runtime before continuing")

    def _require_eager(self) -> None:
        # Python counters, diagnostics and rank coordination must execute on
        # every update. Capture the model, leaving runtime operations outside.
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TrainingRuntime operations cannot be CUDA-captured; use "
                               "cuda_graph_core.capture_model before constructing the runtime")

    def _mark_failed(self) -> None:
        self._failed = True
        self._forward_pending = self._pending_gradients = False
        parameters = self._stepper.parameters() if self._stepper is not None else self.model.parameters()
        for parameter in parameters:
            parameter.grad = None

    def invalidate(self) -> None:
        """Discard owned gradients and block updates during unsafe state changes.

        Checkpoint restoration uses this before mutating model and optimizer
        state, then calls reset_after_restore() only after restoration succeeds.
        """
        self._mark_failed()

    def reset_after_restore(self) -> None:
        """Enable updates after the caller has restored all training state.

        This does not undo parameter, optimizer, or strategy mutations from a
        failed update. Only call it after restoring those states from a trusted
        checkpoint; otherwise reconstruct the model, optimizer, and runtime.
        """
        self._mark_failed()
        self._failed = False

    def prepare_batch(self, batch):
        return move_to_device(batch, self.device, non_blocking=self.non_blocking)

    @property
    def gradient_strategy(self):
        return self._gradient_strategy

    def autocast(self):
        return _RuntimeAutocast(self)

    @contextmanager
    def _autocast(self):
        self._require_eager()
        if self.model.training and torch.is_grad_enabled():
            self._prepare_mode()
        with self.diagnostic_scope("deepopt.compute"), self._precision.autocast() as context:
            yield context

    def compute(self, batch: Any, step: Callable[[Any], Result]) -> Result:
        """Transfer a batch and call any step under the precision context.

        No BatchOutput, Task, metric, or optimizer contract is imposed here.
        This does not change train/eval mode or disable autograd.
        """
        self._require_eager()
        self._in_compute = True
        try:
            batch = self.prepare_batch(batch)
            with self.autocast():
                result = self.core.compute(batch, step)
            if self.model.training and torch.is_grad_enabled():
                self._forward_pending = True
            return result
        finally:
            self._in_compute = False

    def observe(self, kind, tensors, *, name="") -> None:
        """Run configured diagnostics on borrowed tensors without retaining them.

        Observers must not modify training state. An observer failure discards
        owned gradients and requires restoration before another update.
        """
        self._require_healthy()
        try:
            if not isinstance(kind, str) or not kind or not isinstance(name, str):
                raise ValueError("Observations require a nonempty kind and a string name")
            self._observe(kind, tensors, name=name)
        except BaseException:
            self._mark_failed()
            raise

    def backward(self, loss, **kwargs):
        """Forward torch.Tensor.backward options, such as retain_graph."""
        self._require_healthy()
        try:
            self._prepare_mode()
            with self.diagnostic_scope("deepopt.backward", phase="backward"):
                self._observe("loss", loss, name="backward loss")
                self.core.backward(loss, **kwargs)
            self._forward_pending = bool(kwargs.get("retain_graph", False))
            self._pending_gradients = True
        except BaseException:
            self._mark_failed()
            raise

    def zero_grad(self):
        self._require_eager()
        if self.optimizer is None:
            raise RuntimeError("zero_grad requires an optimizer")
        self.optimizer.zero_grad(set_to_none=True)
        self._forward_pending = self._pending_gradients = False

    def step(self, samples: int = 1, *, gradient_divisor: Real | None = None,
             normalize: bool | None = None, distributed_reduction="legacy"):
        """Update owned parameters using a caller-defined positive divisor.

        samples is the legacy integer-only alias. Use gradient_divisor for
        fractional loss weights. A failed update clears owned gradients and
        blocks further updates until restored, since optimizer updates can be
        partial and gradient transformations need not be reversible.
        """
        if self._stepper is None:
            raise RuntimeError("step requires an optimizer")
        self._require_healthy()
        self._prepare_mode()
        with self.core.check_errors("gradient divisor"):
            if type(samples) is not int or samples < 1:
                raise ValueError("samples must be a positive integer")
            if gradient_divisor is not None and samples != 1:
                raise ValueError("Choose samples or gradient_divisor, not both")
            divisor = samples if gradient_divisor is None else gradient_divisor
            try:
                valid_divisor = (isinstance(divisor, Real) and not isinstance(divisor, bool)
                                 and isfinite(divisor) and divisor > 0)
            except (OverflowError, TypeError):
                valid_divisor = False
            if not valid_divisor:
                raise ValueError("gradient_divisor must be a positive finite real number")
            if normalize is None:
                normalize = gradient_divisor is not None or samples != 1
            if type(normalize) is not bool:
                raise ValueError("normalize must be a boolean")
            if distributed_reduction not in ("legacy", "rank_mean", "global_mean", "global_sum"):
                raise ValueError("Invalid distributed reduction policy")
            if distributed_reduction in ("rank_mean", "global_sum") and normalize:
                raise ValueError("Rank-mean and global-sum objectives must not supply normalization weights")
            if distributed_reduction == "global_mean" and not normalize:
                raise ValueError("global_mean requires normalization weights")
        try:
            self.core.validate_window(normalize, distributed_reduction)
            divisor = self.core.gradient_divisor(float(divisor), normalize=normalize, reduction=distributed_reduction)
            with self.diagnostic_scope("deepopt.step", phase="backward"):
                self._stepper.step(divisor)
            self._forward_pending = self._pending_gradients = False
            self.mode_controller.completed_updates += 1
        except BaseException:
            self._mark_failed()
            raise

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _observe(self, kind, tensors, *, name=""):
        self.core.observe(self.checks, kind, tensors, name=name)

    def interested(self, kind):
        return self.core.interested(self.checks, kind)

    def check_errors(self, operation):
        return self.core.check_errors(operation)

    def sync_buffers(self):
        self.core.sync_buffers()
