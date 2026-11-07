"""Engine's single-process training core; no distributed collectives."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict

import torch

from deepopt._config_resolution import _UNSET
from deepopt.engine.execution.configuration import execution_config, prepare_execution_model
from deepopt.runtime import resolve_device


class TrainingCore:
    """Local execution policy used by TrainingRuntime in a single process.

    Prepares opt-in FP8 operations and owns forward/backward execution, local
    observer dispatch and local gradient normalization. DDPCore implements the
    same interface independently.
    Runtime owns shared precision, optimizer execution and failure-state handling;
    Trainer owns epochs, accumulation scheduling, metrics and callbacks.
    """

    distributed = False

    def __init__(self, model, device=_UNSET, *, fp8=_UNSET, config=None):
        self._config_snapshot = execution_config(config, device=device, fp8=fp8, model=model)
        selected = self._config_snapshot.get("runtime", {}).get("device", "cpu")
        self.device = resolve_device(selected) if selected == "auto" else torch.device(selected)
        self.model, self.fp8, self.transformation_report = prepare_execution_model(
            model, self._config_snapshot, fp8)
        self._config_snapshot.setdefault("runtime", {})["fp8"] = asdict(self.fp8)
        from deepopt.opt._autocast import _FP8Scope

        self._fp8_scope = _FP8Scope(self.device.type, self.fp8 if self.fp8.enabled else None)

    @property
    def config(self):
        return deepcopy(self._config_snapshot)

    def compute(self, batch, step):
        with self._fp8_scope.enter():
            return step(batch)

    def backward(self, loss, **kwargs):
        loss.backward(**kwargs)

    def gradient_divisor(self, local_divisor, *, normalize=True, reduction="legacy"):
        return local_divisor

    def validate_window(self, normalize, reduction):
        pass

    def accumulation_context(self, *, synchronize):
        return nullcontext()

    def configure_checks(self, checks, configuration):
        pass

    def observe(self, checks, kind, tensors, *, name=""):
        checks.observe(kind, tensors, name=name)

    def interested(self, checks, kind):
        return checks.interested(kind)

    def sync_buffers(self):
        pass

    def check_errors(self, operation):
        return nullcontext()

    def require_equal(self, value, description):
        pass
