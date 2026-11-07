"""DDP execution core, owned by the distributed cores package."""

import torch
from contextlib import nullcontext
from dataclasses import asdict
from copy import deepcopy
from math import isfinite
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from . import _process as distributed
from deepopt._config_resolution import _UNSET
from deepopt.opt.policy import resolve_fp8_config


class _StepModule(nn.Module):
    """Place functional steps and TrainingModule hooks inside DDP's forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, step):
        return step(batch)


class DDPCore:
    """Prepare optional FP8 and own DDP while preserving parameter/state names.

    TrainingRuntime delegates compute/backward and gradient normalization here.
    The complete step runs inside DDP, including custom functions that access
    submodules directly and TrainingModule hooks. Manual DDP loops must perform
    their forward through runtime.compute; passing an unrelated loss to backward
    would bypass DDP and is rejected.
    """

    distributed = True

    def __init__(self, model, device=_UNSET, options=_UNSET, *, fp8=_UNSET, config=None):
        from deepopt.engine.execution.configuration import execution_config, prepare_execution_model

        with distributed.collective_errors("core configuration"):
            self._config_snapshot = execution_config(config, device=device, fp8=fp8, model=model,
                                                      ddp_options=options)
            runtime = self._config_snapshot.get("runtime", {})
            selected_device = runtime.get("device", "cpu")
            if options is not _UNSET:
                # Explicit options replace the inherited table. execution_config
                # already validated them; preserve the original native kwargs
                # without resolving the whole snapshot a second time.
                runtime["ddp"] = deepcopy(options or {})
            options = runtime.get("ddp", {})
        self.model = model
        self.device = (distributed.initialize_distributed("auto").device
                       if selected_device == "auto" else torch.device(selected_device))
        self.forward_pending = False
        self.wrapper = None
        self.options = {"find_unused_parameters": False, "broadcast_buffers": True,
                        "gradient_as_bucket_view": True, **(options or {})}
        self.validation_sync_buffers = self.options.pop("validation_sync_buffers", True)
        self._observer_interests = set()
        if distributed.world_size() < 2:
            raise ValueError("DDPCore requires an initialized multi-rank process group")
        if isinstance(model, DistributedDataParallel):
            raise ValueError("Pass the unwrapped model; DDPCore wraps the complete step in DDP")
        with distributed.collective_errors("FP8 configuration"):
            self.fp8 = resolve_fp8_config(model, fp8, config=self._config_snapshot)
        distributed.require_equal(asdict(self.fp8), "FP8 configuration")
        distributed.require_equal({**self.options, "validation_sync_buffers": self.validation_sync_buffers},
                                  "DDP options")
        with distributed.collective_errors("FP8 model preparation"):
            self.model, self.fp8, self.transformation_report = prepare_execution_model(
                model, self._config_snapshot, self.fp8)
        model = self.model
        if any(parameter.requires_grad for parameter in model.parameters()):
            self.wrapper = DistributedDataParallel(
                _StepModule(model), device_ids=[self.device.index] if self.device.type == "cuda" else None,
                **self.options)
        if self.fp8.enabled:
            from deepopt.opt.do_nn._cached_weights import CachedWeightModule

            # DDP's initial parameter broadcast can replace rank-local values
            # after a caller has already warmed an adapter's weight cache.
            for module in model.modules():
                if isinstance(module, CachedWeightModule):
                    module.clear_fp8_cache()
        if not self.validation_sync_buffers:
            self.options["validation_sync_buffers"] = False
        self._config_snapshot.setdefault("runtime", {})["fp8"] = asdict(self.fp8)
        self._config_snapshot["runtime"]["ddp"] = deepcopy(self.options)
        from deepopt.opt._autocast import _FP8Scope

        self._fp8_scope = _FP8Scope(self.device.type, self.fp8 if self.fp8.enabled else None)

    @property
    def config(self):
        return deepcopy(self._config_snapshot)

    def compute(self, batch, step):
        with self._fp8_scope.enter():
            if self.wrapper is not None and self.model.training and torch.is_grad_enabled():
                result = self.wrapper(batch, step)
                self.forward_pending = True
                return result
            # Validation can have unequal or empty shards: no per-forward collective.
            return step(batch)

    def backward(self, loss, **kwargs):
        with distributed.collective_errors("backward preconditions"):
            if not self.forward_pending:
                raise RuntimeError("DDP backward requires the forward to run through runtime.compute(batch, step)")
            if kwargs.get("create_graph", False):
                raise ValueError("DDP does not support higher-order gradient synchronization")
        loss.backward(**kwargs)
        self.forward_pending = bool(kwargs.get("retain_graph", False))

    def gradient_divisor(self, local_divisor, *, normalize=True, reduction="legacy"):
        # Native DDP averages numerators. Divide by the average denominator to
        # obtain sum(rank numerators) / sum(rank denominators), including unequal
        # token counts and partial accumulation windows.
        # Scale before summation so very large finite weights do not overflow
        # merely because several ranks participate.
        if reduction == "global_sum":
            return local_divisor / distributed.world_size()
        if not normalize:
            return local_divisor
        divisor = distributed.reduce_number(local_divisor / distributed.world_size())
        if not isfinite(divisor) or divisor <= 0:
            raise ValueError("Distributed gradient divisor must be positive and finite")
        return divisor

    def observe(self, checks, kind, tensors, *, name=""):
        if not self.interested(checks, kind):
            return
        if self.model.training and torch.is_grad_enabled():
            with distributed.collective_errors("observer"):
                checks.observe(kind, tensors, name=name)
        else:
            checks.observe(kind, tensors, name=name)

    def interested(self, checks, kind):
        # All ranks join the union's observer slots even if local legacy checks
        # are disabled. Local observer execution remains independently gated.
        return (checks.allows(kind, collective=True)
                and (self._observer_interests is None or kind in self._observer_interests))

    def configure_checks(self, checks, configuration):
        distributed.require_equal(configuration, "diagnostic policy and mode schedule")
        subscriptions = [None] * distributed.world_size()
        torch.distributed.all_gather_object(subscriptions, checks.interests())
        self._observer_interests = (None if any(item is None for item in subscriptions) else
                                    set().union(*(set(item) for item in subscriptions)))

    def accumulation_context(self, *, synchronize):
        return nullcontext() if self.wrapper is None or synchronize else self.wrapper.no_sync()

    def validate_window(self, normalize, reduction):
        code = ("legacy", "rank_mean", "global_mean", "global_sum").index(reduction)
        # One tensor exchange per update, rather than pickling a boolean for
        # every microbatch. Max(x), -max(-x) give the common bounds.
        values = torch.tensor([int(normalize), -int(normalize), code, -code],
                              dtype=torch.int64, device=distributed.collective_device())
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
        high, negative_low, mode_high, mode_negative_low = values.tolist()
        if high != -negative_low or mode_high != -mode_negative_low:
            raise ValueError("DDP requires matching weighted/unweighted objective modes and distributed reduction on every rank")

    def sync_buffers(self):
        if not self.validation_sync_buffers:
            return
        for buffer in self.model.buffers():
            torch.distributed.broadcast(buffer, src=0)

    def check_errors(self, operation):
        return distributed.collective_errors(operation)

    def require_equal(self, value, description):
        distributed.require_equal(value, description)
