"""Optional automatic accumulation, independent of epochs and metrics."""

import torch
from math import isfinite
from numbers import Real

from deepopt.tasks.base import BatchOutput
from deepopt.engine.execution.runtime import TrainingRuntime
from deepopt.engine.strategies.loss import BackwardLoss, ExplicitObjective, LossStrategy


class OptimizationWindow:
    """Accumulate, normalize, and update at complete or partial boundaries.

    backward(output) returns True when it applies an optimizer update. Call
    flush() after the last batch to apply any pending partial window. Manual
    algorithms can skip this helper and use TrainingRuntime directly. Only
    checkpoint at boundaries; pending gradients are not serialized. A failed
    window discards its pending gradients and cannot be reused after recovery.
    """

    def __init__(self, runtime: TrainingRuntime, *, accumulation_steps: int = 1,
                 loss_strategy: LossStrategy | None = None):
        if type(accumulation_steps) is not int or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
        if runtime.optimizer is None:
            raise ValueError("Automatic optimization requires an optimizer")
        self.runtime = runtime
        self.accumulation_steps = accumulation_steps
        self.loss_strategy = loss_strategy if loss_strategy is not None else ExplicitObjective()
        if not isinstance(self.loss_strategy, LossStrategy):
            raise TypeError("loss_strategy must inherit LossStrategy")
        self.pending_batches = 0
        self._weight = 0
        self._normalize = False
        self._reduction = "legacy"
        self._failed = False

    def _require_healthy(self) -> None:
        if self._failed:
            raise RuntimeError("This optimization window failed; restore training state and "
                               "create a new OptimizationWindow before continuing")

    def backward(self, output: BatchOutput) -> bool:
        self._require_healthy()
        try:
            return self._backward(output)
        except BaseException:
            self._failed = True
            self.pending_batches = self._weight = 0
            self._normalize = False
            self.runtime.zero_grad()
            raise

    def _backward(self, output: BatchOutput) -> bool:
        with self.runtime.diagnostic_scope("deepopt.loss", phase="loss"), \
             self.runtime.core.check_errors("loss preparation"):
            prepared, normalize, total_weight = self._prepare(output)
        self.runtime.backward(prepared.loss)
        self._normalize = normalize
        self._reduction = prepared.distributed_reduction
        self._weight = total_weight
        self.pending_batches += 1
        return self.flush() if self.pending_batches == self.accumulation_steps else False

    def _prepare(self, output):
        output.validate(check_finite=False)
        prepared = self.loss_strategy.prepare(output, accumulation_steps=
                                             max(2, self.accumulation_steps) if self.runtime.distributed
                                             and isinstance(self.loss_strategy, ExplicitObjective)
                                             else self.accumulation_steps)
        if (not isinstance(prepared, BackwardLoss) or not isinstance(prepared.loss, torch.Tensor)
                or prepared.loss.ndim != 0):
            raise ValueError("LossStrategy.prepare must return a BackwardLoss with a scalar tensor")
        if not prepared.loss.is_floating_point():
            raise ValueError("LossStrategy.prepare must return a real floating-point loss")
        if prepared.distributed_reduction not in ("legacy", "rank_mean", "global_mean", "global_sum"):
            raise ValueError("Invalid distributed reduction policy")
        if self.pending_batches and prepared.distributed_reduction != self._reduction:
            raise ValueError("Cannot change distributed reduction within an accumulation window")
        weight = prepared.normalization_weight
        if weight is not None:
            try:
                valid_weight = (isinstance(weight, Real) and not isinstance(weight, bool)
                                and isfinite(weight) and weight > 0)
            except (OverflowError, TypeError):
                valid_weight = False
            if not valid_weight:
                raise ValueError("normalization_weight must be a positive finite real number or None")
            weight = float(weight)
        normalize = weight is not None
        if self.pending_batches and normalize != self._normalize:
            raise ValueError("Cannot mix weighted and unweighted objectives within one accumulation window")
        total_weight = self._weight + (weight or 0)
        try:
            finite_weight = isfinite(total_weight)
        except OverflowError:
            finite_weight = False
        if not finite_weight:
            raise ValueError("Accumulated normalization weight must be finite")
        return prepared, normalize, total_weight

    def flush(self) -> bool:
        self._require_healthy()
        if not self.pending_batches:
            return False
        try:
            self.runtime.step(gradient_divisor=self._weight if self._normalize else 1,
                              normalize=self._normalize, distributed_reduction=self._reduction)
        except BaseException:
            self._failed = True
            self.pending_batches = self._weight = 0
            self._normalize = False
            self.runtime.zero_grad()
            raise
        self.pending_batches = self._weight = 0
        self._normalize = False
        return True
