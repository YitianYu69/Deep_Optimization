"""Execute one normalized, optionally clipped optimizer update."""

import torch
from contextlib import nullcontext

from deepopt.observers import CheckObservers
from deepopt.engine.strategies.gradients import ClipGradNorm, ClipGradValue, GradientStrategy


class OptimizerStepper:
    def __init__(self, model, optimizer, precision_policy, gradient_strategy=None, *, checks=None):
        if gradient_strategy is not None and not isinstance(gradient_strategy, GradientStrategy):
            raise TypeError("gradient_strategy must inherit GradientStrategy")
        if any(p.device != precision_policy.device for p in model.parameters()):
            raise ValueError("Move the model to the training device before constructing its optimizer")
        optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(set(optimizer_ids)) != len(optimizer_ids):
            raise ValueError("Optimizer parameters must not be repeated")
        if not set(optimizer_ids).issubset({id(p) for p in model.parameters()}):
            raise ValueError("Optimizer contains parameters outside the model")
        self.model, self.optimizer = model, optimizer
        self.gradient_strategy = gradient_strategy
        self.checks = checks if checks is not None else CheckObservers()

    def parameters(self):
        """The parameters owned by this optimizer, including frozen entries."""
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    def require_complete_optimizer(self) -> None:
        model_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        optimizer_ids = [id(p) for p in self.parameters()]
        if len(set(optimizer_ids)) != len(optimizer_ids) or not model_ids.issubset(optimizer_ids):
            raise ValueError("Every trainable model parameter must appear once in the optimizer")

    @staticmethod
    def _values(parameter):
        gradient = parameter.grad
        if gradient.is_sparse:
            # Duplicate sparse indices must be combined before finite checking:
            # individually finite values can overflow when added together.
            parameter.grad = gradient = gradient.coalesce()
            return gradient.values()
        if gradient.layout != torch.strided:
            raise ValueError("Optimizer gradients must be dense or sparse COO tensors")
        return gradient

    def step(self, gradient_divisor: float) -> None:
        """Validate all gradients before applying any optimizer updates."""
        guard = getattr(self.checks, "check_errors", lambda operation: nullcontext())
        observe_gradients = self.checks.interested("gradients")
        observe_weights = self.checks.interested("weights")
        if gradient_divisor == 1 and self.gradient_strategy is None and not observe_gradients and not observe_weights:
            # Let the optimizer own its parameter traversal. No diagnostic
            # gradient lists, sparse coalescing passes, or tensor scans here.
            with guard("optimizer update"):
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            return
        parameters = self.parameters()
        with guard("gradient inspection"):
            gradients = [self._values(p) for p in parameters if p.grad is not None]
        self.checks.observe("gradients", gradients, name="gradient")
        if gradient_divisor != 1:
            for gradient in gradients:
                gradient.div_(gradient_divisor)
            self.checks.observe("gradients", gradients, name="normalized gradient")
        if self.gradient_strategy is not None:
            if (isinstance(self.gradient_strategy, (ClipGradNorm, ClipGradValue))
                    and any(p.grad is not None and p.grad.is_sparse for p in parameters)):
                raise ValueError("Built-in gradient clipping does not support sparse gradients; "
                                 "provide a sparse-aware GradientStrategy or omit clipping")
            with guard("gradient processing"):
                diagnostic = self.gradient_strategy.process(iter(parameters))
            if isinstance(diagnostic, torch.Tensor):
                self.checks.observe("gradients", diagnostic, name="gradient processing result")
            # Inspect the actual replacement gradients, not the earlier views.
            processed = [self._values(p) for p in parameters if p.grad is not None]
            self.checks.observe("gradients", processed, name="gradient after processing")
        if observe_weights:
            self.checks.observe("weights", parameters, name="weights before update")
        with guard("optimizer update"):
            self.optimizer.step()
        if observe_weights:
            self.checks.observe("weights", parameters, name="weights after update")
        self.optimizer.zero_grad(set_to_none=True)
