"""Engine's single-process training core; no distributed collectives."""

from contextlib import nullcontext


class TrainingCore:
    """Local execution policy used by TrainingRuntime in a single process.

    Owns forward/backward execution, local observer dispatch and local gradient
    normalization semantics. DDPCore implements the same interface independently.
    Runtime owns shared precision, optimizer execution and failure-state handling;
    Trainer owns epochs, accumulation scheduling, metrics and callbacks.
    """

    distributed = False

    def __init__(self, model, device):
        self.model, self.device = model, device

    def compute(self, batch, step):
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
