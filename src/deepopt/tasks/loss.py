"""Optional loss weighting for advanced steps; ordinary losses are tensors."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from torch import Tensor
from ..observers import FiniteCheck, TensorEvent


def validate_weight(value, name="normalization weight") -> None:
    try:
        valid = not isinstance(value, bool) and isinstance(value, Real) and isfinite(value) and value > 0
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a positive finite real number")


def validate_loss(value: Tensor, *, check_finite: bool = True) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.numel() == 0:
        raise ValueError("Loss must be a nonempty real floating-point tensor")
    if check_finite:
        FiniteCheck()(TensorEvent("loss", "loss", (value,)))


@dataclass(frozen=True)
class LossReport:
    """Explicit reporting numerator and denominator; unrelated to batch size."""

    total: Tensor
    weight: float

    def validate(self, *, check_finite: bool = True):
        validate_loss(self.total, check_finite=check_finite)
        if self.total.ndim != 0:
            raise ValueError("Reporting total must be a scalar tensor")
        validate_weight(self.weight, "reporting weight")


@dataclass(frozen=True)
class LossResult:
    """A literal objective, optional accumulation weight, and optional report.

    No weight means backward the scalar unchanged, including accumulation.
    A weight declares that this scalar is normalized by that amount: accumulate
    value * weight and divide gradients by total weight. A custom LossStrategy
    can consume unreduced values; the default strategy requires a scalar.
    No report means the framework does not invent a loss summary.
    """

    value: Tensor
    normalization_weight: float | None = None
    report: LossReport | None = None

    def validate(self, *, check_finite: bool = True):
        validate_loss(self.value, check_finite=check_finite)
        if self.normalization_weight is not None:
            validate_weight(self.normalization_weight)
            if self.value.ndim != 0:
                raise ValueError("A normalized objective must already be a scalar")
        if self.report is not None:
            if not isinstance(self.report, LossReport):
                raise ValueError("report must be a LossReport or None")
            self.report.validate(check_finite=check_finite)

    @classmethod
    def from_mean(cls, value: Tensor, *, weight: float):
        """Declare an already-computed mean and its actual denominator."""
        validate_weight(weight)
        weight = float(weight)
        return cls(value, weight, LossReport(value.detach().double() * weight, weight))

    @classmethod
    def from_sum(cls, value: Tensor, *, report_weight: float):
        """Preserve an already-computed sum and explicitly choose its report divisor."""
        validate_weight(report_weight, "reporting weight")
        report_weight = float(report_weight)
        return cls(value, report=LossReport(value.detach().double(), report_weight))


