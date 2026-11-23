"""Lazy policy controls shared by portable installed-library benchmarks.

This module is used only by benchmark programs. Importing it does not import
Torch, either optional GPU compiler, or a kernel implementation.
"""

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
import json
import math
from time import perf_counter

from deepopt.kernel_policy import KERNEL_FAMILIES, KernelPolicy, kernel_policy_scope, resolve_kernel_policy


@dataclass(frozen=True)
class BenchmarkBackend:
    policy: KernelPolicy

    @classmethod
    def from_args(cls, args):
        backend = getattr(args, "deepopt_backend", "triton")
        if backend not in ("triton", "tilelang", "mixed"):
            raise ValueError("--deepopt-backend must be triton, tilelang, or mixed")
        options = getattr(args, "deepopt_kernel_policy", {})
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError as error:
                raise ValueError("--deepopt-kernel-policy must be a JSON object") from error
        if not isinstance(options, dict):
            raise ValueError("--deepopt-kernel-policy must be a JSON object")
        mapping = dict(options or {})
        if "backend" in mapping and mapping["backend"] != backend:
            raise ValueError("--deepopt-kernel-policy backend must match --deepopt-backend")
        mapping["backend"] = backend
        return cls(resolve_kernel_policy(mapping))

    @property
    def config(self):
        # Return fresh dictionaries; model configuration resolution may copy or
        # merge them, while the owning policy remains immutable.
        return {"runtime": {"kernels": self.policy.as_dict()}}

    def fp8_backend(self, device):
        return "reference" if str(device).split(":", 1)[0] == "cpu" else "inherit"

    def scope(self):
        return kernel_policy_scope(self.policy)

    def metadata(self, device):
        cpu = str(device).split(":", 1)[0] == "cpu"
        routes = {family: self.policy.route(family) for family in sorted(KERNEL_FAMILIES)}
        selected = set(routes.values()) | set(dict(self.policy.roles).values())
        if self.policy.fallback == "other":
            selected.update(("triton", "tilelang"))
        packages = {}
        if not cpu:
            for name in sorted(selected):
                try:
                    packages[name] = version(name)
                except PackageNotFoundError:
                    packages[name] = None
        return {
            "policy": self.policy.as_dict(),
            "fingerprint": self.policy.fingerprint,
            "resolved_families": routes,
            "role_overrides": dict(self.policy.roles),
            "route_evidence": "policy_resolution",
            "execution": "cpu_reference" if cpu else "cuda",
            "packages": packages,
            "note": "Resolved routes describe selection; GPU traces establish actual launches. "
                    "CPU reference timings do not establish GPU performance.",
        }


@dataclass(frozen=True)
class BenchmarkTimingPlan:
    requested_warmup: int
    requested_iterations: int
    requested_repeats: int
    warmup: int
    iterations: int
    repeats: int
    time_budget_seconds: float
    pilot_seconds: float = 0.0
    cold_first_call_seconds: float = 0.0
    calibration_calls: int = 0

    def sampling(self, elapsed_seconds):
        return {
            "requested_warmup": self.requested_warmup,
            "requested_iterations": self.requested_iterations,
            "requested_repeats": self.requested_repeats,
            "actual_warmup": self.warmup,
            "actual_iterations_per_repeat": [self.iterations] * self.repeats,
            "actual_repeats": self.repeats,
            "time_budget_seconds": self.time_budget_seconds,
            "pilot_seconds": self.pilot_seconds,
            "cold_first_call_seconds": self.cold_first_call_seconds,
            "calibration_calls": self.calibration_calls,
            "budget_exhausted": (self.warmup, self.iterations, self.repeats) != (
                self.requested_warmup, self.requested_iterations, self.requested_repeats),
            "elapsed_seconds": elapsed_seconds,
        }


def timing_plan_from_cost(args, pilot_seconds, *, iterations=None, repeats=None, cold_seconds=0.0, calibration_calls=0):
    """Use requested counts as caps, retaining at least one real timed sample.

    The budget is a target for warmed measurement, not an interrupt for a
    single GPU launch or compiler invocation. Zero preserves exhaustive counts.
    """
    warmup = args.warmup
    iterations = args.iterations if iterations is None else iterations
    repeats = args.repeats if repeats is None else repeats
    seconds = float(getattr(args, "time_budget", 0.0))
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("--time-budget must be finite and nonnegative (0 disables it)")
    if warmup < 0 or iterations < 1 or repeats < 1:
        raise ValueError("measurement counts must be positive; warmup may be zero")
    if not math.isfinite(pilot_seconds) or pilot_seconds < 0:
        raise ValueError("pilot duration must be finite and nonnegative")
    actual_warmup, actual_iterations, actual_repeats = warmup, iterations, repeats
    if seconds and pilot_seconds > 0:
        available = max(0.0, seconds - (pilot_seconds if calibration_calls else 0.0))
        actual_warmup = min(warmup, int(available * .1 / pilot_seconds))
        available = max(0.0, available - actual_warmup * pilot_seconds)
        actual_repeats = min(repeats, max(1, int(available / pilot_seconds)))
        actual_iterations = min(iterations, max(1, int(available / (actual_repeats * pilot_seconds))))
    return BenchmarkTimingPlan(warmup, iterations, repeats, actual_warmup, actual_iterations, actual_repeats,
                               seconds, pilot_seconds, cold_seconds, calibration_calls)


def plan_timing(operation, args, *, synchronize=lambda: None, clock=perf_counter):
    """Warm one cold call, then calibrate using a separately synchronized call."""
    seconds = float(getattr(args, "time_budget", 0.0))
    # Validate before executing even a pilot operation.
    initial = timing_plan_from_cost(args, 0.0)
    if not seconds:
        return initial
    synchronize()
    began = clock()
    operation()
    synchronize()
    cold = clock() - began
    began = clock()
    operation()
    synchronize()
    return timing_plan_from_cost(args, max(clock() - began, 1e-9), cold_seconds=cold, calibration_calls=2)


def capture_warmup_steps(args):
    """Keep the existing three-step capture minimum without costly repeats."""
    return 3 if getattr(args, "time_budget", 0.0) else max(3, args.warmup)


def share_timing_counts(plan, previous):
    """Retain each variant's calibration evidence with equal stateful updates."""
    return replace(plan, warmup=previous.warmup, iterations=previous.iterations, repeats=previous.repeats)
