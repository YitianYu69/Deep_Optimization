"""Metadata-only transfer calibration and latest-feasible prefetch planning.

Predicted boundaries are hints, never permission to allocate GPU storage. The
transport runtime must also observe device progress, event leases and credits.
No bandwidth is inferred from a GPU model name or an asynchronous enqueue.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from statistics import median


def _fit(samples):
    """Fit event latency + bytes/bandwidth inside one observed size domain.

    A tiny message's fixed launch gap must never become the byte rate of a
    gigabyte transfer. The median pair slope is robust to occasional queued
    copies; a repeated single size identifies latency at that size only.
    """
    slopes = [(right.elapsed_ms - left.elapsed_ms) / (right.nbytes - left.nbytes)
              for i, left in enumerate(samples) for right in samples[i + 1:]
              if right.nbytes != left.nbytes]
    slope = max(0.0, median(slopes)) if slopes else 0.0
    latency = max(0.0, median(sample.elapsed_ms - slope * sample.nbytes for sample in samples))
    return latency, slope


def _prediction(samples, nbytes):
    latency, slope = _fit(samples)
    return max(0.0, latency + slope * nbytes)


def _freeze(value):
    if isinstance(value, str):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not isfinite(value):
            raise ValueError("Offload cost identity requires finite metadata")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("Offload cost identity keys must be strings")
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    raise ValueError("Offload cost profiles cannot retain tensors or runtime resources")


def _number(value, name, *, positive=False):
    try:
        valid = type(value) in (int, float) and isfinite(value) and value >= 0 and (not positive or value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def _direction(value):
    if type(value) is not str or value.upper() not in ("D2H", "H2D"):
        raise ValueError("Transfer direction must be D2H or H2D")
    return value.upper()


@dataclass(frozen=True)
class TransferCostSample:
    direction: str
    nbytes: int
    elapsed_ms: float
    enqueue_ms: float = 0.0
    queue_ms: float = 0.0
    load: str = "compute"

    def __post_init__(self):
        object.__setattr__(self, "direction", _direction(self.direction))
        if type(self.nbytes) is not int or self.nbytes < 1:
            raise ValueError("Transfer sample bytes must be a positive integer")
        for name in ("elapsed_ms", "enqueue_ms", "queue_ms"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=name == "elapsed_ms"))
        if self.load not in ("idle", "compute", "bidirectional_compute"):
            raise ValueError("Unknown transfer contention domain")


@dataclass(frozen=True)
class OffloadCostProfile:
    samples: tuple[TransferCostSample, ...] = ()
    identity: tuple = ()
    safety_margin_ms: float = 0.0

    def __post_init__(self):
        if type(self.samples) is not tuple or any(type(sample) is not TransferCostSample for sample in self.samples):
            raise ValueError("Offload profile samples must be immutable measured metadata")
        object.__setattr__(self, "identity", _freeze(self.identity))
        object.__setattr__(self, "safety_margin_ms", _number(self.safety_margin_ms, "safety_margin_ms"))

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def as_dict(self):
        return {"schema_version": 2, "identity": self.identity,
                "samples": tuple(asdict(sample) for sample in self.samples),
                "safety_margin_ms": self.safety_margin_ms,
                "accounting": "size_local_event_latency_plus_bandwidth_separate_enqueue_and_queue",
                "prediction_validation": "held_out_event_error_within_max_2ms_25percent; minimum_five_local_samples"}

    def validate_identity(self, value):
        if _freeze(value) != self.identity:
            raise ValueError("Stale offload cost profile GPU/software/recipe/schedule identity")

    def estimate_ms(self, direction, nbytes, *, load="compute"):
        return self.prediction(direction, nbytes, load=load)["estimated_ms"]

    def prediction(self, direction, nbytes, *, load="compute"):
        direction = _direction(direction)
        if type(nbytes) is not int or nbytes < 1:
            raise ValueError("Estimated transfer bytes must be a positive integer")
        if load not in ("idle", "compute", "bidirectional_compute"):
            raise ValueError("Unknown transfer contention domain")
        # Restrict fitting to a local scale. A 264-byte observation cannot
        # influence the prediction for a 1.27-GB bundle.
        samples = tuple(sample for sample in self.samples if sample.direction == direction and sample.load == load
                        and nbytes / 4 <= sample.nbytes <= nbytes * 4)
        if not samples:
            return dict(estimated_ms=None, qualified=False, samples=0, held_out=0,
                        reason="unobserved_size_or_load_domain")
        low, high = min(s.nbytes for s in samples), max(s.nbytes for s in samples)
        if not low / 2 <= nbytes <= high * 2:
            return dict(estimated_ms=None, qualified=False, samples=len(samples), held_out=0,
                        reason="unsupported_size_extrapolation")
        held = samples[4::5]
        train = tuple(sample for index, sample in enumerate(samples) if index % 5 != 4)
        passed = sum(abs(_prediction(train, sample.nbytes) - sample.elapsed_ms)
                     <= max(2.0, .25 * sample.elapsed_ms) for sample in held) if train else 0
        qualified = len(samples) >= 5 and bool(held) and passed / len(held) >= .9 and low <= nbytes <= high
        latency, slope = _fit(samples)
        event = max(0.0, latency + slope * nbytes)
        # Nonnegative observed residual protects against optimistic fits,
        # without conflating fixed costs with a byte-rate extrapolation.
        residual = max(0.0, max(sample.elapsed_ms - (latency + slope * sample.nbytes) for sample in samples))
        overhead = max(sample.enqueue_ms + sample.queue_ms for sample in samples)
        return dict(estimated_ms=event + residual + overhead + self.safety_margin_ms,
                    event_ms=event, enqueue_queue_ms=overhead, qualified=qualified,
                    samples=len(samples), held_out=len(held), held_out_passed=passed,
                    minimum_observed_bytes=low, maximum_observed_bytes=high,
                    reason="held_out_validated" if qualified else "insufficient_or_inaccurate_held_out_evidence")


class TransferCostModel:
    """Bounded warm-up collector; freezing disables further learning."""

    def __init__(self, *, max_samples=64, safety_margin_ms=0.0):
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        self.max_samples = max_samples
        self.safety_margin_ms = _number(safety_margin_ms, "safety_margin_ms")
        self._samples = []
        self._frozen = False

    def observe(self, direction, nbytes, elapsed_ms, *, enqueue_ms=0.0, queue_ms=0.0, load="compute"):
        if self._frozen:
            raise RuntimeError("Offload cost learning is frozen for accepted timing")
        sample = TransferCostSample(direction, nbytes, elapsed_ms, enqueue_ms, queue_ms, load)
        self._samples.append(sample)
        # Balance direction/load/log-size strata. Protecting the worst byte
        # rate permanently preserved tiny-message latency as fake bandwidth.
        if len(self._samples) > self.max_samples:
            groups = {}
            for index, current in enumerate(self._samples):
                key = current.direction, current.load, current.nbytes.bit_length() // 2
                groups.setdefault(key, []).append(index)
            victim = min(groups.values(), key=lambda indices: (-len(indices), indices[0]))[0]
            self._samples.pop(victim)
        return sample

    def estimate_ms(self, direction, nbytes, *, load="compute"):
        return OffloadCostProfile(tuple(self._samples), safety_margin_ms=self.safety_margin_ms).estimate_ms(direction, nbytes, load=load)

    def freeze(self, *, profile_identity=None):
        profile = OffloadCostProfile(tuple(self._samples), _freeze({} if profile_identity is None else profile_identity), self.safety_margin_ms)
        self._frozen = True
        return profile


@dataclass(frozen=True)
class PrefetchRequest:
    bundle_id: str
    consumer_id: str
    consumer_boundary: int
    submit_boundary: int
    ready_deadline_ms: float | None = None
    estimated_copy_ms: float | None = None
    calibrated: bool = False
    requires_device_progress: bool = True

    def __post_init__(self):
        if (type(self.bundle_id) is not str or not self.bundle_id or type(self.consumer_id) is not str
                or not self.consumer_id or type(self.submit_boundary) is not int or self.submit_boundary < 0
                or type(self.consumer_boundary) is not int or self.consumer_boundary < self.submit_boundary
                or type(self.calibrated) is not bool or self.requires_device_progress is not True):
            raise ValueError("Invalid bounded offload prefetch request")
        for name in ("ready_deadline_ms", "estimated_copy_ms"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _number(value, name))
        if self.calibrated and (self.ready_deadline_ms is None or self.estimated_copy_ms is None):
            raise ValueError("Calibrated prefetch requires measured copy and compute costs")


def plan_prefetches(plan, compute_durations_ms=(), profile=None, *, load="compute"):
    """Return bounded immutable copy hints at existing executable boundaries.

    Missing or unqualified calibration submits at the actual consumer. Early
    requests require held-out copy validation and complete measured compute
    intervals. Runtime pressure may delay a hint and reports that outcome.
    """
    if not plan.policy.enabled or not plan.bundles:
        return ()
    if profile is not None and not isinstance(profile, OffloadCostProfile):
        raise ValueError("Expected a frozen offload cost profile")
    if profile is not None:
        profile.validate_identity(plan.profile_identity)
    durations = dict(enumerate(compute_durations_ms)) if isinstance(compute_durations_ms, (tuple, list)) else dict(compute_durations_ms)
    if any(type(index) is not int or index < 0 for index in durations):
        raise ValueError("Compute costs require nonnegative schedule boundaries")
    durations = {index: None if value is None else _number(value, "compute duration") for index, value in durations.items()}
    maximum = max((consumer.schedule_index for consumer in plan.consumers), default=0)
    # All deadlines are relative to the first emitted progress marker. Input
    # binding indices before it have no measured GPU interval and must not
    # permanently poison otherwise complete coarse-boundary calibration.
    origin = min(plan.boundary_ids, default=0)
    starts, elapsed, known = {}, 0.0, True
    for boundary in range(origin, maximum + 1):
        starts[boundary] = elapsed if known else None
        duration = durations.get(boundary)
        if duration is None:
            known = False
        else:
            elapsed += duration
    bundles = {bundle.bundle_id: bundle for bundle in plan.bundles}
    boundaries = plan.boundary_ids
    previous_use = {}
    requests = []
    for consumer in plan.consumers:
        # A named panel executor schedules its own bounded group windows. A
        # whole-bundle prefetch here would defeat its working-set contract.
        if consumer.panel is not None:
            continue
        predictions = {name: profile.prediction("H2D", bundles[name].bytes, load=load) if profile is not None else
                       dict(estimated_ms=None, qualified=False) for name in consumer.bundle_ids}
        costs = {name: prediction["estimated_ms"] for name, prediction in predictions.items()}
        calibrated = (durations.get(origin) is not None and starts.get(consumer.schedule_index) is not None
                      and all(value is not None for value in costs.values())
                      and all(prediction["qualified"] for prediction in predictions.values()))
        latest = starts[consumer.schedule_index] - sum(costs.values()) if calibrated else None
        for name in consumer.bundle_ids:
            if plan.policy.mode == "demand":
                submit = consumer.schedule_index
            elif calibrated:
                permitted = [boundary for boundary in boundaries if boundary <= consumer.schedule_index
                             and boundary >= previous_use.get(name, 0) and starts.get(boundary) is not None
                             and starts[boundary] <= latest]
                submit = max(permitted, default=max(previous_use.get(name, 0), min(boundaries, default=0)))
            else:
                # Unqualified costs use demand. Early speculative restores
                # without supported deadlines consume memory and cause copies
                # which the next actual consumer immediately evicts.
                submit = consumer.schedule_index
            requests.append(PrefetchRequest(name, consumer.consumer_id, consumer.schedule_index, submit,
                starts.get(consumer.schedule_index) if calibrated else None, costs[name] if calibrated else None, calibrated))
            previous_use[name] = consumer.schedule_index
    return tuple(requests)
