"""Checkpoint-free VAE trajectory oracle. No tensor is persisted to disk.

The reference cache has an explicit byte ceiling. Device graphs are consumed
and released before the other route runs. All elements are compared in bounded
CPU chunks; only scalar errors, names and hashes belong in JSON evidence.
"""

import gc
import json
import math
import random
import time
from contextlib import contextmanager, nullcontext

import torch


CHUNK_ELEMENTS = 262144
MAX_ORACLE_COMPARISONS = 16384


def _metadata_copy(value):
    """Copy JSON metadata without formatting or retaining tensor payloads."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_metadata_copy(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _metadata_copy(item) for key, item in value.items()}
    raise TypeError("Qualification evidence accepts JSON metadata only")


class NumericalComparisonError(AssertionError):
    """A failed unchanged numerical gate with portable partial evidence."""

    def __init__(self, comparison, oracle_report):
        self.comparison = _metadata_copy(comparison)
        self.oracle_report = _metadata_copy(oracle_report)
        self.qualification_evidence = {"schema_version": 1, "passed": False, "partial": True,
                                       "oracle_report": _metadata_copy(oracle_report),
                                       "context": _metadata_copy(oracle_report.get("context", {}))}
        super().__init__("VAE oracle mismatch: " + json.dumps(self.comparison, allow_nan=False))


@contextmanager
def _route_observation(observing):
    """Diagnostic cleanup cannot suppress or replace a numerical failure."""
    observing.__enter__()
    try:
        yield
    except BaseException as error:
        try:
            observing.__exit__(type(error), error, error.__traceback__)
        except BaseException as cleanup_error:
            try:
                error.diagnostic_context_error = {"type": type(cleanup_error).__name__}
            except Exception:
                pass
        raise
    else:
        observing.__exit__(None, None, None)


def _notify_progress(callback, phase, **context):
    """Publish host-only diagnostic metadata without affecting the operation."""
    if callback is not None:
        try:
            callback({"phase": phase, **context})
        except Exception:
            # A progress observer must not replace a numerical/device failure.
            pass


def assert_bf16_forward_domains(input_dtype, output_dtypes, amp_enabled, amp_dtype):
    expected = {"reconstruction": "torch.bfloat16", "mu": "torch.bfloat16", "logvar": "torch.bfloat16",
                "loss": "torch.float32", "l1": "torch.float32", "kl": "torch.float32"}
    if (not amp_enabled or amp_dtype != torch.bfloat16 or input_dtype != torch.float32
            or output_dtypes != expected):
        raise AssertionError(f"CUDA VAE BF16 recipe differs: AMP={amp_enabled}/{amp_dtype}, input={input_dtype}, outputs={output_dtypes}")


def saved_state_evidence(report, compressed):
    """Validate the actual differentiated regions, not selected-fusion counts."""
    from deepopt.compiler.reporting import classify_saved_state

    regions, values, classifications, categories, byte_counts, details = 0, 0, {}, {}, {}, []
    for graph in report.get("graphs", []):
        for region in graph.get("native_autodiff_graphs", []):
            regions += 1
            for value in classify_saved_state(region, compressed):
                values += 1
                category = value["category"]
                categories[category] = categories.get(category, 0) + 1
                byte_counts[category] = byte_counts.get(category, 0) + value["bytes"]
                for representation in value["representations"]:
                    classifications[representation] = classifications.get(representation, 0) + 1
                details.append({"region": regions - 1, **value})
    if not regions or not values:
        raise AssertionError("Native VAE produced no typed saved-state evidence")
    return {"passed": True, "regions": regions, "saved_values": values, "classifications": classifications,
            "categories": categories, "saved_tensor_bytes_by_category": byte_counts, "values": details,
            "byte_accounting_scope": "declared tensor bytes; backing storage and alias deduplication require the runtime storage ledger"}


def rng_state():
    import numpy as np

    state = np.random.get_state()
    return {"torch": torch.get_rng_state().clone(),
            "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": (state[0], torch.from_numpy(state[1].astype("int64")), *state[2:])}


def restore_rng(state):
    import numpy as np

    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype("uint32"), *numpy[2:]))


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def named_tree(value, prefix):
    """Include scalar optimizer counters and None gradient masks, not just tensors."""
    if isinstance(value, torch.Tensor) or value is None:
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from named_tree(value[key], f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        if value and all(isinstance(item, (bool, int, float)) for item in value):
            yield prefix, torch.tensor(value, dtype=torch.float64 if any(isinstance(item, float) for item in value) else None)
            return
        for index, item in enumerate(value):
            yield from named_tree(item, f"{prefix}/{index}")
    elif isinstance(value, (bool, int, float)):
        yield prefix, torch.tensor(value, dtype=torch.float64 if isinstance(value, float) else None)
    else:
        # Strings and structural policy are exact byte tensors.
        yield prefix, torch.tensor(list(repr(value).encode()), dtype=torch.uint8)


class TensorOracle:
    def __init__(self, max_bytes=512 * 2**20, chunk_elements=CHUNK_ELEMENTS, *, diagnostic=False,
                 max_comparisons=MAX_ORACLE_COMPARISONS, context=None):
        if max_bytes < 1 or chunk_elements < 1 or type(max_comparisons) is not int or max_comparisons < 1:
            raise ValueError("Oracle limits must be positive")
        if type(diagnostic) is not bool:
            raise TypeError("diagnostic must be a boolean")
        self.max_bytes, self.chunk_elements = max_bytes, chunk_elements
        self.reference, self.bytes, self.seen, self.errors = {}, 0, set(), []
        self.post_parameters = {}
        self.diagnostic, self.max_comparisons = diagnostic, max_comparisons
        self.context = _metadata_copy(context or {})
        self.recorded_count = 0
        self.metadata_limit_exceeded = False
        self.memory_plan = None

    @property
    def retained_bytes(self):
        """Live oracle payloads; ``bytes`` remains the complete reference total."""
        return (sum(value.numel() * value.element_size() for value in self.reference.values() if value is not None)
                + sum(value.numel() * value.element_size() for pair in self.post_parameters.values() for value in pair))

    def capacity_error(self, message, **details):
        report = self.snapshot()
        report["passed"] = False
        report["memory_error"] = _metadata_copy({"budget_bytes": self.max_bytes,
            "retained_bytes": self.retained_bytes, **details})
        error = MemoryError(message + ": " + json.dumps(report["memory_error"], allow_nan=False))
        error.oracle_report = report
        error.qualification_evidence = {"schema_version": 1, "passed": False, "partial": True,
                                        "oracle_report": _metadata_copy(report)}
        return error

    @staticmethod
    def _tensor_metadata(tensor):
        return {"present": tensor is not None,
                "shape": None if tensor is None else list(tensor.shape),
                "dtype": None if tensor is None else str(tensor.dtype)}

    def _comparison(self, name, expected, actual):
        return {"name": name, "expected": self._tensor_metadata(expected),
                "actual": self._tensor_metadata(actual), "context": _metadata_copy(self.context),
                "shape": None if actual is None else list(actual.shape),
                "elements": 0 if actual is None else actual.numel(),
                "dtype": None if actual is None else str(actual.dtype)}

    def snapshot(self):
        """Return bounded partial evidence without owning any tensor storage."""
        first = next((row for row in self.errors if not row["passed"]), None)
        complete = not self.reference and not self.post_parameters and not self.metadata_limit_exceeded
        return _metadata_copy({"schema_version": 1, "passed": complete and first is None,
                "partial": not complete, "exhaustive": complete, "sampled": False,
                "diagnostic_only": self.diagnostic, "qualification_eligible": not self.diagnostic,
                "reference_bytes": self.bytes, "byte_limit": self.max_bytes,
                "retained_bytes": self.retained_bytes, "reference_memory_plan": self.memory_plan,
                "chunk_elements": self.chunk_elements, "metadata_comparison_limit": self.max_comparisons,
                "metadata_limit_exceeded": self.metadata_limit_exceeded,
                "recorded_tensor_count": self.recorded_count, "completed_comparison_count": len(self.errors),
                "failure_count": sum(not row["passed"] for row in self.errors),
                "first_failure": first, "comparisons": self.errors,
                "unconsumed_names": sorted(self.reference), "unconsumed_count": len(self.reference),
                "pending_parameter_delta_names": sorted(self.post_parameters),
                "context": self.context, "tensor_payloads_in_evidence": False})

    def _raise_failure(self, comparison):
        try:
            report = self.snapshot()
        except Exception as error:
            report = {"schema_version": 1, "passed": False, "partial": True,
                      "context": _metadata_copy(self.context), "first_failure": None,
                      "diagnostic_collection_error": type(error).__name__,
                      "tensor_payloads_in_evidence": False}
        report["passed"] = False
        if report["first_failure"] is None:
            report["first_failure"] = _metadata_copy(comparison)
            report["failure_count"] = max(1, report.get("failure_count", 0))
        raise NumericalComparisonError(comparison, report)

    def _complete(self, comparison, failure=None):
        comparison = {**comparison, "passed": failure is None, "failure": failure}
        if len(self.errors) >= self.max_comparisons:
            self.metadata_limit_exceeded = True
            self._raise_failure({**comparison, "passed": False, "failure": "comparison_metadata_limit_exceeded"})
        self.errors.append(_metadata_copy(comparison))
        if failure is not None and not self.diagnostic:
            self._raise_failure(comparison)

    def raise_if_failed(self):
        """Stop a diagnostic candidate before clipping or optimizer effects."""
        failure = next((row for row in self.errors if not row["passed"]), None)
        if failure is not None:
            self._raise_failure(failure)

    def record(self, name, tensor):
        if name in self.reference:
            raise ValueError(f"Duplicate oracle tensor: {name}")
        if self.recorded_count >= self.max_comparisons:
            raise MemoryError("VAE oracle exceeded its named-comparison metadata limit")
        size = 0 if tensor is None else tensor.numel() * tensor.element_size()
        if self.retained_bytes + size > self.max_bytes:
            raise self.capacity_error("VAE eager oracle exceeded its explicit CPU byte budget",
                                      name=name, required_bytes=self.retained_bytes + size)
        self.reference[name] = None if tensor is None else tensor.detach().to("cpu", copy=True).contiguous()
        self.bytes += size
        self.recorded_count += 1

    def compare(self, name, tensor):
        if name not in self.reference or name in self.seen:
            self._complete(self._comparison(name, None, tensor), "unexpected_or_duplicate_candidate_tensor")
            return
        self.seen.add(name)
        expected = self.reference.pop(name)
        comparison = self._comparison(name, expected, tensor)
        if tensor is None or expected is None:
            self._complete(comparison, "gradient_or_state_presence_differs"
                           if tensor is not None or expected is not None else None)
            return
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            self._complete(comparison, "shape_or_dtype_differs")
            return
        if name.startswith("parameter/"):
            required = self.retained_bytes + expected.numel() * expected.element_size() + tensor.numel() * tensor.element_size()
            if required > self.max_bytes:
                raise self.capacity_error("Post-update comparison exceeds bounded oracle RAM", name=name, required_bytes=required)
            self.post_parameters[name.removeprefix("parameter/")] = (tensor.detach().cpu(), expected)
        if name.startswith("delta/"):
            parameter = self.post_parameters.pop(name.removeprefix("delta/"), None)
            if parameter is not None:
                from benchmarks.analysis.measurement import errors

                key = "parameter_update." + name.removeprefix("delta/")
                parameter_key = "parameter." + name.removeprefix("delta/")
                report = errors({key: tensor, parameter_key: parameter[0]},
                                {key: expected, parameter_key: parameter[1]},
                                rtol=8 * torch.finfo(torch.float32).eps,
                                atol=2 * torch.finfo(torch.float32).eps,
                                max_relative_l2=8 * torch.finfo(torch.float32).eps,
                                update_policy={"parameter_rounding_ulps": 2},
                                chunk_elements=self.chunk_elements)
                self._complete({**comparison, **report["tensors"][key]},
                               None if report["passed"] else "post_parameter_or_update_mismatch")
                return
        # Storage-domain gates are deliberately conservative. In particular an
        # FP32 affine reduction never borrows a BF16 coefficient tolerance.
        # Relaxing a failing field requires a separate accumulation oracle;
        # same-route repeat spread alone cannot justify cross-route bias.
        exact = (not tensor.is_floating_point()
                 or name.startswith(("rng/", "buffer/", "optimizer_policy/", "optimizer/param_groups/", "recipe/", "model/", "initial/"))
                 or name.split("#chunk")[0].endswith("/step"))
        epsilon = torch.finfo(tensor.dtype).eps if tensor.is_floating_point() else 0.
        rtol = 0.0 if exact else 8 * epsilon
        atol = 0.0 if exact else 2 * min(epsilon, torch.finfo(torch.float32).eps)
        relative_l2_limit = 0.0 if exact else 8 * epsilon
        if tensor.dtype == torch.bfloat16 and not exact:
            # Rounded BF16 outputs may differ by a final rounding boundary;
            # global L2 must still stay below one BF16 epsilon, not eight.
            rtol = 2 * epsilon
            relative_l2_limit = epsilon
        maximum, squared, reference_squared = 0., 0., 0.
        failure = None
        elementwise_passed = True
        actual = tensor.detach().reshape(-1)
        target = expected.reshape(-1)
        for start in range(0, actual.numel(), self.chunk_elements):
            a = actual[start:start + self.chunk_elements].cpu()
            b = target[start:start + self.chunk_elements]
            try:
                torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=False, msg=name)
            except AssertionError:
                failure = "elementwise_tolerance_exceeded"
                elementwise_passed = False
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                self._complete({**comparison, "exact": exact, "rtol": rtol, "atol": atol,
                                "relative_l2_limit": relative_l2_limit, "finite": False},
                               "nonfinite_values_cannot_qualify")
                return
            if a.numel():
                error = a.double() - b.double()
                maximum = max(maximum, error.abs().max().item())
                squared += error.square().sum().item()
                reference_squared += b.double().square().sum().item()
        relative_l2 = math.sqrt(squared / max(reference_squared, 1e-300))
        relative_l2_passed = math.isfinite(relative_l2) and relative_l2 <= relative_l2_limit
        if not math.isfinite(relative_l2):
            failure = "nonfinite_error_metric_cannot_qualify"
        elif relative_l2 > relative_l2_limit:
            failure = "relative_l2_exceeds_storage_domain_limit"
        self._complete({**comparison,
                            "max_abs": maximum, "relative_l2": relative_l2,
                            "relative_l2_limit": relative_l2_limit,
                            "rtol": rtol, "atol": atol, "exact": exact,
                            "arithmetic_domain": "exact_state" if exact else "strict_storage_dtype_roundoff",
                            "finite": True, "elementwise_passed": elementwise_passed,
                            "relative_l2_passed": relative_l2_passed,
                            "failure_reasons": (["elementwise_tolerance_exceeded"] if not elementwise_passed else [])
                                + (["relative_l2_exceeds_storage_domain_limit"] if not relative_l2_passed else [])}, failure)

    def finish(self):
        self.raise_if_failed()
        if self.post_parameters:
            name = "delta/" + next(iter(self.post_parameters))
            self._raise_failure({**self._comparison(name, self.reference.get(name), None), "passed": False,
                                 "failure": "candidate_omitted_post_parameter_update_deltas"})
        if self.reference:
            name = next(iter(self.reference))
            self._raise_failure({**self._comparison(name, self.reference[name], None), "passed": False,
                                 "failure": "candidate_omitted_oracle_tensors"})
        return self.snapshot()


class ChunkTransport:
    """Adapt named VAE tensors to the shared authenticated raw-byte protocol.

    The receiving comparer finishes before ACK. One bounded message is in
    flight; small fields share it and larger tensors use sequential chunks.
    """
    def __init__(self, sink, identity, chunk_elements=CHUNK_ELEMENTS):
        from concurrent.futures import ThreadPoolExecutor
        from benchmarks.analysis.numerical_ipc import NumericalSnapshotBroker

        self.sink, self.identity, self.chunk_elements = sink, identity, chunk_elements
        self.message_bytes = 16 * 2**20 if chunk_elements == CHUNK_ELEMENTS else chunk_elements * 4
        owner = getattr(sink, "__self__", None)
        self.owner = owner if isinstance(owner, TensorOracle) else None
        self.recording = self.owner is not None and getattr(sink, "__name__", None) == "record"
        if self.owner is not None:
            self.message_bytes = min(self.message_bytes, max(1, owner.max_bytes // 8))
        self.broker = NumericalSnapshotBroker(max_bytes=max(self.message_bytes, chunk_elements * 8) + 1024, timeout=600)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def __enter__(self):
        self.pending, self.pending_bytes = {}, 0
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                self.flush()
        finally:
            self.pending.clear()
            self.broker.close()
            self.executor.shutdown(wait=True)

    def _admit(self, staged_bytes, *, name=None):
        if self.owner is not None:
            required = self.owner.retained_bytes + staged_bytes
            if required > self.owner.max_bytes:
                raise self.owner.capacity_error("Named tensor staging exceeds bounded oracle RAM",
                    name=name, required_bytes=required, staged_bytes=staged_bytes,
                    pending_bytes=self.pending_bytes, message_bytes=self.message_bytes)

    def preflight_reference(self, fields):
        """Check the exact remaining recipe after lazy optimizer state exists."""
        if not self.recording:
            return
        fields = _metadata_copy(fields)
        by_category, by_optimizer_role = {}, {}
        for item in fields:
            category = item["name"].split("/")[0]
            by_category[category] = by_category.get(category, 0) + item["bytes"]
            if item["name"].startswith("optimizer/state/"):
                role = "/".join(item["name"].split("/")[3:])
                by_optimizer_role[role] = by_optimizer_role.get(role, 0) + item["bytes"]
        remaining = sum(item["bytes"] for item in fields)
        total = self.owner.bytes + self.pending_bytes + remaining
        largest = max(fields, key=lambda item: item["bytes"], default={"name": None, "bytes": 0})
        # Final reference + sender/receiver batching, or one assembled large
        # field and its two wire chunks. Comparison arithmetic has its own
        # fixed chunk_elements bound; model/optimizer storage is outside this cap.
        largest_chunk = max((min(item["bytes"], self.chunk_elements * item["element_size"])
                             for item in fields), default=0)
        staging = max(2 * self.message_bytes, largest["bytes"] + 2 * largest_chunk)
        parameter_pair = max((item["bytes"] for item in fields if item["name"].startswith("parameter/")), default=0)
        self.owner.memory_plan = {"scope": "oracle_retained_CPU_payloads_and_transport_staging",
            "budget_bytes": self.owner.max_bytes, "recorded_reference_bytes": self.owner.bytes,
            "remaining_reference_bytes": remaining, "required_reference_bytes": total,
            "remaining_bytes_by_category": by_category, "optimizer_state_bytes_by_role": by_optimizer_role,
            "remaining_named_tensor_count": len(fields), "largest_remaining_tensor": largest,
            "transport_staging_upper_bound_bytes": staging,
            "candidate_parameter_pair_upper_bound_bytes": parameter_pair,
            "reference_and_transport_upper_bound_bytes": total + staging + parameter_pair,
            "reference_fits_budget": total <= self.owner.max_bytes,
            "adaptive_staging_required": total + staging + parameter_pair > self.owner.max_bytes,
            "fields": fields}
        if total > self.owner.max_bytes:
            raise self.owner.capacity_error("Complete VAE reference cannot fit the declared oracle RAM budget",
                required_reference_bytes=total, intrinsic_reference_shortfall_bytes=total - self.owner.max_bytes)

    def _transfer(self, tensors, consume):
        from benchmarks.analysis.numerical_ipc import send_snapshot

        shapes = {key: None if value is None else tuple(value.shape) for key, value in tensors.items()}
        callback_errors = []

        def consumed(payload):
            try:
                consume(payload)
            except BaseException as error:
                # Publish the original comparer failure before the receiver
                # closes its socket. Future.done() alone has a close/result race.
                callback_errors.append((error, error.__traceback__))
                raise

        future = self.executor.submit(self.broker.receive, self.identity, shapes, on_snapshot=consumed)
        try:
            send_snapshot(self.identity, tensors, channel=self.broker.channel)
        except Exception:
            # Callback failures reject ACK; retain their precise numerical
            # cause. Connected/truncated/malformed transfers are never retried.
            if callback_errors:
                error, traceback = callback_errors[0]
                raise error.with_traceback(traceback)
            if future.done():
                future.result()
            raise
        future.result()

    def flush(self):
        if not self.pending:
            return
        tensors, self.pending = self.pending, {}
        self.pending_bytes = 0
        def consume(payload):
            for key in tensors:
                self.sink(key, payload["tensors"][key])
        self._transfer(tensors, consume)

    def __call__(self, name, tensor):
        size = 0 if tensor is None else tensor.numel() * tensor.element_size()
        # Small named tensors share one bounded message. A connection per
        # scalar exhausts ephemeral TCP ports during long trajectories.
        limit = self.message_bytes
        if size <= limit:
            if name in self.pending or len(self.pending) >= 64 or self.pending_bytes + size > limit:
                self.flush()
            # Recording also copies received bytes into the owned reference;
            # comparing retains only the sender and receiver transport copies.
            copies = 3 if self.recording else 2
            try:
                self._admit(copies * (self.pending_bytes + size), name=name)
            except MemoryError:
                if not self.pending:
                    raise
                self.flush()
                self._admit(copies * size, name=name)
            self.pending[name] = None if tensor is None else tensor.detach().to("cpu", copy=True).contiguous()
            self.pending_bytes += size
            return
        self.flush()
        chunk_bytes = min(size, self.chunk_elements * tensor.element_size())
        self._admit((2 if self.recording else 1) * size + 2 * chunk_bytes, name=name)
        flat = tensor.detach().reshape(-1)
        assembled = torch.empty(flat.numel(), dtype=flat.dtype, device="cpu")
        for offset in range(0, flat.numel(), self.chunk_elements):
            key = f"{name}#chunk{offset}"
            value = flat[offset:offset + self.chunk_elements].cpu().contiguous()
            def consume(payload, offset=offset, key=key):
                chunk = payload["tensors"][key]
                assembled[offset:offset + chunk.numel()].copy_(chunk)
                if offset + chunk.numel() == assembled.numel():
                    self.sink(name, assembled.reshape(tensor.shape))
            self._transfer({key: value}, consume)


def post_update_reference_fields(model, optimizer_state, random_state):
    """Exact emission inventory, inspecting metadata without cloning payloads."""
    fields = []

    def append(name, value):
        fields.append({"name": name, "shape": None if value is None else list(value.shape),
            "dtype": None if value is None else str(value.dtype),
            "elements": 0 if value is None else value.numel(),
            "element_size": 0 if value is None else value.element_size(),
            "bytes": 0 if value is None else value.numel() * value.element_size()})

    for name, parameter in model.named_parameters():
        append("parameter/" + name, parameter)
        append("delta/" + name, parameter)
    for name, buffer in model.named_buffers():
        append("buffer/" + name, buffer)
    for name, value in named_tree(optimizer_state, "optimizer"):
        append(name, value)
        if isinstance(value, torch.Tensor):
            # Only the encoded device name is emitted, not a tensor copy.
            fields.append({"name": "optimizer_policy/device/" + name,
                "shape": [len(value.device.type.encode())], "dtype": "torch.uint8",
                "elements": len(value.device.type.encode()), "element_size": 1,
                "bytes": len(value.device.type.encode())})
    for name, value in named_tree(random_state, "rng"):
        append(name, value)
    return fields


def move_optimizer(optimizer, device):
    destination = torch.device(device)
    placement = getattr(optimizer, "_vae_qualification_state_devices", {})

    def move(value, path):
        if isinstance(value, torch.Tensor):
            if destination.type == "cpu":
                placement[path] = value.device.type
                return value.cpu()
            # Preserve intentionally host-resident scalar/counter state. Moving
            # every leaf to CUDA would silently change the optimizer's recipe.
            target = "cpu" if placement.get(path) == "cpu" else destination
            return value.to(target)
        if isinstance(value, dict):
            return {key: move(item, (*path, str(key))) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, (*path, str(index))) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(move(item, (*path, str(index))) for index, item in enumerate(value))
        return value
    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = move(state, (id(parameter),))
    optimizer._vae_qualification_state_devices = placement


def resume_roundtrip(model, optimizer, max_bytes):
    """Exercise model/custom-optimizer/RNG restoration entirely in bounded RAM."""
    from copy import deepcopy

    model_state, optimizer_state, random_state = model.state_dict(), optimizer.state_dict(), rng_state()
    size = sum(value.numel() * value.element_size()
               for _, value in named_tree((model_state, optimizer_state, random_state), "resume")
               if value is not None)
    if size > max_bytes:
        raise MemoryError("In-memory resume state exceeds the explicit oracle byte budget")
    snapshot = deepcopy((model_state, optimizer_state, random_state))
    def perturb(value):
        if isinstance(value, torch.Tensor):
            value.zero_()
            return value
        if isinstance(value, dict):
            return {key: perturb(item) for key, item in value.items()}
        if isinstance(value, list):
            return [perturb(item) for item in value]
        if isinstance(value, tuple):
            return tuple(perturb(item) for item in value)
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 7
        if isinstance(value, float):
            return value + .5
        return value

    with torch.no_grad():
        for value in model_state.values():
            value.zero_()
        for parameter, state in optimizer.state.items():
            optimizer.state[parameter] = perturb(state)
    for group in optimizer.param_groups:
        group["lr"] = 0.
    torch.rand(1)
    random.random()
    model.load_state_dict(snapshot[0], strict=True)
    optimizer.load_state_dict(snapshot[1])
    restore_rng(snapshot[2])
    expected = dict(named_tree(snapshot, "resume"))
    actual = dict(named_tree((model.state_dict(), optimizer.state_dict(), rng_state()), "resume"))
    if expected.keys() != actual.keys():
        raise AssertionError("In-memory resume changed the state structure")
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
    return {"passed": True, "snapshot_bytes": size, "byte_limit": max_bytes,
            "storage": "bounded_cpu_RAM", "files_written": 0}


def assert_dtype_edge_contracts(report):
    """Inventory completeness and declared storage contracts are separate gates."""
    if not report.get("complete"):
        raise RuntimeError("Eager dtype inventory is incomplete")
    validation = report.get("storage_policy_validation", {})
    if validation.get("valid") is not True:
        failures = [item for item in validation.get("contracts", ()) if not item.get("valid")]
        raise RuntimeError(f"Eager declared activation dtype contract failed: {failures[:16]}")


def eager_dtype_diagnostic(model, images, compression, *, bf16_policy="torch_amp", managed=True,
                           device="cuda:0", max_bytes=512 * 2**20):
    """Separate cloned eager forward; no hooks on compiler/timing paths."""
    from copy import deepcopy
    import deepopt
    from deepopt._bf16_policy import resolve_bf16_policy
    from deepopt.opt.precision.diagnostics import DTypeEdgeRecorder

    policy = resolve_bf16_policy(bf16_policy)
    size = sum(value.numel() * value.element_size() for value in (*model.parameters(), *model.buffers()))
    if size > max_bytes:
        raise MemoryError("Eager dtype diagnostic clone exceeds bounded CPU state budget")
    with preserve_rng():
        diagnostic = deepcopy(model).to(device)
        recorder = DTypeEdgeRecorder(diagnostic)
        selected_device = torch.device(device)
        forward_context = (deepopt.autocast(selected_device.type, dtype=torch.bfloat16, fp8=False,
            activation_compression=compression, bf16_policy=policy, config={}) if managed else
            torch.autocast(selected_device.type, dtype=torch.bfloat16, enabled=selected_device.type == "cuda"))
        with torch.no_grad(), forward_context, recorder:
            outputs = diagnostic(images.to(device))
        del outputs, diagnostic
        gc.collect()
    report = recorder.report()
    report.update(bf16_policy=policy.as_dict(), cloned_model_state_bytes=size,
                  caller_rng_restored=True, caller_model_unmodified=True)
    return report


def emit_update(model, optimizer, images, beta, compression, sink, *, bf16_policy="torch_amp", managed=True,
                grad_clip_norm=None, progress=None, diagnostic=False):
    """Consume one complete graph, including backward after the AMP scope exits."""
    import deepopt
    from benchmarks.analysis.measurement import SavedStorageLifetimes
    from torch.nn import functional as F

    from deepopt._bf16_policy import resolve_bf16_policy, bf16_policy_scope

    policy = resolve_bf16_policy(bf16_policy)
    logical_model = getattr(model, "_orig_mod", model)
    started = time.perf_counter()
    _notify_progress(progress, "input_prepare")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("VAE qualification requires FP32 master parameters")
    optimizer.zero_grad(set_to_none=True)
    x = images.detach().clone().requires_grad_()
    tracker = SavedStorageLifetimes((*model.parameters(), *model.buffers()))
    _notify_progress(progress, "forward")
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        forward_context = (deepopt.autocast(x.device.type, dtype=torch.bfloat16, fp8=False,
                             activation_compression=compression, bf16_policy=policy, config={}) if managed else
                           torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"))
        with forward_context:
            reconstruction, mu, logvar = model(x)
            amp_enabled = torch.is_autocast_enabled(x.device.type)
            amp_dtype = torch.get_autocast_dtype(x.device.type)
        with bf16_policy_scope("torch_amp", x.device.type), torch.autocast(x.device.type, enabled=False):
            _notify_progress(progress, "loss")
            l1 = F.l1_loss(reconstruction.float(), x.float())
            kl = .5 * (mu.float().square() + logvar.float().exp() - 1 - logvar.float()).sum((1, 2, 3)).mean()
            loss = l1 + beta * kl
    dtypes = {"reconstruction": str(reconstruction.dtype), "mu": str(mu.dtype), "logvar": str(logvar.dtype),
              "loss": str(loss.dtype), "l1": str(l1.dtype), "kl": str(kl.dtype)}
    if any(value.dtype != torch.float32 for value in (loss, l1, kl)):
        raise AssertionError("VAE objective components must accumulate in FP32")
    if x.device.type == "cuda":
        assert_bf16_forward_domains(x.dtype, dtypes, amp_enabled, amp_dtype)
    _notify_progress(progress, "output_comparison")
    for name, value in (("reconstruction", reconstruction), ("mu", mu), ("logvar", logvar),
                        ("loss", loss), ("l1", l1), ("kl", kl)):
        sink(f"output/{name}", value)
    for name, value in named_tree(policy.as_dict(), "recipe/bf16_policy"):
        sink(name, value)
    sink("recipe/beta", torch.tensor(beta, dtype=torch.float64))
    _notify_progress(progress, "backward")
    loss.backward()
    if x.device.type == "cuda":
        if x.grad is None or x.grad.dtype != torch.float32:
            raise AssertionError("CUDA VAE input gradient must be FP32")
        if any(parameter.requires_grad and (parameter.grad is None or parameter.grad.dtype != torch.float32)
               for parameter in model.parameters()):
            raise AssertionError("CUDA VAE master-parameter gradients must be present and FP32")
    _notify_progress(progress, "gradient_comparison")
    sink("gradient/input", x.grad)
    for name, parameter in logical_model.named_parameters():
        sink(f"gradient/{name}", parameter.grad)
    # Small gradient fields may still be awaiting a batched ACK. Consume them
    # before allowing an optimizer to step past a queued numerical failure.
    flush = getattr(sink, "flush", None)
    if callable(flush):
        flush()
    _notify_progress(progress, "gradient_comparison", all_gradients_compared=True)
    if diagnostic:
        comparer = getattr(getattr(sink, "sink", sink), "__self__", None)
        if isinstance(comparer, TensorOracle):
            comparer.raise_if_failed()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    # Retain only one parameter's initial value while emitting its true delta.
    # Optimizers update all parameters together, so initial values live on CPU.
    before = {name: parameter.detach().cpu().clone() for name, parameter in logical_model.named_parameters()}
    _notify_progress(progress, "optimizer")
    optimizer.step()
    _notify_progress(progress, "state_comparison")
    optimizer_state = optimizer.state_dict()
    state = rng_state()
    state["python"] = torch.tensor(list(repr(state["python"]).encode()), dtype=torch.uint8)
    preflight = getattr(sink, "preflight_reference", None)
    if callable(preflight):
        preflight(post_update_reference_fields(logical_model, optimizer_state, state))
    for name, parameter in logical_model.named_parameters():
        sink(f"parameter/{name}", parameter)
        sink(f"delta/{name}", parameter.detach().cpu() - before.pop(name))
    for name, buffer in logical_model.named_buffers():
        sink(f"buffer/{name}", buffer)
    for name, value in named_tree(optimizer_state, "optimizer"):
        sink(name, value)
        if isinstance(value, torch.Tensor):
            sink(f"optimizer_policy/device/{name}", torch.tensor(list(value.device.type.encode()), dtype=torch.uint8))
    for name, value in named_tree(state, "rng"):
        sink(name, value)
    _notify_progress(progress, "release")
    optimizer.zero_grad(set_to_none=True)
    del reconstruction, mu, logvar, loss, l1, kl, x
    gc.collect()
    live = tracker.snapshot()["live_nonpersistent_bytes"]
    if live:
        raise AssertionError(f"VAE update retained {live} nonpersistent saved bytes")
    _notify_progress(progress, "complete")
    return {"saved_nonpersistent_bytes_after_update": live, "effective_dtypes": dtypes,
            "bf16_policy": policy.as_dict(),
            "diagnostic_only": diagnostic, "qualification_eligible": not diagnostic,
            "autocast_enabled": amp_enabled, "autocast_dtype": str(amp_dtype),
            "oracle_inclusive_update_seconds": time.perf_counter() - started}


def paired_trajectory(eager, native, eager_optimizer, native_optimizer, images, betas, compression,
                      *, device="cuda:0", max_bytes=512 * 2**20, bf16_policy="torch_amp", managed=True,
                      grad_clip_norm=None, progress=None, diagnostic=False, evidence_callback=None, route_context=None):
    """Sequential device residency, paired stochastic draws, full named parity."""
    from deepopt._bf16_policy import resolve_bf16_policy

    policy = resolve_bf16_policy(bf16_policy)
    if compression.get("enabled") and (compression.get("group_size") != 1024
                                         or not compression.get("norm_xhat_recenter")):
        raise ValueError("VAE qualification requires group_size=1024 and recentering")
    if type(eager_optimizer) is not type(native_optimizer):
        raise ValueError("Paired optimizer classes differ")
    if type(diagnostic) is not bool:
        raise TypeError("diagnostic must be a boolean")
    slot_label = "native_is_legacy_slot_name;ownership_requires_separate_compiler_evidence"

    def descriptor(model):
        logical = getattr(model, "_orig_mod", model)
        compilation = getattr(logical, "_deepopt_compile_config", {})
        memory = getattr(logical, "_deepopt_activation_memory_plan", None)
        return {"model_type": f"{type(model).__module__}:{type(model).__qualname__}",
                "compile_backend": compilation.get("backend") if isinstance(compilation, dict) else None,
                "torch_compiled_wrapper": logical is not model,
                "activation_memory": memory.as_dict() if memory is not None else None}

    slots = {"eager": {"role": "reference", **descriptor(eager)},
             "native": {"role": "candidate", **descriptor(native)}}
    context = {"component": "paired_trajectory", "phase": "initial_state_comparison", "route": None, "update": None}
    evidence, row, oracle, initial_report, resume = [], None, None, None, None

    def partial(oracle_report):
        return _metadata_copy({"schema_version": 1, "passed": False, "partial": True,
                "diagnostic_only": diagnostic, "qualification_eligible": False,
                "comparison_scope": "outputs_and_all_gradients_before_failed_candidate_optimizer" if diagnostic else "fail_fast",
                "bf16_policy": policy.as_dict(), "candidate_slot_label": slot_label,
                "route_slots": slots, "context": context, "updates": evidence,
                "partial_update": row, "initial_state_oracle": initial_report,
                "oracle_report": oracle_report, "checkpoint_free": True,
                "tensor_payloads_in_evidence": False})

    def publish(report):
        if evidence_callback is not None:
            try:
                evidence_callback(_metadata_copy(report))
            except Exception:
                # Diagnostics must never change a numerical admission result.
                pass

    try:
        _notify_progress(progress, "initial_state_comparison", route=None, update=None)
        oracle = TensorOracle(max_bytes, diagnostic=diagnostic, context=context)
        for name, value in named_tree(getattr(eager, "_orig_mod", eager).state_dict(), "model"):
            oracle.record(name, value)
        context.update(route="native", role="candidate")
        oracle.context = _metadata_copy(context)
        for name, value in named_tree(getattr(native, "_orig_mod", native).state_dict(), "model"):
            oracle.compare(name, value)
        for name, value in named_tree(eager_optimizer.state_dict(), "initial/optimizer"):
            oracle.record(name, value)
        for name, value in named_tree(native_optimizer.state_dict(), "initial/optimizer"):
            oracle.compare(name, value)
        initial_report = oracle.finish()
        for index, beta in enumerate(betas):
            state = rng_state()
            oracle = TensorOracle(max_bytes, diagnostic=diagnostic)
            row = {"update": index + 1, "beta": beta}
            for label, model, optimizer, sink in (("eager", eager, eager_optimizer, oracle.record),
                                                  ("native", native, native_optimizer, oracle.compare)):
                route_state = {"completed": False, "backward_completed": False,
                               "gradient_comparisons_complete": False,
                               "optimizer_attempted": False, "optimizer_completed": False}
                row[label] = route_state

                def route_progress(event):
                    context.update(phase=event["phase"], route=label, update=index + 1,
                                   role=slots[label]["role"])
                    oracle.context = _metadata_copy(context)
                    if event["phase"] == "gradient_comparison":
                        route_state["backward_completed"] = True
                        if event.get("all_gradients_compared"):
                            route_state["gradient_comparisons_complete"] = True
                    elif event["phase"] == "optimizer":
                        route_state["optimizer_attempted"] = True
                    elif event["phase"] == "state_comparison":
                        route_state["optimizer_completed"] = True
                    elif event["phase"] == "complete":
                        route_state["completed"] = True
                    _notify_progress(progress, event["phase"], route=label, update=index + 1)

                restore_rng(state)
                route_progress({"phase": "device_transfer"})
                model.to(device)
                if {id(parameter) for parameter in model.parameters()} != {
                        id(parameter) for group in optimizer.param_groups for parameter in group["params"]}:
                    raise AssertionError("Device transfer changed optimizer parameter ownership")
                route_progress({"phase": "optimizer_transfer"})
                move_optimizer(optimizer, device)
                route_progress({"phase": "route_context"})
                observing = route_context(label, model) if route_context is not None else nullcontext()
                with _route_observation(observing), ChunkTransport(sink, {"update": index + 1, "route": label,
                                                       "bf16_policy": policy.as_dict()}) as transport:
                    update_report = emit_update(model, optimizer, images.to(device), beta, compression, transport,
                        bf16_policy=policy, managed=managed, grad_clip_norm=grad_clip_norm,
                        progress=route_progress, diagnostic=diagnostic)
                route_state.update(update_report)
                route_progress({"phase": "device_offload"})
                model.cpu()
                move_optimizer(optimizer, "cpu")
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                gc.collect()
            context.update(phase="trajectory_comparison", route=None, update=index + 1, role=None)
            oracle.context = _metadata_copy(context)
            _notify_progress(progress, "trajectory_comparison", route=None, update=index + 1)
            row["oracle"] = oracle.finish()
            evidence.append(row)
            row = None
            if len(betas) > 1 and index + 1 == len(betas) // 2:
                resume = resume_roundtrip(native, native_optimizer, max_bytes)
                resume["after_update"] = index + 1
        report = {"passed": bool(evidence) and not diagnostic,
                "numerical_comparisons_passed": bool(evidence), "diagnostic_only": diagnostic,
                "qualification_eligible": not diagnostic, "bf16_policy": policy.as_dict(),
                "updates": evidence, "initial_state_oracle": initial_report, "checkpoint_free": True,
                "comparison": "all_named_elements", "transport": "numerical_ipc_named_chunks",
                "candidate_slot_label": slot_label, "route_slots": slots, "resume": resume,
                "optimizer": f"{type(eager_optimizer).__module__}:{type(eager_optimizer).__qualname__}"}
        publish(report)
        return report
    except BaseException as error:
        # Preserve this exact exception and traceback, including callback errors
        # returned over IPC. No collector or observer may replace the cause.
        try:
            oracle_report = getattr(error, "oracle_report", None)
            if oracle_report is None and oracle is not None:
                oracle_report = oracle.snapshot()
            report = partial(oracle_report)
            if hasattr(error, "diagnostic_context_error"):
                report["diagnostic_context_error"] = _metadata_copy(error.diagnostic_context_error)
            error.qualification_evidence = report
            if oracle_report is not None:
                error.oracle_report = _metadata_copy(oracle_report)
            publish(report)
        except BaseException as diagnostic_error:
            try:
                error.qualification_evidence = partial({"passed": False, "partial": True,
                    "diagnostic_collection_error": type(diagnostic_error).__name__,
                    "tensor_payloads_in_evidence": False})
            except BaseException:
                pass
        raise
