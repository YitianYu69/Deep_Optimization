"""Bounded, invocation-local numerical evidence for the model matrix.

Only eager oracles from one trial retain tensors. Candidate receipts retain
scalar comparison reports; no tensor serialization or filesystem I/O occurs.
"""

from copy import deepcopy
import json
import threading
import time
import uuid


_FIXED_POLICY = {"rtol": 0.02, "atol": 0.002, "max_relative_l2": 0.1}
_CONTEXT_FIELDS = ("suite_run_id", "trial", "strategy", "job_nonce")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _failure(reason, detail=None):
    result = {"passed": False, "finite": False, "reason": reason, "tensors": {},
              "full_tensor_finite": False, "same_output_and_gradient_shapes": False,
              "same_sampling_policy": False}
    if detail is not None:
        result["error"] = str(detail)
    return result


class ModelNumericalComparisons:
    """Accept only the active job and fail closed on absent numerical evidence.

    ``validate_row`` returns an evidence result, and ``compare_rows`` returns an
    errors-style result. ``end_job`` closes admission; successful rows still
    require individual receipts. Call ``clear_trial`` between trials. Instances
    belong to one suite invocation and must not be reused to resume a run.
    """

    def __init__(self, max_retained_bytes=512 * 2**20, receipt_timeout=0.5):
        if isinstance(max_retained_bytes, bool) or not isinstance(max_retained_bytes, int) or max_retained_bytes <= 0:
            raise ValueError("Numerical retention limit must be a positive integer")
        if not 0 <= receipt_timeout < float("inf"):
            raise ValueError("Receipt timeout must be finite and nonnegative")
        self.max_retained_bytes = max_retained_bytes
        self.receipt_timeout = receipt_timeout
        self._condition = threading.Condition(threading.RLock())
        self._active = None
        self._suite_run_id = None
        self._trial = None
        self._error = None
        self._receipts = {}
        self._eager = {}
        self._storages = {}
        self._retained_bytes = 0

    @property
    def retained_bytes(self):
        with self._condition:
            return self._retained_bytes

    def start_job(self, job, suite_run_id):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"Numerical broker failed: {self._error}")
            if self._active is not None:
                raise RuntimeError("A numerical job is already active")
            trial, strategy = job.get("trial"), job.get("strategy")
            if not isinstance(suite_run_id, str) or not suite_run_id:
                raise ValueError("A fresh suite identity is required")
            if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0 or not isinstance(strategy, str) or not strategy:
                raise ValueError("Job trial/strategy identity is invalid")
            if self._suite_run_id not in (None, suite_run_id):
                raise ValueError("A numerical store belongs to one suite invocation")
            if self._trial not in (None, trial):
                raise ValueError("Release the previous trial's eager tensors before starting another trial")
            tasks = job.get("expected_tasks")
            if not isinstance(tasks, list) or not tasks or any(not isinstance(task, dict) for task in tasks):
                raise ValueError("A numerical job must declare its exact expected tasks")
            expected = {_canonical(task): deepcopy(task) for task in tasks}
            ids = [task.get("id") for task in tasks]
            if (len(expected) != len(tasks) or len(set(ids)) != len(ids)
                    or any(not isinstance(value, str) or not value for value in ids)):
                raise ValueError("Expected numerical tasks must have unique nonempty identities")
            context = {"suite_run_id": suite_run_id, "trial": trial, "strategy": strategy, "job_nonce": uuid.uuid4().hex}
            self._suite_run_id, self._trial = suite_run_id, trial
            self._active = {"context": context, "expected": expected, "received": set()}
            return deepcopy(context)

    def end_job(self):
        with self._condition:
            self._active = None
            self._condition.notify_all()

    def fail(self, error):
        with self._condition:
            if self._error is None:
                self._error = f"{type(error).__name__}: {error}"
            self._condition.notify_all()

    def clear_trial(self):
        with self._condition:
            if self._active is not None:
                raise RuntimeError("Cannot release eager oracles while a numerical job is active")
            self._eager.clear()
            self._storages.clear()
            self._retained_bytes = 0
            self._trial = None

    @staticmethod
    def _scope(identity):
        task = identity["task"]
        return identity["trial"], _canonical(task["workload"]), task["case"]

    @staticmethod
    def _key(identity):
        return identity["worker_run_id"], _canonical(identity["task"])

    @staticmethod
    def _validate_payload(tensors, metadata, descriptor):
        import torch

        if (not isinstance(tensors, dict) or not tensors or any(not isinstance(name, str) or not name for name in tensors)
                or any(value is not None and not isinstance(value, torch.Tensor) for value in tensors.values())
                or not any(value is not None for value in tensors.values())):
            raise ValueError("Numerical payload must contain named tensors or unused gradients")
        if not isinstance(metadata, dict) or metadata.get("full_tensor_finite") is not True:
            raise ValueError("Numerical evidence must report full-tensor finiteness")
        cap = metadata.get("max_sampled_elements_per_float_tensor")
        if (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0
                or not isinstance(metadata.get("sampling"), str) or not metadata["sampling"]):
            raise ValueError("Numerical evidence must declare its sampling policy and cap")
        shapes = metadata.get("original_shapes")
        names = {name for name, value in tensors.items() if value is not None}
        if (not isinstance(shapes, dict) or shapes.keys() != names
                or any(not isinstance(shape, list) or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                       for dim in shape) for shape in shapes.values())):
            raise ValueError("Original numerical tensor shapes are incomplete or malformed")
        if not isinstance(descriptor, dict) or not descriptor:
            raise ValueError("Numerical transport descriptor is missing")
        # These copies must remain small JSON evidence, never tensor payloads.
        _canonical(metadata)
        _canonical(descriptor)
        storages = {}
        for name, value in tensors.items():
            if value is None:
                if not name.startswith(("input_grad.", "parameter_grad.")):
                    raise ValueError("Only unused gradients may be represented by None")
                continue
            if value.device.type != "cpu" or value.layout != torch.strided or value.requires_grad or value.is_complex():
                raise ValueError("Numerical transport must contain detached dense real CPU tensors")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Numerical tensor payload is nonfinite")
            numel = 1
            for dim in shapes[name]:
                numel *= dim
            sampled = value.is_floating_point() and cap and numel > cap
            expected_shape = [cap] if sampled else shapes[name]
            if list(value.shape) != expected_shape:
                raise ValueError(f"Sampled tensor shape disagrees with its declared original shape: {name}")
            storage = value.untyped_storage()
            storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        return storages

    def accept(self, payload):
        """Consume a complete IPC payload, retaining tensors only for eager."""
        try:
            with self._condition:
                if self._error is not None:
                    raise RuntimeError(f"Numerical broker failed: {self._error}")
                if self._active is None or not isinstance(payload, dict):
                    raise ValueError("No numerical job admits this payload")
                identity = payload.get("identity")
                if not isinstance(identity, dict) or any(identity.get(key) != self._active["context"][key] for key in _CONTEXT_FIELDS):
                    raise ValueError("Numerical payload belongs to another job or suite")
                worker_id = identity.get("worker_run_id")
                task = identity.get("task")
                if not isinstance(worker_id, str) or not worker_id or not isinstance(task, dict):
                    raise ValueError("Numerical payload lacks its exact task/worker identity")
                task_key = _canonical(task)
                if task_key not in self._active["expected"]:
                    raise ValueError("Unexpected numerical task")
                key = self._key(identity)
                if task_key in self._active["received"] or key in self._receipts:
                    raise ValueError("Duplicate numerical task receipt")
                tensors, metadata, descriptor = (payload.get(name) for name in ("tensors", "metadata", "descriptor"))
                storages = self._validate_payload(tensors, metadata, descriptor)
                if sum(storages.values()) > self.max_retained_bytes:
                    raise MemoryError("A numerical payload exceeds the in-memory tensor limit")
                if descriptor.get("identity") != identity or descriptor.get("metadata") != metadata:
                    raise ValueError("Numerical descriptor differs from its payload identity or metadata")
                scope = self._scope(identity)
                receipt = {"identity": deepcopy(identity), "metadata": deepcopy(metadata), "descriptor": deepcopy(descriptor)}
                if identity["strategy"] == "eager":
                    if scope in self._eager:
                        raise ValueError("Duplicate eager oracle for a workload/case/trial")
                    extra = sum(size for storage, size in storages.items() if storage not in self._storages)
                    if self._retained_bytes + extra > self.max_retained_bytes:
                        raise MemoryError("Eager numerical oracles exceed the aggregate in-memory retention limit")
                    self._eager[scope] = {"key": key, "tensors": tensors, "metadata": receipt["metadata"]}
                    self._storages.update(storages)
                    self._retained_bytes += extra
                else:
                    reference = self._eager.get(scope)
                    receipt["reference_key"] = reference["key"] if reference is not None else None
                    receipt["comparison"] = (self._compare(tensors, metadata, reference["tensors"], reference["metadata"], _FIXED_POLICY)
                                             if reference is not None else _failure("missing_eager_oracle"))
                self._receipts[key] = receipt
                self._active["received"].add(task_key)
                self._condition.notify_all()
        except Exception as error:
            self.fail(error)
            raise

    @staticmethod
    def _compare(actual, metadata, expected, reference_metadata, policy):
        from analysis.measurement import errors

        shapes = metadata["original_shapes"] == reference_metadata["original_shapes"]
        sampling = all(metadata.get(key) == reference_metadata.get(key)
                       for key in ("sampling", "max_sampled_elements_per_float_tensor", "numerical_probe"))
        result = errors(actual, expected, **policy)
        result["full_tensor_finite"] = metadata.get("full_tensor_finite") is True and reference_metadata.get("full_tensor_finite") is True
        result["same_output_and_gradient_shapes"] = shapes
        result["same_sampling_policy"] = sampling
        result["finite"] &= result["full_tensor_finite"]
        result["passed"] &= result["finite"] and shapes and sampling
        return result

    def _receipt(self, row):
        if not isinstance(row, dict) or not isinstance(row.get("task"), dict):
            return None, _failure("invalid_numerical_row")
        if (row.get("snapshot") is not None or row.get("snapshot_file") is not None
                or row.get("measurement_policy", {}).get("numerical_snapshot_mode") != "memory"):
            return None, _failure("numerical_disk_artifact_or_wrong_transport")
        worker_id = row.get("numerical_worker_run_id")
        if not isinstance(worker_id, str) or not worker_id:
            return None, _failure("missing_numerical_worker_identity")
        key = worker_id, _canonical(row["task"])
        deadline = time.monotonic() + self.receipt_timeout
        while key not in self._receipts and self._error is None and self._active is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._condition.wait(remaining)
        if self._error is not None:
            return None, _failure("numerical_broker_failed", self._error)
        receipt = self._receipts.get(key)
        if receipt is None:
            return None, _failure("missing_numerical_receipt")
        if (row.get("comparison_input") != receipt["descriptor"]
                or row.get("numerical_snapshot") != receipt["metadata"]
                or any(name in row and row[name] != receipt["identity"][name]
                       for name in ("suite_run_id", "trial", "strategy"))):
            return None, _failure("mismatched_numerical_receipt")
        return receipt, None

    def validate_row(self, row):
        with self._condition:
            try:
                receipt, error = self._receipt(row)
            except Exception as invalid:
                return _failure("invalid_numerical_row", invalid)
            return error if error is not None else {"passed": True, "worker_run_id": receipt["identity"]["worker_run_id"]}

    def compare_rows(self, row, reference, **errors_policy):
        with self._condition:
            try:
                actual_receipt, error = self._receipt(row)
                if error is not None:
                    return error
                expected_receipt, error = self._receipt(reference)
                if error is not None:
                    return error
                policy = ({name: value for name, value in errors_policy.items()
                           if value is not None or name in ("rtol", "atol")}
                          if errors_policy else dict(_FIXED_POLICY))
                if actual_receipt["identity"]["strategy"] != "eager":
                    if policy != _FIXED_POLICY:
                        return _failure("candidate_comparison_policy_mismatch")
                    if actual_receipt.get("reference_key") != self._key(expected_receipt["identity"]):
                        return _failure("candidate_eager_receipt_mismatch")
                    return deepcopy(actual_receipt["comparison"])
                actual = self._eager.get(self._scope(actual_receipt["identity"]))
                expected = self._eager.get(self._scope(expected_receipt["identity"]))
                if (actual is None or expected is None or actual["key"] != self._key(actual_receipt["identity"])
                        or expected["key"] != self._key(expected_receipt["identity"])):
                    return _failure("eager_oracle_released")
                return self._compare(actual["tensors"], actual["metadata"], expected["tensors"], expected["metadata"], policy)
            except Exception as invalid:
                return _failure("invalid_numerical_comparison", invalid)
