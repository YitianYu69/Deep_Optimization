"""Bounded numerical localization for VAE diagnostics, outside timed updates.

CPU snapshots live only in this process. Reports contain metadata, hashes and
scalar errors; tensor checkpoints are never written. Matched module boundaries
are not a proof of the first differing kernel in the complete graph.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


DEFAULT_MODULES = ("encoder.0", "decoder.0", "decoder.4", "decoder.8")
_VJP_OPERATIONS = frozenset(("resize.nearest2d_metadata_vjp", "convolution.native_dx_db",
    "normalization.vjp", "normalization.layer_norm_vjp", "activation.coefficient_vjp",
    "bf16.gemm_dx", "bf16.gemm_dx_packed_derivative", "bf16.gemm_dx_native_derivative"))


def decoder_geometry_metadata():
    return [{"module": f"decoder.{index * 4}", "input_shape": [128, channels, size // 2, size // 2],
             "output_shape": [128, channels, size, size], "scales": [2., 2.],
             "recompute_scale_factor": None, "dtype": "torch.bfloat16",
             "execution_observed": False}
            for index, (channels, size) in enumerate(((512, 56), (352, 112), (192, 224)))]


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        return None
    return next((tensor for item in values if (tensor := _first_tensor(item)) is not None), None)


def _metadata(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return str(value)
    raise TypeError("Diagnostic metadata contains a non-metadata value")


def _description(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "stride": list(tensor.stride()), "elements": tensor.numel(),
            "logical_bytes": tensor.numel() * tensor.element_size()}


def _host_chunks(tensor, chunk_elements):
    """Logical row-major chunks without flattening a whole strided activation."""
    value = tensor.detach()
    for start in range(0, value.numel(), chunk_elements):
        stop = min(start + chunk_elements, value.numel())
        if value.is_contiguous():
            part = value.view(-1)[start:stop]
        else:
            position = torch.arange(start, stop, device=value.device, dtype=torch.int64)
            indices = []
            for size in reversed(value.shape):
                indices.append(position.remainder(size))
                position = position.div(size, rounding_mode="floor")
            part = value[tuple(reversed(indices))]
        yield start, part.to("cpu", copy=True).contiguous()


def _storage_budget(dtype):
    # Keep qualification.TensorOracle's declared storage-domain budgets.
    if not dtype.is_floating_point:
        return {"rtol": 0., "atol": 0., "relative_l2_limit": 0., "exact": True}
    epsilon = torch.finfo(dtype).eps
    return {"rtol": (2 if dtype == torch.bfloat16 else 8) * epsilon,
            "atol": 2 * min(epsilon, torch.finfo(torch.float32).eps),
            "relative_l2_limit": epsilon if dtype == torch.bfloat16 else 8 * epsilon,
            "exact": False}


def _scan(tensor, chunk_elements, *, snapshot=None, expected=None):
    expected = expected.detach() if expected is not None else None
    digest = hashlib.sha256()
    total = squared = error_squared = reference_squared = maximum_error = 0.
    minimum = maximum = None
    finite_count = 0
    elementwise = equal = True
    budget = _storage_budget(tensor.dtype)
    reference_chunks = (iter(_host_chunks(expected, chunk_elements)) if expected is not None
                        and (expected.device.type != "cpu" or not expected.is_contiguous()) else None)
    for start, part in _host_chunks(tensor, chunk_elements):
        digest.update(part.view(torch.uint8).numpy().tobytes())
        if snapshot is not None:
            snapshot.view(-1)[start:start + part.numel()].copy_(part)
        numbers = part.double()
        finite = torch.isfinite(numbers)
        count = int(finite.sum())
        finite_count += count
        if count:
            valid = numbers[finite]
            total += float(valid.sum())
            squared += float(valid.square().sum())
            low, high = float(valid.min()), float(valid.max())
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
        if expected is not None:
            reference = (next(reference_chunks)[1] if reference_chunks is not None else
                         expected.view(-1)[start:start + part.numel()])
            equal = equal and torch.equal(part.view(torch.uint8), reference.contiguous().view(torch.uint8))
            elementwise = elementwise and bool(torch.isclose(part, reference, rtol=budget["rtol"],
                                                              atol=budget["atol"], equal_nan=False).all())
            difference = numbers - reference.double()
            if bool(torch.isfinite(difference).all()):
                maximum_error = max(maximum_error, float(difference.abs().max()))
                error_squared += float(difference.square().sum())
                reference_squared += float(reference.double().square().sum())
            else:
                elementwise = False
    result = {"logical_sha256": digest.hexdigest(), "finite_elements": finite_count,
              "nonfinite_elements": tensor.numel() - finite_count, "sum": total,
              "l2_norm": math.sqrt(squared), "minimum": minimum, "maximum": maximum}
    if expected is not None:
        relative = math.sqrt(error_squared / max(reference_squared, 1e-300))
        result.update(max_abs=maximum_error, relative_l2=relative, bitwise_equal=equal, **budget,
                      passed=elementwise and finite_count == tensor.numel() and relative <= budget["relative_l2_limit"])
    return result


class BoundaryTrace:
    """Compare stable module outputs/cotangents with native wrapper boundaries.

    Use an unlinked instance for reference collection and a second instance
    linked with ``reference=...`` for candidate comparison. Both support eager
    and native contexts, allowing same-route repeat controls. Native input
    adjoints are reported separately when no eager boundary represents them.
    """

    def __init__(self, *, reference=None, modules=DEFAULT_MODULES, max_bytes=128 * 2**20,
                 chunk_elements=262144, max_boundaries=2048, artifact_dir=None,
                 max_artifact_bytes=64 * 2**20, max_artifact_files=256):
        for name, value in (("max_bytes", max_bytes), ("chunk_elements", chunk_elements),
                            ("max_boundaries", max_boundaries), ("max_artifact_bytes", max_artifact_bytes),
                            ("max_artifact_files", max_artifact_files)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.reference = reference
        self.modules = tuple(modules)
        if any(not isinstance(name, str) or not name for name in self.modules):
            raise ValueError("Boundary modules must be nonempty exact module paths")
        self.max_bytes, self.chunk_elements, self.max_boundaries = max_bytes, chunk_elements, max_boundaries
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir is not None else None
        self.max_artifact_bytes, self.max_artifact_files = max_artifact_bytes, max_artifact_files
        self._snapshots, self._rows, self._counts, self._seen = {}, [], {}, set()
        self._bytes, self._omitted, self._active = 0, 0, False
        self._errors, self._artifacts, self._artifact_omissions = [], [], []
        self._artifact_keys, self._artifact_bytes = set(), 0
        self._cuda = False

    def _selected(self, module):
        return module in self.modules

    def _observe(self, module, phase, role, tensor, *, source_node=None, operation=None,
                 occurrence=None, paired=True):
        if not self._selected(module) or not isinstance(tensor, torch.Tensor):
            return None
        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(tensor, FakeTensor) or tensor.device.type == "meta" or tensor.is_complex():
            self._errors.append({"module": module, "phase": phase, "reason": "requires_real_noncomplex_tensor"})
            return None
        if tensor.device.type == "cuda":
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Numerical boundary observation is outside CUDA Graph capture")
            self._cuda = True
        count_key = module, phase, role
        occurrence = self._counts.get(count_key, 0) + 1 if occurrence is None else occurrence
        self._counts[count_key] = max(occurrence, self._counts.get(count_key, 0))
        key = f"{phase}/{module}/{role}/{occurrence}"
        if len(self._rows) >= self.max_boundaries:
            self._omitted += 1
            return occurrence
        row = {"key": key, "module": module, "phase": phase, "role": role, "occurrence": occurrence,
               "source_node": source_node, "operation": operation, "paired": paired, **_description(tensor)}
        self._rows.append(row)
        if not paired:
            row["status"] = "unpaired_native_adjoint"
            if row["logical_bytes"] <= self.max_bytes:
                row.update(_scan(tensor, self.chunk_elements))
            else:
                row["summary_status"] = "unavailable_size_budget"
            return occurrence
        self._seen.add(key)
        if self.reference is None:
            if self._bytes + row["logical_bytes"] > self.max_bytes:
                row["status"] = "unavailable_reference_byte_budget"
                self._omitted += 1
                return occurrence
            snapshot = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu")
            row.update(_scan(tensor, self.chunk_elements, snapshot=snapshot), status="reference_recorded")
            self._snapshots[key] = snapshot
            self._bytes += row["logical_bytes"]
        else:
            expected = self.reference._snapshots.get(key)
            if expected is None:
                row["status"] = "unavailable_reference_boundary"
                self._omitted += 1
            elif tuple(expected.shape) != tuple(tensor.shape) or expected.dtype != tensor.dtype:
                row.update(status="compared", passed=False, reason="shape_or_dtype_differs",
                           expected_shape=list(expected.shape), expected_dtype=str(expected.dtype))
            else:
                row.update(_scan(tensor, self.chunk_elements, expected=expected), status="compared")
        return occurrence

    @contextmanager
    def eager(self, model):
        if self._active:
            raise RuntimeError("BoundaryTrace contexts are not reentrant")
        if any(getattr(module, "_compiled_call_impl", None) is not None for module in model.modules()):
            raise ValueError("Eager boundary hooks require a separate uncompiled model")
        handles = []
        self._active = True
        try:
            for name, module in model.named_modules():
                if not self._selected(name):
                    continue
                def after(owner, args, output, name=name):
                    tensor = _first_tensor(output)
                    if tensor is None:
                        return
                    occurrence = self._observe(name, "forward", "output", tensor)
                    if tensor.requires_grad and occurrence is not None:
                        def cotangent(gradient, name=name, occurrence=occurrence):
                            self._observe(name, "backward", "cotangent", gradient, occurrence=occurrence)
                        handles.append(tensor.register_hook(cotangent))
                handles.append(module.register_forward_hook(after))
            yield self
        except BaseException as error:
            self._errors.append({"phase": "eager_context", "exception": type(error).__name__})
            raise
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    @contextmanager
    def native(self):
        from deepopt.compiler.runtime import native_boundary_observation

        if self._active:
            raise RuntimeError("BoundaryTrace contexts are not reentrant")
        self._active = True
        try:
            with native_boundary_observation(self):
                yield self
        except BaseException as error:
            self._errors.append({"phase": "native_context", "exception": type(error).__name__})
            raise
        finally:
            self._active = False

    def _save_artifact(self, artifact, phase):
        if self.artifact_dir is None:
            return
        source = getattr(artifact, "source", None)
        if not isinstance(source, str):
            self._artifact_omissions.append({"phase": phase, "reason": "source_unavailable"})
            return
        digest = hashlib.sha256(source.encode()).hexdigest()
        metadata = None
        try:
            metadata = json.dumps(_metadata({
                "phase": phase, "boundary_metadata": getattr(artifact, "boundary_metadata", ()),
                "final_schedule": artifact.final_schedule, "logical_lifetimes": artifact.logical_lifetimes,
                "tensor_files_saved": 0}), indent=2, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            self._artifact_omissions.append({"phase": phase, "reason": "metadata_serialization_failed",
                                             "exception": type(error).__name__})
        metadata_digest = hashlib.sha256((metadata or "").encode()).hexdigest()
        identity = phase, digest, metadata_digest
        if identity in self._artifact_keys:
            return
        self._artifact_keys.add(identity)
        prefix = f"native_{phase}_{digest[:12]}_{metadata_digest[:12]}"
        files = [(prefix + ".py", source)]
        if metadata is not None:
            files.append((prefix + ".json", metadata))
        for family in ("triton_artifacts", "tilelang_artifacts"):
            for index, kernel in enumerate(getattr(artifact, family, ())):
                text = getattr(kernel, "source", None)
                if isinstance(text, str):
                    files.append((f"{prefix}_{family}_{index}.py", text))
        for name, text in files:
            payload = text.encode()
            if (len(self._artifacts) >= self.max_artifact_files
                    or self._artifact_bytes + len(payload) > self.max_artifact_bytes):
                self._artifact_omissions.append({"name": name, "reason": "artifact_budget", "bytes": len(payload)})
                continue
            try:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                path = self.artifact_dir / name
                if path.exists():
                    if path.read_bytes() != payload:
                        raise FileExistsError("Diagnostic artifact path has different existing contents")
                else:
                    with path.open("xb") as stream:
                        stream.write(payload)
                self._artifacts.append({"path": str(path), "bytes": len(payload),
                                        "sha256": hashlib.sha256(payload).hexdigest()})
                self._artifact_bytes += len(payload)
            except OSError as error:
                self._artifact_omissions.append({"name": name, "reason": "artifact_write_failed",
                                                 "exception": type(error).__name__})

    def run_artifact(self, artifact, *args, phase):
        if phase not in ("forward", "backward"):
            raise ValueError("Native boundary observation phase must be forward or backward")
        self._save_artifact(artifact, phase)
        rows = {row["index"]: row for row in getattr(artifact, "boundary_metadata", ())}
        trace = self
        class Observer:
            def boundary(self, index, values):
                for node in rows.get(index, {}).get("nodes", ()):
                    module, operation = node.get("module"), node.get("operation", "")
                    if not trace._selected(module):
                        continue
                    common = {"source_node": node.get("source_node"), "operation": operation}
                    outputs = node.get("outputs", ())
                    if phase == "forward" and operation.startswith("autodiff_") and outputs:
                        trace._observe(module, phase, "output", values.get(outputs[0]), **common)
                    elif phase == "backward" and operation in _VJP_OPERATIONS:
                        inputs = node.get("inputs", ())
                        if inputs:
                            trace._observe(module, phase, "cotangent", values.get(inputs[0]), **common)
                        if outputs:
                            count = trace._counts.get((module, phase, "input_adjoint"), 0) + 1
                            key = f"{phase}/{module}/input_adjoint/{count}"
                            paired = trace.reference is None or key in trace.reference._snapshots
                            trace._observe(module, phase, "input_adjoint", values.get(outputs[0]), paired=paired, **common)
            def finish(self, error=None):
                if error is not None:
                    trace._errors.append({"phase": phase, "exception": type(error).__name__})
        return artifact.observed_fn(*args, _physical_observer=Observer())

    def report(self):
        missing = [] if self.reference is None else sorted(set(self.reference._snapshots) - self._seen)
        differences = [row for row in self._rows if row.get("passed") is False]
        paired = [row for row in self._rows if row["paired"]]
        missing_modules = sorted(set(self.modules) - {row["module"] for row in paired})
        coverage_complete = bool(paired) and not (missing or missing_modules or self._omitted)
        context_succeeded = not self._errors
        complete = coverage_complete and context_succeeded
        return {"scope": "matched_observed_module_primary_outputs_and_cotangents_only",
            "global_first_divergence_verified": False, "timing_eligible": False,
            "cuda_execution_observed": self._cuda, "complete": complete,
            "coverage_complete": coverage_complete, "context_succeeded": context_succeeded,
            "mode": "reference" if self.reference is None else "comparison",
            "passed": complete and not differences if self.reference is not None else None,
            "first_differing_boundary": deepcopy(differences[0]) if differences else None,
            "boundaries": deepcopy(self._rows), "missing_reference_boundaries": missing,
            "missing_modules": missing_modules, "omitted_boundaries": self._omitted,
            "errors": deepcopy(self._errors), "reference_bytes": self._bytes,
            "reference_byte_limit": self.max_bytes, "chunk_elements": self.chunk_elements,
            "tensor_checkpoints_written": 0,
            "artifacts": {"files": deepcopy(self._artifacts), "bytes": self._artifact_bytes,
                          "omissions": deepcopy(self._artifact_omissions), "byte_limit": self.max_artifact_bytes,
                          "file_limit": self.max_artifact_files, "tensor_files_saved": 0}}


def _copy_strided(tensor):
    result = torch.empty_strided(tuple(tensor.shape), tuple(tensor.stride()),
                                 device=tensor.device, dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def _input(shape, dtype, device, layout, seed):
    if layout not in ("nchw", "channels_last", "strided"):
        raise ValueError("Replay layout must be nchw, channels_last or strided")
    generator = torch.Generator(device=device).manual_seed(seed)
    stored_shape = (*shape[:-1], shape[-1] * 2) if layout == "strided" else shape
    value = torch.empty(stored_shape, dtype=dtype, device=device).uniform_(-1, 1, generator=generator)
    if layout == "channels_last":
        value = value.contiguous(memory_format=torch.channels_last)
    return value[..., ::2] if layout == "strided" else value


def cancellation_cotangent(shape, *, device="cpu", dtype=torch.bfloat16, layout="nchw"):
    """Noninteger BF16 loads distinguish FP32 accumulation from BF16 sums."""
    gradient = _input(tuple(shape), dtype, torch.device(device), layout, 8173)
    gradient[..., 0::2, 0::2] = 1.
    gradient[..., 0::2, 1::2] = 1. / 256
    gradient[..., 1::2, 0::2] = -1.
    gradient[..., 1::2, 1::2] = -1. / 512
    return gradient


def _comparison(actual, expected, chunk_elements):
    if actual is None or expected is None:
        return {"passed": actual is expected, "reason": "gradient_presence",
                "actual_present": actual is not None, "expected_present": expected is not None}
    row = {"actual": _description(actual), "expected": _description(expected)}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {**row, "passed": False, "reason": "shape_or_dtype_differs"}
    return {**row, **_scan(actual, chunk_elements, expected=expected)}


def _ordered_2x_adjoint(gradient, input_shape, chunk_elements):
    """Bounded independent four-load oracle, rounding only at BF16 stores."""
    result = torch.empty(input_shape, dtype=gradient.dtype, device=gradient.device)
    _, channels, height, width = input_shape
    for start in range(0, result.numel(), chunk_elements):
        position = torch.arange(start, min(start + chunk_elements, result.numel()),
                                device=gradient.device, dtype=torch.int64)
        w, h = position.remainder(width), position.div(width, rounding_mode="floor").remainder(height)
        c = position.div(width * height, rounding_mode="floor").remainder(channels)
        n = position.div(width * height * channels, rounding_mode="floor")
        value = gradient[n, c, h * 2, w * 2].float() + gradient[n, c, h * 2, w * 2 + 1].float()
        value = value + gradient[n, c, h * 2 + 1, w * 2].float()
        value = value + gradient[n, c, h * 2 + 1, w * 2 + 1].float()
        result.view(-1)[start:start + value.numel()].copy_(value.to(gradient.dtype))
    return result


def _prepared_pair(model, compression, *, device):
    from deepopt.config import CompileConfig
    from deepopt.engine.execution.compilation import compile_model
    from deepopt.opt.policy import prepare_fp8_model

    config = {"runtime": {"precision": "bf16", "bf16_policy": "bf16_storage",
                          "fp8": {"enabled": False}, "activation_compression": compression}}
    eager = prepare_fp8_model(model.to(device), False, config=config)[0]
    native = deepcopy(eager)
    compile_model(native, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
        options={"require_full_aot": True, "use_native_autodiff": True}), config=config)
    return eager, native


def _replay(eager, native, source, compression, *, modules, layout, chunk_elements,
            max_bytes, artifact_dir=None):
    from deepopt import autocast
    from deepopt.compiler import compilation_report

    device = source.device
    reference_trace = BoundaryTrace(modules=modules, max_bytes=max_bytes, chunk_elements=chunk_elements)
    candidate_trace = BoundaryTrace(reference=reference_trace, modules=modules, max_bytes=max_bytes,
                                    chunk_elements=chunk_elements, artifact_dir=artifact_dir)
    reference_input = _copy_strided(source).requires_grad_()
    native_input = _copy_strided(source).requires_grad_()
    context = dict(dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                   activation_compression=compression, config={})
    try:
        with reference_trace.eager(eager):
            with autocast(device.type, **context):
                expected = eager(reference_input)
            cotangent = cancellation_cotangent(expected.shape, device=device, dtype=expected.dtype, layout=layout)
            expected.backward(cotangent)
        with candidate_trace.native():
            with autocast(device.type, **context):
                actual = native(native_input)
            native_cotangent = _copy_strided(cotangent)
            actual.backward(native_cotangent)
    except BaseException as error:
        error.boundary_localization = {"scope": "partial_isolated_replay_boundaries",
            "reference": reference_trace.report(), "candidate": candidate_trace.report(),
            "tensor_checkpoints_written": 0}
        raise
    comparisons = {"output": _comparison(actual, expected, chunk_elements),
                   "input_gradient": _comparison(native_input.grad, reference_input.grad, chunk_elements)}
    for (name, parameter), (other, reference) in zip(native.named_parameters(), eager.named_parameters()):
        if name != other:
            raise ValueError("Replay parameter names differ")
        if parameter.grad is not None and reference.grad is not None:
            comparisons["parameter_gradient/" + name] = _comparison(parameter.grad, reference.grad, chunk_elements)
        elif parameter.grad is not reference.grad:
            comparisons["parameter_gradient/" + name] = {"passed": False, "reason": "gradient_presence_differs"}
    compiler = compilation_report(native).as_dict()
    native_owner = (compiler.get("runtime_fallbacks", 0) == 0 and bool(compiler.get("graphs"))
                    and compiler.get("native_autodiff_operations", 0) > 0
                    and all(not graph.get("eager_autograd_operations") and not graph.get("legacy_backward_operations")
                            and not graph.get("unsupported_deepopt_operations")
                            for graph in compiler["graphs"]))
    row = {"scope": "isolated_same_input_same_cotangent_replay_not_full_model_qualification",
           "cuda_execution_observed": device.type == "cuda", "source": _description(source),
           "identical_inputs": torch.equal(reference_input, native_input),
           "identical_cotangents": torch.equal(cotangent, native_cotangent),
           "cotangent": {**_description(cotangent), **_scan(cotangent, chunk_elements)},
           "native_ownership_verified": native_owner, "comparisons": comparisons,
           "passed_scope": "isolated_numerical_comparisons_and_native_ownership",
           "reference_boundaries": reference_trace.report(), "candidate_boundaries": candidate_trace.report(),
           "passed": native_owner and all(item["passed"] for item in comparisons.values())}
    return row, reference_input.grad, native_input.grad, cotangent


def _device_check(device, large):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Replay device must be CPU or CUDA")
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Replay requires actual BF16 CUDA support")
    if large and device.type != "cuda":
        raise ValueError("Original batch128 decoder geometry replay requires CUDA")
    return device


def nearest_replay_diagnostics(*, device="cpu", large=False, layouts=("nchw", "channels_last", "strided"),
                               compressed=(False, True), include_noninteger=True, geometry_indices=(0, 1, 2),
                               max_bytes=128 * 2**20, chunk_elements=262144, artifact_dir=None):
    """Replay nearest independently with a shared cotangent and ordered oracle."""
    from deepopt._bf16_policy import BF16Policy
    from deepopt.opt.nn.primitive_state import Nearest2dGeometry, nearest2d_precision
    from torch._prims_common import suggest_memory_format
    from examples.vae.qualification import preserve_rng

    device = _device_check(device, large)
    geometries, rows = decoder_geometry_metadata(), []
    with preserve_rng():
        for index in geometry_indices:
            target = geometries[index]
            shape = tuple(target["input_shape"]) if large else (1, 8, 5 + index, 7 + index)
            scales = ((2., 2.),) if large or not include_noninteger else ((2., 2.), (1.7, 2.3))
            for scale in scales:
                for layout in layouts:
                    for packed in compressed:
                        torch.compiler.reset()
                        compression = dict(enabled=packed, bits=2, group_size=1024, norm_xhat_recenter=True)
                        model = nn.Sequential()
                        model.add_module("resize", nn.Upsample(scale_factor=scale, mode="nearest"))
                        eager, native = _prepared_pair(model, compression, device=device)
                        source = _input(shape, torch.bfloat16, device, layout, 717 + index)
                        row, eager_gradient, native_gradient, cotangent = _replay(eager, native, source, compression,
                            modules=("resize",), layout=layout, chunk_elements=chunk_elements, max_bytes=max_bytes,
                            artifact_dir=artifact_dir)
                        geometry = Nearest2dGeometry(shape, tuple(cotangent.shape[-2:]), scale, source.dtype,
                            nearest2d_precision(source.dtype, source.device, BF16Policy("bf16_storage")),
                            suggest_memory_format(source), None, tuple(source.stride()))
                        direct = geometry.backward(cotangent)
                        row.update(target_geometry=target, target_geometry_executed=large, layout=layout,
                                   compression=compression, scales=list(scale),
                                   direct_adjoint=_comparison(native_gradient, direct, chunk_elements))
                        if scale == (2., 2.):
                            oracle = _ordered_2x_adjoint(cotangent, shape, chunk_elements)
                            row["ordered_fp32_accumulation_oracle"] = {
                                "native": _comparison(native_gradient, oracle, chunk_elements),
                                "eager": _comparison(eager_gradient, oracle, chunk_elements),
                                "scratch_scope": "bounded_chunk_fp32_arithmetic_no_full_fp32_cotangent_copy"}
                            row["passed"] = row["passed"] and all(row["ordered_fp32_accumulation_oracle"][name]["passed"]
                                                                  for name in ("native", "eager"))
                            del oracle
                        row["passed"] = row["passed"] and row["direct_adjoint"]["passed"]
                        rows.append(row)
                        del direct, cotangent, source, eager_gradient, native_gradient, eager, native
    return {"scope": "isolated_nearest_replays", "global_first_divergence_verified": False,
            "declared_decoder_geometries": geometries, "large_geometry_requested": large,
            "cuda_execution_observed": device.type == "cuda", "cases": rows,
            "passed": bool(rows) and all(row["passed"] for row in rows), "tensor_checkpoints_written": 0}


def chain_replay_diagnostics(*, device="cpu", large=False, layouts=("nchw", "channels_last", "strided"),
                             compressed=(False, True), geometry_indices=(0, 1, 2),
                             max_bytes=128 * 2**20, chunk_elements=262144, artifact_dir=None):
    """Replay Conv/nearest/depthwise boundaries with identical weights and seed."""
    from examples.vae.qualification import preserve_rng

    device = _device_check(device, large)
    geometries, rows = decoder_geometry_metadata(), []
    with preserve_rng():
        for index in geometry_indices:
            target = geometries[index]
            shape = tuple(target["input_shape"]) if large else (1, 8, 5 + index, 7 + index)
            for layout in layouts:
                for packed in compressed:
                    torch.compiler.reset()
                    torch.manual_seed(9127 + index)
                    compression = dict(enabled=packed, bits=2, group_size=1024, norm_xhat_recenter=True)
                    channels = shape[1]
                    model = nn.Sequential()
                    model.add_module("conv", nn.Conv2d(channels, channels, 1, bias=False))
                    model.add_module("resize", nn.Upsample(scale_factor=2, mode="nearest"))
                    model.add_module("depthwise", nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False))
                    eager, native = _prepared_pair(model, compression, device=device)
                    source = _input(shape, torch.float32, device, layout, 617 + index)
                    row, *gradients = _replay(eager, native, source, compression,
                        modules=("conv", "resize", "depthwise"), layout=layout, chunk_elements=chunk_elements,
                        max_bytes=max_bytes, artifact_dir=artifact_dir)
                    row.update(target_geometry=target, target_geometry_executed=large,
                               layout=layout, compression=compression)
                    rows.append(row)
                    del source, gradients, eager, native
    return {"scope": "isolated_conv_nearest_depthwise_replays", "global_first_divergence_verified": False,
            "declared_decoder_geometries": geometries, "large_geometry_requested": large,
            "cuda_execution_observed": device.type == "cuda", "cases": rows,
            "passed": bool(rows) and all(row["passed"] for row in rows), "tensor_checkpoints_written": 0}
