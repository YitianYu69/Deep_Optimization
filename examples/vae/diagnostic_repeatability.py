"""Metadata-only, fixed-cotangent probes for the final encoder BlurPool VJP.

Completion means the requested executions or deterministic-policy rejections
were recorded. Stability and numerical correctness are separate questions;
this helper neither changes a tolerance nor qualifies a full-model route.
"""

from contextlib import contextmanager
import hashlib
from math import prod, sqrt
import os
from pathlib import Path
import platform
import traceback

import torch
from torch.nn import functional as F


_CHUNK = 65536
_PADS = (1, 2, 1, 2)


def _error(error):
    return {"type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error), "traceback": traceback.format_exc()}


def _flags():
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory}


@contextmanager
def _control_flags(strict):
    previous = _flags()
    try:
        torch.use_deterministic_algorithms(strict, warn_only=False)
        torch.backends.cudnn.benchmark = not strict
        torch.backends.cudnn.deterministic = strict
        yield _flags()
    finally:
        torch.use_deterministic_algorithms(previous["deterministic_algorithms"],
                                         warn_only=previous["deterministic_warn_only"])
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]


def _metadata(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "stride": list(value.stride()), "layout": str(value.layout),
            "contiguous": value.is_contiguous(), "elements": value.numel(),
            "bytes": value.numel() * value.element_size()}


def _fingerprint(value):
    flat = value.detach().cpu().contiguous().reshape(-1)
    digest, squared, total = hashlib.sha256(), 0., 0.
    finite = 0
    for start in range(0, flat.numel(), _CHUNK):
        chunk = flat[start:start + _CHUNK]
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
        wide = chunk.double()
        mask = torch.isfinite(wide)
        finite += int(mask.sum())
        squared += float(wide[mask].square().sum())
        total += float(wide[mask].sum())
    return {**_metadata(value), "logical_sha256": digest.hexdigest(), "l2_norm": sqrt(squared),
            "sum": total, "finite_elements": finite, "nonfinite_elements": value.numel() - finite}


def _regions(indices, height, width):
    """Classify cells receiving multiple reflected preimages, including corners.

    Reflection excludes the outermost source cell itself: a left pad of one
    duplicates source index one. These are accumulation bands, not a generic
    one-pixel image border.
    """
    row, column = (indices // width) % height, indices % width
    left, right, top, bottom = _PADS
    vertical = ((row >= 1) & (row <= top)) | ((row >= height - 1 - bottom) & (row <= height - 2))
    horizontal = ((column >= 1) & (column <= left)) | ((column >= width - 1 - right) & (column <= width - 2))
    return {"corner": vertical & horizontal, "edge": vertical ^ horizontal,
            "interior": ~vertical & ~horizontal}


def _comparison(actual, reference, *, source_spatial):
    actual, reference = actual.reshape(-1), reference.reshape(-1)
    maximum = squared = reference_squared = 0.
    mismatches = 0
    region_counts = {"edge": 0, "corner": 0, "interior": 0} if source_spatial else None
    finite = True
    for start in range(0, actual.numel(), _CHUNK):
        left, right = actual[start:start + _CHUNK], reference[start:start + _CHUNK]
        changed = (left.view(torch.uint8).reshape(-1, left.element_size())
                   != right.view(torch.uint8).reshape(-1, right.element_size())).any(1)
        mismatches += int(changed.sum())
        if region_counts is not None:
            indices = torch.arange(start, start + left.numel())
            for region, mask in _regions(indices, *source_spatial).items():
                region_counts[region] += int((changed & mask).sum())
        a, b = left.double(), right.double()
        chunk_finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        finite &= chunk_finite
        if chunk_finite:
            difference = a - b
            maximum = max(maximum, float(difference.abs().max()))
            squared += float(difference.square().sum())
            reference_squared += float(b.square().sum())
    relative = sqrt(squared / reference_squared) if reference_squared else 0. if squared == 0 else None
    return {"bitwise_equal": mismatches == 0, "mismatched_elements": mismatches,
            "mismatched_regions": region_counts, "finite": finite,
            "max_abs": maximum if finite else None,
            "absolute_l2": sqrt(squared) if finite else None,
            "relative_l2": relative if finite else None,
            "relative_l2_infinite": reference_squared == 0 and squared != 0}


def _seeded(shape, generator, device):
    # Correlated, alternating noninteger pairs exercise cancellation without
    # depending on the caller's CPU or CUDA random-number generator.
    count = prod(shape)
    base = torch.randn((count + 1) // 2, generator=generator, dtype=torch.float32).mul_(.431).add_(.137)
    value = torch.empty(count, dtype=torch.float32)
    value[::2] = base
    value[1::2] = -base[:count // 2] + .0073
    return value.reshape(shape).to(device=device, dtype=torch.bfloat16)


def _run_probe(name, implementation, function, *, repeats, strict, device, inputs, source_spatial):
    row = {"name": name, "implementation": implementation, "completed": False, "passed": False,
           "repeat_stable": None, "deterministic_blocked": False, "cuda_execution_observed": False,
           "observed_cuda_launches": None,
           "same_arguments_each_repeat": True, "inputs": inputs, "repetitions": [],
           "region_partition": "reflection_multiple_preimage_bands" if source_spatial else None}
    baseline = None
    try:
        for iteration in range(repeats):
            value = function()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                row["cuda_execution_observed"] = True
            actual = value.detach().cpu().contiguous()
            if baseline is None:
                baseline = actual.clone()
                row["output"] = _metadata(value)
                row["reference_iteration"] = 0
            if actual.dtype != baseline.dtype or actual.shape != baseline.shape:
                raise ValueError("Repeated probe changed its output shape or dtype")
            row["repetitions"].append({"iteration": iteration,
                "logical_sha256": _fingerprint(actual)["logical_sha256"],
                **_comparison(actual, baseline, source_spatial=source_spatial)})
            del value, actual
        row.update(completed=True, passed=True, status="completed",
                   repeat_stable=all(item["bitwise_equal"] for item in row["repetitions"]),
                   max_abs=max((item["max_abs"] for item in row["repetitions"]
                                if item["max_abs"] is not None), default=None))
    except Exception as error:
        message = str(error).lower().replace("'", "").replace('"', "")
        blocked = strict and isinstance(error, RuntimeError) and any(phrase in message for phrase in (
            "does not have a deterministic implementation",
            "you set torch.use_deterministic_algorithms(true)",
            "deterministic behavior was enabled"))
        row.update(error=_error(error), completed=blocked, passed=blocked,
                   deterministic_blocked=blocked,
                   status="deterministic_blocked" if blocked else "execution_failure")
    return row, baseline


def _stable(rows):
    values = [row["repeat_stable"] for row in rows]
    return False if False in values else None if None in values else True


def _source_file(module):
    path = Path(module.__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_repeatability_diagnostics(device="cuda:0", seed=42, repeats=4, original_shapes=True):
    """Replay fixed BlurPool adjoints under two separately labeled policies.

    ``passed``/``completed`` concern execution and error collection only.
    ``repeat_stable`` records byte stability; a deterministic-policy rejection
    remains visible as ``deterministic_blocked`` rather than a fallback.
    No tensors or checkpoints are written, and no full-model gate is evaluated.
    """
    report = {"schema_version": 1, "scope": "backward_repeatability_operator_diagnostics",
        "status": "unqualified", "completed": False, "passed": False,
        "qualification_eligible": False, "cuda_qualified": False, "timing_eligible": False,
        "math_correctness_assessed": False, "cuda_execution_observed": False,
        "tensor_checkpoints_written": 0, "device": str(device), "seed": seed,
        "repeats": repeats, "original_shapes": original_shapes, "controls": [],
        "repeat_stable": None, "deterministic_blocked": False,
        "completion_semantics": "all requested attempts completed or recognized deterministic rejections recorded"}
    try:
        if type(seed) is not int or type(repeats) is not int or repeats < 2 or type(original_shapes) is not bool:
            raise ValueError("seed must be an integer, repeats >= 2, and original_shapes a Boolean")
        selected = torch.device(device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("Repeatability probes support CPU semantic execution or CUDA")
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU substitution was executed")
        from deepopt.opt.nn import padding
        import deepopt

        initial_flags = _flags()
        report.update(initial_flags=initial_flags,
            execution_scope="CUDA_operator_replay" if selected.type == "cuda" else "CPU_semantic_only",
            software={"python": platform.python_version(), "torch": torch.__version__,
                "torch_git": torch.version.git_version,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "deepopt_file": str(deepopt.__file__)},
            implementation_sources={"helper": {"path": str(Path(__file__).resolve()),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                "padding_adjoint": _source_file(padding)})
        if selected.type == "cuda":
            props = torch.cuda.get_device_properties(selected)
            report["software"]["gpu"] = {"name": props.name, "uuid": str(props.uuid),
                "capability": [props.major, props.minor], "total_memory": props.total_memory}
        shape = (1, 512, 56, 56) if original_shapes else (1, 8, 5, 7)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        source = _seeded(shape, generator, selected)
        padded = F.pad(source, _PADS, mode="reflect")
        coefficients = torch.tensor([1., 3., 3., 1.], dtype=torch.float32)
        weights = (torch.outer(coefficients, coefficients) / 64).reshape(1, 1, 4, 4)
        weights = weights.repeat(shape[1], 1, 1, 1).to(device=selected, dtype=torch.bfloat16)
        # Convolution's exact formula for a four-tap filter and three pads.
        output_shape = (*shape[:2], (padded.shape[2] - 4) // 2 + 1, (padded.shape[3] - 4) // 2 + 1)
        blur_cotangent = _seeded(output_shape, generator, selected)
        pad_cotangent = _seeded(tuple(padded.shape), generator, selected)
        geometry = padding.PaddingGeometry.from_tensor(source, _PADS, "reflect")
        fixed_values = {"source": source, "padded_source": padded, "weights": weights,
                        "blur_cotangent": blur_cotangent, "pad_cotangent": pad_cotangent}
        fingerprints = {name: _fingerprint(value) for name, value in fixed_values.items()}
        report.update(geometry={"module": "encoder.12", "source_shape": list(shape),
            "padded_shape": list(padded.shape), "output_shape": list(output_shape), "pads": list(_PADS),
            "mode": "reflect", "kernel_size": [4, 4], "stride": [2, 2], "groups": shape[1],
            "kernel_recipe": "outer([1,3,3,1], [1,3,3,1]) / 64",
            "region_definition": "corner=both axes have reflected preimages; edge=one axis; interior=neither"},
            fixed_inputs=fingerprints,
            bounded_footprint={"largest_input_tensor_bytes": max(item["bytes"] for item in fingerprints.values()),
                "comparison_chunk_elements": _CHUNK, "payloads_saved": False},
            cotangent_recipe="seeded_adjacent_noninteger_cancellation_pairs_BF16",
            source_norm=fingerprints["source"]["l2_norm"])

        def convolution_dx():
            return torch.ops.aten.convolution_backward.default(blur_cotangent, padded, weights, None,
                [2, 2], [0, 0], [1, 1], False, [0, 0], shape[1], [True, False, False])[0]

        def raw_blur_vjp():
            with torch.inference_mode(False), torch.enable_grad():
                leaf = source.detach().clone().requires_grad_()
                output = F.conv2d(F.pad(leaf, _PADS, mode="reflect"), weights,
                                  stride=2, groups=shape[1])
                return torch.autograd.grad(output, leaf, blur_cotangent)[0]

        probes = (
            ("raw_reflection_backward", "torch.ops.aten.reflection_pad2d_backward.default",
             lambda: torch.ops.aten.reflection_pad2d_backward.default(pad_cotangent, source, list(_PADS)),
             {"cotangent": fingerprints["pad_cotangent"], "source": fingerprints["source"]}, shape[-2:]),
            ("metadata_padding_adjoint", "installed deepopt.opt.nn.padding.padding_adjoint",
             lambda: padding.padding_adjoint(pad_cotangent, geometry),
             {"cotangent": fingerprints["pad_cotangent"], "source_geometry": report["geometry"]}, shape[-2:]),
            ("grouped_convolution_dx", "torch.ops.aten.convolution_backward.default; output_mask=[True,False,False]",
             convolution_dx, {"cotangent": fingerprints["blur_cotangent"],
                "source": fingerprints["padded_source"], "weights": fingerprints["weights"]}, None),
            ("raw_blur_pool_vjp", "PyTorch autograd of F.pad(reflect) followed by F.conv2d",
             raw_blur_vjp, {"cotangent": fingerprints["blur_cotangent"],
                "source": fingerprints["source"], "weights": fingerprints["weights"]}, shape[-2:]),
        )
        for strict in (False, True):
            control = {"name": "strict_deterministic" if strict else "recorded_nondeterministic",
                       "probes": [], "scope": "operator_only_not_full_model", "comparisons": {},
                       "requested_flags": {"deterministic_algorithms": strict,
                           "deterministic_warn_only": False, "cudnn_benchmark": not strict,
                           "cudnn_deterministic": strict}}
            report["controls"].append(control)
            first_outputs = {}
            with _control_flags(strict) as effective, torch.autocast(selected.type, enabled=False):
                control["effective_flags"] = effective
                for name, implementation, function, inputs, spatial in probes:
                    row, baseline = _run_probe(name, implementation, function, repeats=repeats, strict=strict,
                        device=selected, inputs=inputs, source_spatial=spatial)
                    control["probes"].append(row)
                    if baseline is not None:
                        first_outputs[name] = baseline
                if {"raw_reflection_backward", "metadata_padding_adjoint"} <= first_outputs.keys():
                    control["comparisons"]["raw_vs_metadata_first_outputs"] = _comparison(
                        first_outputs["metadata_padding_adjoint"], first_outputs["raw_reflection_backward"],
                        source_spatial=shape[-2:])
            control["fixed_input_hashes_after"] = {name: _fingerprint(value)["logical_sha256"]
                                                    for name, value in fixed_values.items()}
            unchanged = all(control["fixed_input_hashes_after"][name] == value["logical_sha256"]
                            for name, value in fingerprints.items())
            control.update(completed=all(row["completed"] for row in control["probes"]),
                passed=unchanged and all(row["passed"] for row in control["probes"]),
                fixed_inputs_unchanged=unchanged,
                repeat_stable=_stable(control["probes"]),
                deterministic_blocked=any(row["deterministic_blocked"] for row in control["probes"]))
        report.update(completed=all(row["completed"] for row in report["controls"]),
            passed=all(row["passed"] for row in report["controls"]),
            repeat_stable=_stable(report["controls"]),
            deterministic_blocked=any(row["deterministic_blocked"] for row in report["controls"]),
            cuda_execution_observed=any(row["cuda_execution_observed"] for control in report["controls"]
                                        for row in control["probes"]),
            final_flags=_flags(), caller_flags_restored=_flags() == initial_flags)
    except Exception as error:
        report["error"] = _error(error)
    return report
