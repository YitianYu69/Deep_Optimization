"""Opt-in benchmark qualification. Importing this module never imports Torch.

These checks qualify the measured executor on its actual represented operands;
FP8/codec approximation quality is reported separately. They are not a claim of
complete backend parity or a substitute for the CUDA qualification suite.
"""

from dataclasses import dataclass
import math
from statistics import median
from time import perf_counter


QUALIFICATION_SCHEMA_VERSION = 1


def validation_valid(value, required=()):
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("status") != "passed":
        return False
    if not isinstance(value.get("oracle"), str) or not value["oracle"].strip():
        return False
    declared = value.get("required_checks")
    if (not isinstance(declared, list) or not declared or not isinstance(required, (list, tuple))
            or any(not isinstance(name, str) or not name for name in (*declared, *required))):
        return False
    checks = value.get("checks")
    names = set(required) | set(declared)
    if not isinstance(checks, dict) or not checks or not names.issubset(checks):
        return False
    for check in checks.values():
        if not isinstance(check, dict) or check.get("status") != "passed" or check.get("finite") is not True:
            return False
        if not isinstance(check.get("metric"), str) or not check["metric"]:
            return False
        if any(not isinstance(check.get(key), (int, float)) or isinstance(check[key], bool)
               or not math.isfinite(check[key]) or check[key] < 0 for key in ("value", "limit")):
            return False
        if check["value"] > check["limit"]:
            return False
    return True


def failed_check(reason):
    return {"status": "failed", "finite": False, "metric": "required_tensor_check", "value": 1.,
            "limit": 0., "reason": str(reason)}


def validation_result(checks, required, oracle, **metadata):
    checks = dict(checks)
    for name in required:
        checks.setdefault(name, failed_check(f"Missing requested {name}"))
    result = {"schema_version": 1, "status": "passed", "oracle": oracle,
              "validation_policy": "torch-dtype-default-per-role-l2-v1",
              "required_checks": list(required), "checks": checks, **metadata}
    if not validation_valid(result, required):
        result["status"] = "failed"
    return result


def tensor_check(actual, expected, *, comparison_dtype=None, exact=False):
    """Per-role full finite check and dtype-default numerical comparison.

    The normalized L2 bound is the L2 form of Torch's default elementwise
    atol/rtol budget: atol*sqrt(numel) + rtol*||reference||. Tolerances follow
    the executor's destination precision, never observed FP8 pipeline errors.
    """
    import torch
    from torch.testing._comparison import default_tolerances

    if actual is None or expected is None:
        return failed_check("Requested actual/reference tensor is missing")
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape:
        return failed_check(f"Shape mismatch: {tuple(actual.shape)} versus {tuple(expected.shape)}")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return failed_check("Actual or independent reference contains nonfinite values")
    dtype = comparison_dtype or actual.dtype
    if exact:
        value = float(not torch.equal(actual, expected))
        return {"status": "passed" if value == 0 else "failed", "finite": True,
                "metric": "exact_tensor_mismatch", "value": value, "limit": 0.}
    rtol, atol = default_tolerances(torch.empty((), dtype=dtype))
    difference = actual.double() - expected.double()
    norm = float(expected.double().norm())
    bound = atol * math.sqrt(expected.numel()) + rtol * norm
    error = float(difference.norm())
    value = error / bound if bound else float(error != 0)
    if not math.isfinite(value):
        return failed_check("Numerical comparison overflowed")
    return {"status": "passed" if value <= 1 else "failed", "finite": True,
            "metric": "l2_error_over_dtype_default_budget", "value": value, "limit": 1.,
            "relative_l2": error / max(norm, 1e-30), "max_abs": float(difference.abs().max()) if difference.numel() else 0.,
            "rtol": rtol, "atol": atol, "comparison_dtype": str(dtype),
            "tolerance_policy": "torch.testing dtype defaults; per-role L2 budget"}


def _scales(scale, shape, block):
    if block is None:
        return scale
    return scale.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)[:shape[0], :shape[1]]


def cpu_dense(value):
    """Decode actual FP8 bytes/scales on CPU without a backend decoder."""
    if not hasattr(value, "_scale_inv"):
        return value.detach().cpu()
    data = value._data.detach().cpu().float()
    scale = value._scale_inv.detach().cpu()
    dense = data * _scales(scale, data.shape, value._block_shape)
    shape = tuple(value._logical_shape)
    if value._activation_layout == "channels":
        return dense.reshape(shape[0], *shape[2:], shape[1]).movedim(-1, 1)
    return dense.reshape(shape)


def _round_like(expected, actual):
    import torch

    if not hasattr(actual, "_scale_inv"):
        return expected.to(actual.dtype)
    shape = tuple(actual._data.shape)
    value = expected.movedim(1, -1).reshape(shape) if actual._activation_layout == "channels" else expected.reshape(shape)
    scale = _scales(actual._scale_inv.detach().cpu(), shape, actual._block_shape).double()
    maximum = torch.finfo(actual._data.dtype).max
    result = (value / scale).clamp(-maximum, maximum).to(actual._data.dtype).float() * scale
    logical = tuple(actual._logical_shape)
    return result.reshape(logical[0], *logical[2:], logical[1]).movedim(-1, 1) if actual._activation_layout == "channels" else result.reshape(logical)


def _quality(actual, expected):
    difference = actual.double() - expected.double()
    return {"relative_l2": float(difference.norm() / expected.double().norm().clamp_min(1e-30)),
            "max_abs": float(difference.abs().max()) if difference.numel() else 0.}


def _indices(size, maximum=32):
    import torch
    return torch.linspace(0, size - 1, min(size, maximum)).round().long().unique()


def _operand(value):
    data = value.data.detach().cpu().double()
    return data * _scales(value.scale_inv.detach().cpu().double(), data.shape, value.block_shape)


def qualify_linear(case):
    """Check final output/gradients against the actual executor operands.

    Large GEMMs use deterministic <=32x32 output samples with the complete
    reduction dimension, avoiding multi-billion-operation CPU oracle matmuls.
    Every final output/gradient tensor is checked for finiteness in full.
    """
    import torch
    from unittest.mock import patch
    from deepopt.opt.nn.autograd import linear

    required = ["output"] + (["dx", "dweight"] if case.training else [])
    if case.training and case.source.bias is not None:
        required.append("dbias")
    records = {}
    original = linear.gemm_nt

    def observed(a, b, **kwargs):
        output = original(a, b, **kwargs)
        role = {"forward": "output", "dx": "dx", "dw": "dweight"}.get(kwargs.get("diagnostic_role"))
        if role is not None:
            aa, bb = _operand(a), _operand(b)
            rows, columns = _indices(aa.shape[0]), _indices(bb.shape[0])
            expected = aa[rows] @ bb[columns].T
            if kwargs.get("bias") is not None:
                expected += kwargs["bias"].detach().cpu().double()[columns]
            dtype = kwargs.get("out_dtype", output.dtype)
            records[role] = (rows, columns, expected.to(dtype), dtype)
        return output

    try:
        with patch.object(linear, "gemm_nt", observed):
            output = case.step()
        actuals = {"output": output, "dx": case.x.grad, "dweight": case.source.weight.grad,
                   "dbias": case.source.bias.grad if case.source.bias is not None else None}
        checks = {}
        for role in required:
            actual = actuals[role]
            if role == "dbias":
                checks[role] = tensor_check(actual, case.grad.detach().cpu().double().sum(0).to(actual.dtype) if actual is not None else None)
            elif role not in records:
                checks[role] = failed_check("The requested executor role was not observed")
            elif actual is None or not bool(torch.isfinite(actual).all()):
                checks[role] = failed_check("Missing or nonfinite final tensor")
            else:
                rows, columns, expected, dtype = records[role]
                matrix = actual.detach().cpu().reshape(-1, actual.shape[-1])
                checks[role] = tensor_check(matrix[rows][:, columns], expected, comparison_dtype=dtype)
                checks[role]["full_tensor_finite"] = True
                checks[role]["sample_shape"] = list(expected.shape)
        return validation_result(checks, required, "CPU FP64 matmul on identical actual quantized operands/scales",
                                 scope="deterministic output samples, complete K reductions; full-tensor finite checks",
                                 approximation_quality="errors_vs_fp32 is a separate pipeline diagnostic")
    except Exception as error:
        return validation_result({"output": failed_check(error)}, required, "CPU identical-operand GEMM oracle")


def _decode_packed(data, step, minimum, shape, group, bits):
    import torch

    rows, columns = shape
    groups_per_row = (columns + group - 1) // group
    shifts = torch.arange(32 // bits, dtype=torch.int64) * bits
    codes = ((data.long()[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(data.shape[0], -1)[:, :group]
    decoded = minimum.double()[:, None] + codes.double() * step.double()[:, None]
    return decoded.reshape(rows, groups_per_row * group)[:, :columns]


def qualify_layer_norm(forward, source, model, gradient, suffix, *, compressed=False):
    """Independent native output/statistics and actual saved-state backward oracle."""
    import torch
    from unittest.mock import patch
    from deepopt.opt.nn.autograd.normalization.layer import LayerNormFunction

    required = ["output", "dx"]
    norm = model[0]
    if norm.weight is not None:
        required.append("dweight")
    if norm.bias is not None:
        required.append("dbias")
    captured = {}
    original_forward, original_backward = LayerNormFunction.forward, LayerNormFunction.backward

    def capture_forward(ctx, x, *args):
        result = original_forward(ctx, x, *args)
        captured.update(ctx=ctx, input=cpu_dense(x).double())
        return result

    def capture_backward(ctx, dy):
        captured["dy"] = cpu_dense(dy).double()
        return original_backward(ctx, dy)

    try:
        model.zero_grad(set_to_none=True)
        source.grad = None
        with patch.object(LayerNormFunction, "forward", staticmethod(capture_forward)), \
                patch.object(LayerNormFunction, "backward", staticmethod(capture_backward)):
            output = forward()
            ctx = captured["ctx"]
            saved = [value.detach().cpu() if value is not None else None for value in ctx.saved_tensors]
            output.backward(gradient)
        values = captured["input"].reshape(-1, math.prod(suffix))
        variance, mean = torch.var_mean(values, 1, correction=0)
        inverse = torch.rsqrt(variance + ctx.geometry.eps)
        h_native = (values - mean[:, None]) * inverse[:, None]
        weight = norm.weight.detach().cpu().double().reshape(-1) if norm.weight is not None else torch.ones(values.shape[1], dtype=torch.float64)
        expected_y = torch.nn.functional.layer_norm(captured["input"], tuple(suffix),
            norm.weight.detach().cpu().double() if norm.weight is not None else None,
            norm.bias.detach().cpu().double() if norm.bias is not None else None, ctx.geometry.eps)
        data, scale, minimum, actual_inverse, actual_mean, _, shift, gain = saved
        checks = {"output": tensor_check(cpu_dense(output), _round_like(expected_y, output), comparison_dtype=source.dtype),
                  "saved_inverse": tensor_check(actual_inverse, inverse.float(), comparison_dtype=torch.float32)}
        if compressed:
            decoded = _decode_packed(data, scale, minimum, values.shape, ctx.compression["group_size"], ctx.compression["bits"])
            expected_shift = decoded.mean(1)
            decoded_variance = (decoded - expected_shift[:, None]).square().mean(1)
            expected_gain = torch.where(decoded_variance > 0, (h_native.square().mean(1) / decoded_variance.clamp_min(1e-12)).sqrt(), 0.)
            checks["saved_shift"] = tensor_check(shift, expected_shift.float(), comparison_dtype=torch.float32)
            checks["saved_gain"] = tensor_check(gain, expected_gain.float(), comparison_dtype=torch.float32)
            h = (decoded - shift.double()[:, None]) * gain.double()[:, None]
        else:
            checks["saved_mean"] = tensor_check(actual_mean, mean.float(), comparison_dtype=torch.float32)
            h = (values - actual_mean.double()[:, None]) * actual_inverse.double()[:, None]
        dy = captured["dy"].reshape_as(values)
        u = dy * weight
        expected_dx = ((u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * actual_inverse.double()[:, None]).reshape(source.shape)
        checks["dx"] = tensor_check(source.grad, expected_dx.to(source.dtype), comparison_dtype=source.dtype)
        if "dweight" in required:
            checks["dweight"] = tensor_check(norm.weight.grad, (dy * h).sum(0).reshape(suffix).to(norm.weight.dtype))
        if "dbias" in required:
            checks["dbias"] = tensor_check(norm.bias.grad, dy.sum(0).reshape(suffix).to(norm.bias.dtype))
        quality = {"xhat": _quality(h, h_native)} if compressed else {}
        return validation_result(checks, required, "CPU native LayerNorm output/statistics; CPU-decoded actual saved-state backward",
                                 compression_quality=quality, scope="full tensors; actual represented input and upstream gradient")
    except Exception as error:
        return validation_result({"output": failed_check(error)}, required, "independent LayerNorm/saved-state oracle")


def qualify_pooling(sample, source, represented, gradient, operation, options, *, compressed=False, compression_errors=None, actual_indices=None, actual_dx=None):
    """Native CPU pooling oracle for deterministic max/average fixtures."""
    import torch

    required = ["output", "dx"]
    try:
        x = cpu_dense(represented).double().requires_grad_()
        kwargs = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in options.items()}
        expected = getattr(torch.nn.functional, operation)(x, **kwargs)
        expected_indices = expected[1] if isinstance(expected, tuple) else None
        expected = expected[0] if isinstance(expected, tuple) else expected
        expected.backward(cpu_dense(gradient).double())
        checks = {"output": tensor_check(cpu_dense(sample), _round_like(expected.detach(), sample), comparison_dtype=source.dtype),
                  "dx": tensor_check(actual_dx if actual_dx is not None else source.grad,
                                     x.grad.to(source.dtype), comparison_dtype=source.dtype)}
        if expected_indices is not None:
            checks["indices"] = tensor_check(actual_indices, expected_indices, exact=True)
        if compressed:
            for key in ("output_max_abs_vs_compression_off", "dx_max_abs_vs_compression_off"):
                value = (compression_errors or {}).get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    checks["compression_" + key] = failed_check("Missing/nonfinite exact compression comparison")
                else:
                    checks["compression_" + key] = {"status": "passed" if value == 0 else "failed", "finite": True,
                        "metric": "exact_compression_max_abs", "value": value, "limit": 0.}
        return validation_result(checks, required, "native CPU pooling/autograd on actual represented input",
                                 scope="full output and input gradient; isolated deterministic pooling")
    except Exception as error:
        return validation_result({"output": failed_check(error)}, required, "native CPU pooling/autograd")


@dataclass(frozen=True)
class QualifiedTimingPlan:
    base: object
    pilots: tuple

    def __getattr__(self, name):
        return getattr(self.base, name)

    def sampling(self, elapsed_seconds):
        return {**self.base.sampling(elapsed_seconds), "pilot_samples_seconds": list(self.pilots),
                "pilot_selection": "median_of_three_warm_calls", "qualification_timing": True,
                "stage_accounting": {"first_call_seconds": self.base.cold_first_call_seconds,
                    "warm_calibration_seconds": sum(self.pilots),
                    "warmup_measurement_and_other_seconds": max(0., elapsed_seconds - self.base.cold_first_call_seconds - sum(self.pilots))}}


def qualified_timing_plan(operation, args, *, synchronize=lambda: None, clock=perf_counter):
    """Cold accounting always runs, including exhaustive qualification runs."""
    from dataclasses import replace
    from types import SimpleNamespace
    from deepopt._benchmarking import timing_plan_from_cost

    timing_plan_from_cost(args, 0.)  # Reject malformed caps before executing.
    synchronize()
    began = clock()
    operation()
    synchronize()
    cold = clock() - began
    pilots = []
    for _ in range(3):
        began = clock()
        operation()
        synchronize()
        pilots.append(max(clock() - began, 1e-9))
    budget = float(getattr(args, "time_budget", 0.))
    remaining = max(budget - sum(pilots), 1e-12) if budget else 0.
    effective = SimpleNamespace(**{**vars(args), "time_budget": remaining})
    plan = timing_plan_from_cost(effective, median(pilots), cold_seconds=cold)
    plan = replace(plan, time_budget_seconds=budget, calibration_calls=4)
    return QualifiedTimingPlan(plan, tuple(pilots))
