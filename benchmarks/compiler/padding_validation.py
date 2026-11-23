"""Small independent geometry/rounding oracle for padding qualification.

This CPU diagnostic is outside throughput measurement. It never retains the
source activation or serializes tensors. It bounds ordinary finite additions;
it is not a CUDA nondeterminism qualification until the CUDA controls execute.
"""

import math


def padding_locations(shape, pads, mode, *, max_elements=1_000_000):
    """Map each padded output cell to its original flat input coordinate."""
    import torch

    shape, pads = tuple(shape), tuple(pads)
    rank = len(pads) // 2
    if mode not in ("reflect", "replicate") or not pads or len(pads) % 2 or rank > 3:
        raise ValueError("Oracle requires reflection/replication padding of one to three axes")
    if len(shape) not in (rank + 1, rank + 2) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Oracle requires a positive native padding input shape")
    if type(max_elements) is not int or max_elements <= 0:
        raise ValueError("max_elements must be a positive integer")
    output_shape = list(shape)
    for index in range(rank):
        axis = len(shape) - index - 1
        left, right = pads[2 * index:2 * index + 2]
        if type(left) is not int or type(right) is not int:
            raise ValueError("Padding values must be integers")
        if mode == "reflect" and (left >= shape[axis] or right >= shape[axis]):
            raise ValueError("Reflection padding must be smaller than the original axis")
        output_shape[axis] += left + right
    if any(n <= 0 for n in output_shape):
        raise ValueError("Oracle requires a positive padded output shape")
    if max(math.prod(shape), math.prod(output_shape)) > max_elements:
        raise ValueError("Padding oracle exceeds its explicit diagnostic element bound")
    # Build directly at final output geometry. Sequential axis expansion can
    # violate the bound before a different axis is cropped back down.
    locations = torch.zeros(output_shape, dtype=torch.int64)
    for axis, size in enumerate(shape):
        index = len(shape) - axis - 1
        left = pads[2 * index] if index < rank else 0
        coordinate = torch.arange(output_shape[axis]) - left
        if index < rank:
            if mode == "reflect":
                coordinate = torch.where(coordinate < 0, -coordinate,
                                         torch.where(coordinate >= size, 2 * size - 2 - coordinate, coordinate))
            else:
                coordinate = coordinate.clamp(0, size - 1)
            if bool(((coordinate < 0) | (coordinate >= size)).any()):
                raise ValueError("Invalid reflection mapping for the supplied original geometry")
        view = [1] * len(shape)
        view[axis] = output_shape[axis]
        locations.add_(coordinate.reshape(view) * math.prod(shape[axis + 1:]))
    return locations


def padding_reference(gradient, shape, pads, mode, *, max_elements=1_000_000):
    """Return FP64 sums, contributor counts and sums of absolute terms."""
    import torch

    if gradient.dtype not in (torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Padding oracle requires BF16, FP32 or FP64 upstream gradients")
    locations = padding_locations(shape, pads, mode, max_elements=max_elements)
    if tuple(gradient.shape) != tuple(locations.shape):
        raise ValueError("Upstream gradient does not match padded geometry")
    terms = gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(terms).all()):
        raise ValueError("Rounding oracle requires finite upstream gradients")
    if bool(((terms.abs() > 0) & (terms.abs() < torch.finfo(gradient.dtype).tiny)).any()):
        raise ValueError("Subnormal upstream gradients require an explicit device flush policy")
    offsets = locations.reshape(-1)
    count = math.prod(shape)
    total = torch.zeros(count, dtype=torch.float64).scatter_add_(0, offsets, terms)
    absolute = torch.zeros_like(total).scatter_add_(0, offsets, terms.abs())
    if not bool(torch.isfinite(total).all() and torch.isfinite(absolute).all()):
        raise ValueError("FP64 padding oracle overflow; fixture needs a wider exact reference")
    contributors = torch.bincount(offsets, minlength=count)
    return {"sum": total.reshape(shape), "sum_abs": absolute.reshape(shape), "dtype": gradient.dtype,
            "contributors": contributors.reshape(shape)}


def padding_rounding_budget(reference, dtype):
    """gamma_(m-1)*sum(abs(g)), including FP64 oracle rounding/underflow.

    The first term added to zero is exact. The budget includes every subsequent
    target-dtype rounding and also covers wider accumulation with a final cast.
    Discrete geometry controls remain exact and use no floating error budget.
    """
    import torch

    if dtype not in (torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Padding qualification supports BF16, FP32 and FP64")
    if reference["dtype"] != dtype:
        raise ValueError("Padding error budget requires matching upstream/output dtypes")
    operations = (reference["contributors"] - 1).clamp_min(0).to(torch.float64)
    unit = torch.finfo(dtype).eps / 2
    product = operations * unit
    if bool((product >= 1).any()):
        raise ValueError("Contributor count exceeds the meaningful rounding-bound domain")
    gamma = product / (1 - product)
    oracle_unit = torch.finfo(torch.float64).eps / 2
    oracle_product = operations * oracle_unit
    oracle_gamma = oracle_product / (1 - oracle_product)
    # The FP64 absolute-term sum is also rounded; enclose that error rather
    # than adding an unexplained relative/absolute test tolerance.
    absolute_upper = reference["sum_abs"] / (1 - oracle_gamma)
    underflow = reference["contributors"] * (torch.finfo(dtype).tiny * unit)
    budget = (gamma + oracle_gamma) * absolute_upper + underflow
    if not bool(torch.isfinite(budget).all() and torch.isfinite(reference["sum"]).all()):
        raise ValueError("Padding oracle sum and rounding budget must be finite")
    return budget


def padding_error_report(actual, reference):
    """Scalar-only evidence; every cell must satisfy its own rounding budget."""
    import torch

    value = actual.detach().to(device="cpu", dtype=torch.float64)
    if tuple(value.shape) != tuple(reference["sum"].shape):
        raise ValueError("Adjoint result has incorrect input geometry")
    budget = padding_rounding_budget(reference, actual.dtype)
    error = (value - reference["sum"]).abs()
    finite = bool(torch.isfinite(value).all())
    violations = error > budget
    finite_error = bool(torch.isfinite(error).all())
    return {"passed": finite and finite_error and not bool(violations.any()), "finite": finite,
            "violating_elements": int(violations.sum()), "max_abs_error": float(error.max()) if finite_error else None,
            "max_budget": float(budget.max()),
            "max_contributors": int(reference["contributors"].max()),
            "policy": "per_cell_gamma_m_minus_one_sum_abs", "dtype": str(actual.dtype)}
