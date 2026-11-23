"""Host-only planning and shared dispatch for saved-coefficient multiply VJPs.

Importing this module never imports Triton. The planner is also the native
fusion cost gate, so a selected reduction cannot allocate a larger partial
buffer than the one admitted during compilation.
"""

from dataclasses import asdict, dataclass
from math import prod

import torch


DEFAULT_SCRATCH_BYTES = 8 * 1024 * 1024


def contiguous_strides(shape):
    strides, stride = [], 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


@dataclass(frozen=True)
class PackedMultiplyPlan:
    shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    padded_target: tuple[int, ...]
    reduction_shape: tuple[int, ...]
    outputs: int
    reduced_elements: int
    parts: int
    output_tile: int
    output_tiles: int
    reduction_loops: int
    scratch_bytes: int
    reduction_block: int = 256

    @property
    def direct(self):
        return self.shape == self.target_shape

    def as_dict(self):
        return asdict(self)


def plan_packed_multiply(shape, target_shape, scratch_bytes=DEFAULT_SCRATCH_BYTES):
    """Plan one invocation without allocating tensor payloads or GPU state."""
    shape, target_shape = tuple(shape), tuple(target_shape)
    if (len(shape) > 8 or len(target_shape) > len(shape)
            or any(type(size) is not int or size < 0 for size in (*shape, *target_shape))):
        raise ValueError("broadcast_geometry")
    if type(scratch_bytes) is not int or scratch_bytes < 0:
        raise ValueError("reduction_scratch_bytes")
    target = (1,) * (len(shape) - len(target_shape)) + target_shape
    if any(target_size not in (1, size) for size, target_size in zip(shape, target)):
        raise ValueError("broadcast_or_sum_to_size_contract")
    reduction = tuple(size if target_size == 1 else 1 for size, target_size in zip(shape, target))
    outputs, reduced = prod(target_shape), prod(reduction)
    if shape == target_shape or not outputs or not reduced:
        return PackedMultiplyPlan(shape, target_shape, target, reduction, outputs, reduced,
                                  0, 0, 0, 0, 0)
    reduction_tiles = (reduced + 255) // 256
    parts = min(reduction_tiles, 256)
    available_outputs = min(8192, scratch_bytes // (parts * 4))
    if available_outputs < 1:
        raise ValueError("bounded_reduction_cost_gate")
    # Round the requested output width up for its tail, then round the allowed
    # width down. Both are powers of two and the allocation fits the cap.
    output_tile = min(1 << (outputs - 1).bit_length(), 1 << (available_outputs.bit_length() - 1))
    return PackedMultiplyPlan(shape, target_shape, target, reduction, outputs, reduced,
        parts, output_tile, (outputs + output_tile - 1) // output_tile,
        (reduction_tiles + parts - 1) // parts, parts * output_tile * 4)


def packed_multiply_rejection(*, gradient_shape, gradient_dtype, gradient_device,
                              options, shape, product_dtype, target_shape,
                              target_dtype=None, kernel_policy=None):
    """Return the same explicit eligibility reason for eager and native calls."""
    if torch.device(gradient_device).type != "cuda":
        return "device"
    if any(dtype not in (torch.bfloat16, torch.float32) for dtype in
           (gradient_dtype, options.get("logical_dtype"), product_dtype, target_dtype or product_dtype)):
        return "dtype_or_scalar_promotion"
    if options.get("traversal") != "logical_row_major" or options.get("source_cast") is not None:
        return "codec_traversal_or_cast"
    if (type(options.get("bits")) is not int or options["bits"] not in (1, 2, 4, 8)
            or type(options.get("group_size")) is not int or options["group_size"] <= 0):
        return "codec_contract"
    try:
        if tuple(torch.broadcast_shapes(gradient_shape, options["logical_shape"])) != tuple(shape):
            return "broadcast_or_sum_to_size_contract"
        plan_packed_multiply(shape, target_shape)
    except (RuntimeError, KeyError, TypeError, ValueError):
        return "broadcast_geometry"
    if kernel_policy is not None:
        if kernel_policy.route("compiler_pointwise", "compiler.backward") != "triton":
            return "backend_has_no_packed_multiply"
        if kernel_policy.pinned("activation_codec", "compiler.backward") and kernel_policy.route(
                "activation_codec", "compiler.backward") != "triton":
            return "consumed_role_backend_conflict"
    return None


def multiply(gradient, data, step, minimum, *, options, shape, product_dtype, target_shape,
             target_dtype=None, scratch_bytes=DEFAULT_SCRATCH_BYTES, kernel_policy=None,
             execution_info=None, allow_materialized=True):
    """Execute the bounded packed route or a deliberately reported codec oracle.

    Native static-graph tests use the CPU oracle; CPU execution is never
    reported as packed CUDA execution. No error from an admitted CUDA launch is
    swallowed and retried with activation-sized materialization.
    """
    target_dtype = target_dtype or product_dtype
    reason = packed_multiply_rejection(gradient_shape=gradient.shape, gradient_dtype=gradient.dtype,
        gradient_device=gradient.device, options=options, shape=shape, product_dtype=product_dtype,
        target_shape=target_shape, target_dtype=target_dtype, kernel_policy=kernel_policy)
    if reason is None and (getattr(gradient, "layout", torch.strided) != torch.strided
            or getattr(gradient, "is_nested", False)):
        reason = "gradient_layout"
    try:
        launch = plan_packed_multiply(shape, target_shape, scratch_bytes)
    except ValueError as error:
        launch = None
        reason = reason or str(error)
    if reason is None:
        from deepopt.opt.backends.triton.packed_multiply import multiply as implementation

        output = implementation(gradient, data, step, minimum, options=options, shape=shape,
            product_dtype=product_dtype, target_shape=target_shape, target_dtype=target_dtype,
            launch_plan=launch)
        if execution_info is not None:
            execution_info.update(route="packed_streaming", selected=True, rejection_reason=None,
                decoded_full_temporary_bytes=0, product_full_temporary_bytes=0,
                scratch_bytes=launch.scratch_bytes, launch_plan=launch.as_dict(),
                output_bytes=output.numel() * output.element_size())
        return output
    if not allow_materialized:
        raise ValueError(f"packed multiplication is unsupported: {reason}")
    from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

    logical = tuple(options["logical_shape"])
    matrix = options.get("matrix_shape") or ((prod(logical[:-1]), logical[-1]) if logical else (1, 1))
    coefficient = dequantize_unpack_activation(PackedActivation(data, step, minimum, matrix,
        options["group_size"], bits=options["bits"]), dtype=options["logical_dtype"], row_alignment=1).reshape(logical)
    product = (gradient * coefficient).to(product_dtype)
    output = product if tuple(shape) == tuple(target_shape) else product.sum_to_size(target_shape)
    output = output.to(target_dtype)
    if execution_info is not None:
        execution_info.update(route="materialized", selected=False, rejection_reason=reason,
            decoded_full_temporary_bytes=coefficient.numel() * coefficient.element_size(),
            product_full_temporary_bytes=product.numel() * product.element_size(), scratch_bytes=0,
            output_bytes=output.numel() * output.element_size())
    return output
