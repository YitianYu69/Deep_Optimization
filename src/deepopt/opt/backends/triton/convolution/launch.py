"""Pointer-free launch choices and canonical convolution metadata.

Stable choices avoid runtime autotuning inside CUDA graph capture. These are
conservative initial tiles, not performance claims for every convolution.
"""


def _three(values, fill):
    values = tuple(values)
    return (fill,) * (3 - len(values)) + values


def geometry_arguments(geometry):
    """Normalize one-, two-, and three-dimensional spatial indices to D/H/W."""
    if geometry.rank not in (1, 2, 3):
        raise ValueError("Convolution rank must be 1, 2, or 3")
    inputs = _three(geometry.input_spatial, 1)
    valid_inputs = _three(geometry.valid_input_spatial, 1)
    outputs = _three(geometry.output_spatial, 1)
    kernel = _three(geometry.kernel_size, 1)
    stride = _three(geometry.stride, 1)
    padding = _three(geometry.padding, 0)
    dilation = _three(geometry.dilation, 1)
    return {
        "BATCH": geometry.batch_size,
        "ID": inputs[0], "IH": inputs[1], "IW": inputs[2],
        "VALID_I0": valid_inputs[0], "VALID_I1": valid_inputs[1], "VALID_I2": valid_inputs[2],
        "OD": outputs[0], "OH": outputs[1], "OW": outputs[2],
        "KD": kernel[0], "KH": kernel[1], "KW": kernel[2],
        "SD": stride[0], "SH": stride[1], "SW": stride[2],
        "PD": padding[0], "PH": padding[1], "PW": padding[2],
        "DD": dilation[0], "DH": dilation[1], "DW": dilation[2],
        "CI": geometry.in_channels_per_group,
        "CO": geometry.out_channels_per_group,
        "GROUPS": geometry.groups,
    }


def operand_arguments(prefix, operand):
    """Preserve real data and scale strides, including transposed scale grids."""
    rows, cols = operand.data.shape
    block_rows, block_cols = operand.block_shape or (rows, cols)
    return {
        prefix + "_RS": operand.data.stride(0),
        prefix + "_CS": operand.data.stride(1),
        prefix + "_BR": block_rows,
        prefix + "_BC": block_cols,
        prefix + "_SRS": operand.scale_inv.stride(0),
        prefix + "_SCS": operand.scale_inv.stride(1),
    }


def needs_int64(a, b, geometry, *, extra_tensors=()):
    """Choose wide indices from shapes/strides alone, without device reads.

    Contiguous size alone misses strided operands and metadata grids. Spatial
    intermediates can also be large when padding or dilation is unusually
    large, even if the final valid activation allocation is small.
    """
    limit = (1 << 31) - 1
    tensors = (a.data, a.scale_inv, b.data, b.scale_inv, *extra_tensors)
    if any(sum((size - 1) * abs(stride) for size, stride in zip(tensor.shape, tensor.stride())) >= limit
           for tensor in tensors):
        return True
    if max(geometry.batch_size * geometry.input_volume * geometry.in_channels,
           geometry.batch_size * geometry.output_volume * geometry.out_channels,
           geometry.out_channels * geometry.in_channels_per_group * geometry.kernel_volume) >= limit:
        return True
    return any(max(input_size, output_size * stride) + padding + kernel * dilation >= limit
               for input_size, output_size, stride, padding, kernel, dilation in
               zip(geometry.input_spatial, geometry.output_spatial, geometry.stride,
                   geometry.padding, geometry.kernel_size, geometry.dilation))


def launch_options(num_stages="auto", *, tensorwise=True):
    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in (1, 2, 3, 4, 5)):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")
    return {
        # Hopper WGMMA requires M divisible by 64. An M=32 FP8 dot falls
        # back to MMA v2 and FP16 operand promotion on SM90, losing the native
        # FP8 compute path even though its source pointers still contain FP8.
        "BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64 if tensorwise else 32,
        "num_warps": 4, "num_stages": 2 if num_stages == "auto" else num_stages,
    }


def weight_splits(rows, cols, reduction, groups, *, block_m=64, block_n=32):
    """Bound deterministic dW partial storage and expose long reductions.

    Small kernels otherwise launch too few CTAs for a large spatial batch.
    Partial buffers contain only weight-gradient tiles, never image patches.
    """
    tiles = ((rows + block_m - 1) // block_m) * ((cols + block_n - 1) // block_n) * groups
    if reduction < 2048 or tiles >= 128:
        return 1
    desired = min(16, (128 + tiles - 1) // tiles, (reduction + 1023) // 1024)
    memory_bound = max(1, (32 * 1024 * 1024) // (4 * rows * cols * groups))
    return min(desired, memory_bound)
