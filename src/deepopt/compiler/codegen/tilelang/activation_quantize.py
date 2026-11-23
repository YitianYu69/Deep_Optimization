"""GELU/SiLU producers fused with current-scale FP8 conversion and saving.

The activation is recomputed inside the statistics and cast/codec kernels;
there is no materialized dense activation. Reduction workspaces are bounded by
one FP32 value per 1024 inputs. All kernels use the selected TileLang runtime.
"""

from functools import lru_cache

import torch

from ..source import load_source, write_source
from .expressions import expression


def _activated(activation, dtype, value):
    result = expression(activation[0], [value], {"approximate": repr(activation[1])})
    return f'T.cast(T.cast({result}, "{dtype}"), "float32")'


def _converted(activation, dtype, fp8, value, maximum):
    activated = _activated(activation, dtype, value)
    scaled = f"({activated} * scale[0])"
    clamped = f"T.if_then_else({activated} != {activated}, {activated}, T.min(T.max({scaled}, -{maximum!r}), {maximum!r}))"
    return f'T.cast({clamped}, "{fp8}")'


def generate_preparation_source(rows, cols, dtype, fp8, activation, *, paired=False, group=0, bits=2):
    count = rows * cols
    chunks = (count + 1023) // 1024
    padded_cols, padded_rows = (cols + 15) // 16 * 16, (rows + 15) // 16 * 16
    maximum = float(torch.finfo(getattr(torch, fp8)).max)
    active = _activated(activation, dtype, "x[offset]")
    converted = _converted(activation, dtype, fp8, "x[offset]", maximum)
    lines = [
        "def build(device):",
        "    from deepopt.opt.backends.tilelang._runtime import runtime, compile_kernel",
        "    _, T = runtime()",
        "    @T.prim_func",
        f'    def statistics(x: T.Tensor(({count},), "{dtype}"), out: T.Tensor(({chunks},), "float32")):',
        f"        with T.Kernel({(chunks + 31) // 32}, threads=32) as block:",
        "            for lane in T.Parallel(32):",
        '                chunk = T.cast(block, "int64") * 32 + lane',
        '                amax = T.alloc_local((1,), "float32")',
        "                amax[0] = 0.0",
        f"                if chunk < {chunks}:",
        "                    for i in T.serial(1024):",
        "                        offset = chunk * 1024 + i",
        f"                        if offset < {count}:",
        f"                            value = {active}",
        "                            amax[0] = T.if_then_else(amax[0] != amax[0], amax[0], T.if_then_else(value != value, value, T.max(amax[0], T.abs(value))))",
        "                    out[chunk] = amax[0]",
        "    @T.prim_func",
    ]
    if group:
        vpw, words, qmax = 32 // bits, (group + 32 // bits - 1) // (32 // bits), (1 << bits) - 1
        groups = (count + group - 1) // group
        lines.extend((
            f'    def cast(x: T.Tensor(({count},), "{dtype}"), scale: T.Tensor((1,), "float32"), normal: T.Tensor(({rows * padded_cols},), "{fp8}"), payload: T.Tensor(({groups * words},), "int32"), steps: T.Tensor(({groups},), "float32"), minima: T.Tensor(({groups},), "float32")):',
            f"        with T.Kernel({(groups + 31) // 32}, threads=32) as block:",
            "            for lane in T.Parallel(32):",
            '                g = T.cast(block, "int64") * 32 + lane',
            '                low = T.alloc_local((1,), "float32")',
            '                high = T.alloc_local((1,), "float32")',
            '                word = T.alloc_local((1,), "uint32")',
            '                low[0] = T.infinity("float32")',
            '                high[0] = -T.infinity("float32")',
            f"                if g < {groups}:",
            f"                    for i in T.serial({group}):",
            f"                        offset = g * {group} + i",
            f"                        if offset < {count}:",
            f"                            rounded = {converted}",
            f"                            normal[offset // {cols} * {padded_cols} + offset % {cols}] = rounded",
            '                            value = T.cast(rounded, "float32")',
            "                            low[0] = T.min(low[0], value)",
            "                            high[0] = T.max(high[0], value)",
            f"                            if offset % {cols} == {cols - 1}:",
            f"                                for padding in T.serial({padded_cols - cols}):",
            f'                                    normal[offset // {cols} * {padded_cols} + {cols} + padding] = T.cast(0, "{fp8}")',
            f'                    step = T.ieee_fdiv(high[0] - low[0], T.float32({qmax}))',
            "                    steps[g] = step",
            "                    minima[g] = low[0]",
            f"                    for w in T.serial({words}):",
            "                        word[0] = T.uint32(0)",
            f"                        for digit in T.serial({vpw}):",
            f"                            i = w * {vpw} + digit",
            f"                            offset = g * {group} + i",
            f"                            if i < {group} and offset < {count}:",
            f'                                value = T.cast({converted}, "float32")',
            "                                normalized = T.if_then_else(step > 0, T.ieee_fdiv(value - low[0], step), 0.0)",
            f"                                clipped = T.min(T.max(normalized, 0.0), T.float32({qmax}))",
            "                                lower = T.floor(clipped)",
            '                                integer = T.cast(lower, "int32")',
            "                                fraction = clipped - lower",
            '                                code = integer + T.cast((fraction > 0.5) | ((fraction == 0.5) & ((integer & 1) != 0)), "int32")',
            f'                                word[0] = word[0] | (T.cast(code, "uint32") << (digit * {bits}))',
            f'                        payload[g * {words} + w] = T.cast(word[0], "int32")',
        ))
    else:
        total = max(rows * padded_cols, cols * padded_rows if paired else 0)
        signature = f'x: T.Tensor(({count},), "{dtype}"), scale: T.Tensor((1,), "float32"), normal: T.Tensor(({rows * padded_cols},), "{fp8}")'
        if paired:
            signature += f', transposed: T.Tensor(({cols * padded_rows},), "{fp8}")'
        lines.extend((
            f"    def cast({signature}):",
            f"        with T.Kernel({(total + 127) // 128}, threads=128) as block:",
            "            for lane in T.Parallel(128):",
            '                pos = T.cast(block, "int64") * 128 + lane',
            f"                if pos < {rows * padded_cols}:",
            f"                    offset = pos // {padded_cols} * {cols} + pos % {padded_cols}",
            f"                    if pos % {padded_cols} < {cols}:",
            f"                        normal[pos] = {converted}",
            "                    else:",
            f'                        normal[pos] = T.cast(0, "{fp8}")',
        ))
        if paired:
            lines.extend((
                f"                if pos < {cols * padded_rows}:",
                f"                    offset = pos % {padded_rows} * {cols} + pos // {padded_rows}",
                f"                    if pos % {padded_rows} < {rows}:",
                f"                        transposed[pos] = {converted}",
                "                    else:",
                f'                        transposed[pos] = T.cast(0, "{fp8}")',
            ))
    lines.extend(("    return compile_kernel(statistics, device), compile_kernel(cast, device)", ""))
    return "\n".join(lines)


@lru_cache(maxsize=128)
def _kernels(rows, cols, dtype, fp8, activation, paired, group, bits, device):
    source = generate_preparation_source(rows, cols, dtype, fp8, activation, paired=paired, group=group, bits=bits)
    module = load_source(write_source(source))
    return module.build(device)


def prepare(x, *, activation, fp8_dtype, paired, compression):
    from deepopt.opt.backends.tilelang._runtime import require_cuda, dtype_name
    from deepopt.opt.backends.tilelang.quantization.fp8 import amax_scale
    from deepopt.opt.quantization.tensor import QuantizedTensor
    from deepopt.opt.quantization.activation import PackedActivation
    from deepopt.opt._execution import finite_checks_enabled

    device = require_cuda(x)
    if x.ndim != 2 or not x.is_contiguous() or not x.numel():
        raise ValueError("Fused activation preparation requires a nonempty contiguous matrix")
    if activation[0] not in ("gelu", "silu") or activation[1] not in ("none", "tanh"):
        raise ValueError("Unsupported fused activation")
    group, bits = (compression["group_size"], compression["bits"]) if compression is not None else (0, 2)
    if compression is not None and (type(group) is not int or not 0 < group <= 16384 or bits not in (1, 2, 4, 8)):
        raise ValueError("Fused packing requires 1/2/4/8 bits and groups up to 16384")
    rows, cols = x.shape
    with torch.cuda.device(device):
        statistics, convert = _kernels(rows, cols, dtype_name(x.dtype), dtype_name(fp8_dtype), tuple(activation), bool(paired and not group), group, bits, str(device))
        partial = torch.empty(((x.numel() + 1023) // 1024, 1), device=device, dtype=torch.float32)
        statistics(x.view(-1), partial.view(-1))
        scale, inverse = amax_scale(partial, torch.finfo(fp8_dtype).max, None, finite_checks_enabled())
        normal = torch.empty((rows, (cols + 15) // 16 * 16), device=device, dtype=fp8_dtype)
        transposed, packed = None, None
        if group:
            groups, words = (x.numel() + group - 1) // group, (group + 32 // bits - 1) // (32 // bits)
            payload = torch.empty((groups, words), device=device, dtype=torch.int32)
            steps = torch.empty(groups, device=device, dtype=torch.float32)
            minima = torch.empty_like(steps)
            convert(x.view(-1), scale.view(-1), normal.view(-1), payload.view(-1), steps, minima)
            packed = PackedActivation(payload, steps, minima, tuple(x.shape), group, None, bits)
        elif paired:
            transposed = torch.empty((cols, (rows + 15) // 16 * 16), device=device, dtype=fp8_dtype)
            convert(x.view(-1), scale.view(-1), normal.view(-1), transposed.view(-1))
            transposed = QuantizedTensor(transposed[:, :rows], inverse)
        else:
            convert(x.view(-1), scale.view(-1), normal.view(-1))
    return QuantizedTensor(normal[:, :cols], inverse), transposed, packed
