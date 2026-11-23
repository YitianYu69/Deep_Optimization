"""Implicit FP8 convolution in TileLang, with exact source-scale ownership.

The initial schedule contracts one logical reduction coordinate at a time.
This is intentionally conservative: every dot operand remains the stored FP8
value, and each partial is scaled before FP32 accumulation, even when scale
rectangles cut through spatial gathers. It allocates no unfolded activation.
"""

from functools import lru_cache
from math import prod
import torch

from .._runtime import runtime, compile_kernel, dtype_name, require_cuda


def _three(values, fill):
    return (fill,) * (3 - len(values)) + tuple(values)


def depthwise_eligible(geometry):
    return geometry.groups == geometry.in_channels == geometry.out_channels and geometry.kernel_volume > 1


@lru_cache(maxsize=256)
def _kernel(mode, a_shape, b_shape, a_dtype, b_dtype, a_blocks, b_blocks, geometry, dtype, has_bias, device):
    _, T = runtime()
    batch, ci, co, groups, inputs, valid_inputs, outputs, kernel_size, strides, padding, dilations = geometry
    iz, iy, ix = inputs
    vz, vy, vx = valid_inputs
    oz, oy, ox = outputs
    kz, ky, kx = kernel_size
    sz, sy, sx = strides
    pz, py, px = padding
    dz, dy, dx = dilations
    iv, ov, kv = prod(inputs), prod(outputs), prod(kernel_size)
    abr, abc = a_blocks
    bbr, bbc = b_blocks
    ashape = ((a_shape[0] + abr - 1) // abr, (a_shape[1] + abc - 1) // abc)
    bshape = ((b_shape[0] + bbr - 1) // bbr, (b_shape[1] + bbc - 1) // bbc)
    rows = co if mode == "dw" else batch * (iv if mode == "dx" else ov)
    cols = ci * kv if mode == "dw" else ci if mode == "dx" else co
    reduction = batch * ov if mode == "dw" else co * kv if mode == "dx" else ci * kv
    outshape = (groups * co, ci * kv) if mode == "dw" else (rows, groups * cols)
    bias_size = groups * cols
    tensorwise = prod(ashape) == 1 and prod(bshape) == 1

    @T.prim_func
    def execute(A: T.Tensor(a_shape, a_dtype), B: T.Tensor(b_shape, b_dtype),
                AS: T.Tensor(ashape, "float32"), BS: T.Tensor(bshape, "float32"),
                Bias: T.Tensor((bias_size,), "float32"), Y: T.Tensor(outshape, dtype)):
        with T.Kernel(T.ceildiv(rows, 16) * T.ceildiv(cols, 16) * groups, threads=32) as block:
            bid = T.cast(block, "int64")
            tiles_m, tiles_n = T.ceildiv(rows, 16), T.ceildiv(cols, 16)
            group = bid // (tiles_m * tiles_n)
            bm, bn = bid // tiles_n % tiles_m, bid % tiles_n
            a = T.alloc_shared((16, 32), a_dtype)
            b = T.alloc_shared((16, 32), b_dtype)
            sa = T.alloc_shared((16,), "float32")
            sb = T.alloc_shared((16,), "float32")
            partial = T.alloc_fragment((16, 16), "float32")
            total = T.alloc_fragment((16, 16), "float32")
            T.clear(total)
            for red in T.serial(reduction):
                for i, j in T.Parallel(16, 32):
                    m, n = bm * 16 + i, bn * 16 + i
                    if mode == "forward":
                        ch, kp = red // kv, red % kv
                        ozp, oyp, oxp = (m % ov) // (oy * ox), (m % ov) // ox % oy, m % ox
                        zp = ozp * sz - pz + (kp // (ky * kx)) * dz
                        yp = oyp * sy - py + (kp // kx % ky) * dy
                        xp = oxp * sx - px + (kp % kx) * dx
                        ar = (m // ov * iz + zp) * iy * ix + yp * ix + xp
                        ac = group * ci + ch
                        br, bc = group * co + n, red
                        av = (m < rows) & (zp >= 0) & (zp < vz) & (yp >= 0) & (yp < vy) & (xp >= 0) & (xp < vx)
                        bv = n < cols
                    elif mode == "dx":
                        ch, kp = red // kv, red % kv
                        izp, iyp, ixp = (m % iv) // (iy * ix), (m % iv) // ix % iy, m % ix
                        zp = izp + pz - (kp // (ky * kx)) * dz
                        yp = iyp + py - (kp // kx % ky) * dy
                        xp = ixp + px - (kp % kx) * dx
                        ar = (m // iv * oz + zp // sz) * oy * ox + yp // sy * ox + xp // sx
                        ac = group * co + ch
                        br, bc = group * co + ch, n * kv + kp
                        av = (m < rows) & (izp < vz) & (iyp < vy) & (ixp < vx) & (zp >= 0) & (zp < oz * sz) & (zp % sz == 0) & (yp >= 0) & (yp < oy * sy) & (yp % sy == 0) & (xp >= 0) & (xp < ox * sx) & (xp % sx == 0)
                        bv = n < cols
                    else:
                        ch, kp = n // kv, n % kv
                        ozp, oyp, oxp = (red % ov) // (oy * ox), (red % ov) // ox % oy, red % ox
                        zp = ozp * sz - pz + (kp // (ky * kx)) * dz
                        yp = oyp * sy - py + (kp // kx % ky) * dy
                        xp = oxp * sx - px + (kp % kx) * dx
                        ar, ac = red, group * co + m
                        br = (red // ov * iz + zp) * iy * ix + yp * ix + xp
                        bc = group * ci + ch
                        av = m < rows
                        bv = (n < cols) & (zp >= 0) & (zp < vz) & (yp >= 0) & (yp < vy) & (xp >= 0) & (xp < vx)
                    a[i, j] = T.cast(0, a_dtype)
                    b[i, j] = T.cast(0, b_dtype)
                    if j == 0:
                        sa[i] = 0.0
                        sb[i] = 0.0
                        if av:
                            a[i, 0] = A[ar, ac]
                            sa[i] = AS[ar // abr, ac // abc]
                        if bv:
                            b[i, 0] = B[br, bc]
                            sb[i] = BS[br // bbr, bc // bbc]
                T.gemm(a, b, partial, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(16, 16):
                    if tensorwise:
                        total[i, j] += partial[i, j]
                    else:
                        total[i, j] += partial[i, j] * sa[i] * sb[j]
            for i, j in T.Parallel(16, 16):
                m, n = bm * 16 + i, bn * 16 + j
                if (m < rows) & (n < cols):
                    value = total[i, j] * AS[0, 0] * BS[0, 0] if tensorwise else total[i, j]
                    if mode == "dw":
                        Y[group * co + m, n] = T.cast(value, dtype)
                    elif has_bias:
                        Y[m, group * cols + n] = T.cast(value + Bias[group * cols + n], dtype)
                    else:
                        Y[m, group * cols + n] = T.cast(value, dtype)

    return compile_kernel(execute, device)


def _execute(a, b, geometry, mode, out_dtype, bias=None, output=None):
    device = require_cuda(a.data, b.data, a.scale_inv, b.scale_inv, bias)
    if torch.version.hip is not None or torch.cuda.get_device_capability(device) < (9, 0):
        raise ValueError("TileLang FP8 convolution requires an NVIDIA CUDA GPU with compute capability >= 9")
    if a.data.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or b.data.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("TileLang convolution operands must contain FP8 values")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Convolution output dtype must be FP32, BF16, or FP16")
    expected_a = geometry.input_matrix_shape if mode == "forward" else geometry.grad_matrix_shape
    expected_b = geometry.input_matrix_shape if mode == "dw" else geometry.weight_matrix_shape
    if tuple(a.data.shape) != expected_a or tuple(b.data.shape) != expected_b:
        raise ValueError("FP8 operands do not match convolution geometry")
    expected_bias = geometry.in_channels if mode == "dx" else geometry.out_channels
    if bias is not None:
        if tuple(bias.shape) != (expected_bias,):
            raise ValueError("Convolution bias must match the output channels")
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Convolution bias must have FP32, BF16, or FP16 dtype")
    from ..activation.common import cast
    geom = (geometry.batch_size, geometry.in_channels_per_group, geometry.out_channels_per_group,
        geometry.groups, _three(geometry.input_spatial, 1), _three(geometry.valid_input_spatial, 1),
        _three(geometry.output_spatial, 1), _three(geometry.kernel_size, 1), _three(geometry.stride, 1),
        _three(geometry.padding, 0), _three(geometry.dilation, 1))
    shape = geometry.weight_matrix_shape if mode == "dw" else geometry.input_matrix_shape if mode == "dx" else geometry.grad_matrix_shape
    dtype = torch.float32 if output is not None else out_dtype
    result = torch.empty(shape, device=a.data.device, dtype=dtype)
    bias_size = geometry.in_channels if mode == "dx" else geometry.out_channels if mode == "forward" else geometry.groups * geometry.in_channels_per_group * geometry.kernel_volume
    bias_data = cast(bias, torch.float32).contiguous() if bias is not None else torch.empty(bias_size, device=a.data.device)
    if result.numel():
        _kernel(mode, tuple(a.data.shape), tuple(b.data.shape), dtype_name(a.data.dtype), dtype_name(b.data.dtype),
            a.block_shape or tuple(a.data.shape), b.block_shape or tuple(b.data.shape), geom,
            dtype_name(dtype), bias is not None, str(a.data.device))(
                a.data.contiguous(), b.data.contiguous(), a.scale_inv.contiguous(), b.scale_inv.contiguous(), bias_data, result)
    return output.convert(result, matrix=True) if output is not None else result


def forward(qx, qw, geometry, *, bias=None, out_dtype=torch.float32, num_stages="auto", gradient_output=None, autotune=True):
    return _execute(qx, qw, geometry, "forward", out_dtype, bias, gradient_output)


def input_gradient(qg, qw, geometry, *, out_dtype=torch.float32, num_stages="auto", gradient_output=None, bias=None, autotune=True):
    return _execute(qg, qw, geometry, "dx", out_dtype, bias, gradient_output)


def weight_gradient(qx, qg, geometry, *, out_dtype=torch.float32, num_stages="auto", autotune=True):
    return _execute(qg, qx, geometry, "dw", out_dtype)


def execution_mode(*operands):
    return "native_fp8"
