"""Decode, independently rescale, transpose and cast an incoming FP8 gradient."""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_quantize(data, inverse, output, out_inverse,
                        M: tl.constexpr, N: tl.constexpr, RS: tl.constexpr, CS: tl.constexpr,
                        IBR: tl.constexpr, IBC: tl.constexpr, SRS: tl.constexpr, SCS: tl.constexpr,
                        BR: tl.constexpr, BC: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    row = group // triton.cdiv(M, BC) * BR + offset // BC
    col = group % triton.cdiv(M, BC) * BC + offset % BC
    valid = (offset < BR * BC) & (row < N) & (col < M)
    raw = tl.load(data + col * RS + row * CS, valid, other=0).to(tl.float32)
    scale_inv = tl.load(inverse + col // IBR * SRS + row // IBC * SCS, valid, other=0)
    value = raw * scale_inv
    amax = tl.max(tl.abs(value), 0)
    scale = tl.where(amax == 0, 1.0, tl.div_rn(MAXIMUM, tl.maximum(amax, 1e-12)))
    tl.store(out_inverse + group, tl.div_rn(1.0, scale))
    value = tl.minimum(tl.maximum(value * scale, -MAXIMUM), MAXIMUM)
    tl.store(output + row * M + col, value.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)


def transpose_quantize(q, dtype, blocks):
    from deepopt.opt.quantization.tensor import QuantizedTensor

    with torch.cuda.device(q.data.device):
        m, n = q.data.shape
        br, bc = blocks
        data = torch.empty((n, m), dtype=dtype, device=q.data.device)
        inverse = torch.empty((triton.cdiv(n, br), triton.cdiv(m, bc)), dtype=torch.float32, device=q.data.device)
        ibr, ibc = q.block_shape or (m, n)
        _transpose_quantize[(inverse.numel(),)](
            q.data, q.scale_inv, data, inverse, m, n, *q.data.stride(), ibr, ibc, *q.scale_inv.stride(),
            br, bc, torch.finfo(dtype).max, triton.next_power_of_2(br * bc))
    return QuantizedTensor(data, inverse, blocks)
