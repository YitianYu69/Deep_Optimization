"""Scale-aware FP8 bias-gradient reduction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bias_partial(data, scales, partial, M: tl.constexpr, N: tl.constexpr,
                  RS: tl.constexpr, CS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                  SRS: tl.constexpr, SCS: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    c, part = pid // PARTS, pid % PARTS
    row = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(data + row * RS + c * CS, row < M, other=0).to(tl.float32)
    inverse = tl.load(scales + row // BR * SRS + c // BC * SCS, row < M, other=0)
    tl.store(partial + c * PARTS + part, tl.sum(value * inverse, 0))


@triton.jit
def _bias_sum_chunks(partial, output, COUNT: tl.constexpr, NEXT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    c, part = pid // NEXT, pid % NEXT
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(partial + c * COUNT + offsets, offsets < COUNT, other=0)
    tl.store(output + c * NEXT + part, tl.sum(values, 0))


@triton.jit
def _bias_finish(partial, output, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partial + tl.program_id(0) * PARTS + offsets, offsets < PARTS, other=0)
    tl.store(output + tl.program_id(0), tl.sum(values, 0))


def bias(q):
    with torch.cuda.device(q.data.device):
        m, n = q.data.shape
        parts = triton.cdiv(m, 4096)
        partial = torch.empty(n, parts, device=q.data.device, dtype=torch.float32)
        output = torch.empty(n, device=q.data.device, dtype=torch.float32)
        br, bc = q.block_shape or (m, n)
        _bias_partial[(n * parts,)](q.data, q.scale_inv, partial, m, n, *q.data.stride(), br, bc,
                                  *q.scale_inv.stride(), parts, 4096)
        while parts > 1024:
            next_parts = triton.cdiv(parts, 256)
            reduced = torch.empty(n, next_parts, device=q.data.device, dtype=torch.float32)
            _bias_sum_chunks[(n * next_parts,)](partial, reduced, parts, next_parts, 256)
            partial, parts = reduced, next_parts
        _bias_finish[(n,)](partial, output, parts, triton.next_power_of_2(parts))
    return output
