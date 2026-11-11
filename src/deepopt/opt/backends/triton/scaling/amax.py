import torch

import triton
import triton.language as tl



@triton.jit
def block_amax_kernel(
    x_ptr, amax_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_COLS_BLOCKS: tl.constexpr,
):

    pid = tl.program_id(0)

    row = pid // NUM_COLS_BLOCKS
    col = pid % NUM_COLS_BLOCKS

    row_offsets = row * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + row_offsets[:, None] * N + col_offsets[None, :]
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-12)

    tl.store(amax_ptr + pid, amax)


def block_amax(x, block_row, block_col):
    M, N = x.shape

    num_row_blocks = triton.cdiv(M, block_row)
    num_col_blocks = triton.cdiv(N, block_col)

    amax = torch.empty(
        (num_row_blocks, num_col_blocks),
        dtype=torch.float32,
        device=x.device,
    )

    grid = (num_row_blocks * num_col_blocks,)
    block_amax_kernel[grid](
        x, amax,
        M, N,
        BLOCK_M=block_row,
        BLOCK_N=block_col,
        NUM_COLS_BLOCKS=num_col_blocks,
    )

    return amax




@triton.jit
def block_amax_scalar_kernel(
    x_ptr, scalar_ptr, scalar_inv_ptr,
    M, N,
    FP8_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_COL_BLOCKS: tl.constexpr
):
    pid = tl.program_id(0)

    row = pid // NUM_COL_BLOCKS
    col = pid % NUM_COL_BLOCKS

    row_offsets = row * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + row_offsets[:, None] * N + col_offsets[None, :]
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    scalar = tl.where(amax == 0, 1.0, FP8_MAX / tl.maximum(amax, 1e-12))
    scalar_inv = 1.0 / scalar

    tl.store(scalar_ptr + pid, scalar)
    tl.store(scalar_inv_ptr + pid, scalar_inv)



def block_amax_scalar(x, block_row, block_col, fp8_max):
    # Preserve the legacy helper while sharing strided, bounded reductions.
    from deepopt.opt.backends.triton.quantization.fp8 import amax_scale

    return amax_scale(x, fp8_max, (block_row, block_col), False)




@triton.jit
def global_amax_kernel(
    x_ptr, amax_ptr,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    tl.atomic_max(amax_ptr, amax)


def global_amax_scalar(x, fp8_max):
    from deepopt.opt.backends.triton.quantization.fp8 import amax_scale

    scalar, scalar_inv = amax_scale(x, fp8_max, None, False)
    return scalar.reshape(1), scalar_inv.reshape(1)
