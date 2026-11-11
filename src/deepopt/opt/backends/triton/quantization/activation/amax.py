import torch
import triton
import triton.language as tl

def _bits_consts(bits: int, group_size: int):
    if bits not in(1, 2, 3, 4, 8):
        raise ValueError(f"Unsupported bits: {bits}")
    vpw = 32 // bits
    nwords = group_size // vpw
    qmax = (1 << bits) - 1
    return vpw, nwords, qmax


@triton.jit
def amax_quantize_pack_kernel(
    x_ptr, packed_ptr, scalar_ptr,
    stride_x0, stride_x1,
    stride_p0, stride_p1,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    QMAX: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < GROUP_SIZE

    x = tl.load(x_ptr + pid * stride_x0 + offs * stride_x1, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))

    scalar = QMAX / amax
    scalar = tl.where(amax > 0, scalar, 1.0)
    scalar_inv = 1.0 / scalar

    tl.store(scalar_ptr + pid, scalar_inv.to(tl.float16))

    x_scaled = x * scalar
    x_scaled = tl.where(mask, x_scaled, 0.0)
    x_scaled = tl.round(x_scaled)
    x_scaled = x_scaled.to(tl.int32)
    x_scaled = tl.maximum(tl.minimum(x_scaled, QMAX), 0)

    x_scaled = tl.reshape(x_scaled, (NWORDS, VPW))
    words = tl.sum(x_scaled << (tl.arange(0, VPW) * BITS)[None, :], axis=1)

    words_offs = tl.arange(0, NWORDS)
    words_mask = words_offs < NWORDS
    tl.store(packed_ptr + pid * stride_p0 + words_offs * stride_p1, words, mask=words_mask)



@triton.jit
def amax_dequantize_unpack_kernel(
    packed_ptr, scalar_ptr, x_ptr,
    stride_p0, stride_p1,
    stride_x0, stride_x1,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    QMAX: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    offs = tl.arange(0, NWORDS)
    mask = offs < NWORDS

    packed = tl.load(packed_ptr + pid * stride_p0 + offs * stride_p1, mask=mask, other=0)
    scalar = tl.load(scalar_ptr + pid).to(tl.float32)

    x = (packed[:, None] >> (tl.arange(0, VPW) * BITS)[None, :]) & QMAX
    x = x.to(tl.float32) * scalar
    x = tl.reshape(x, (BLOCK_SIZE,))

    x_offs = tl.arange(0, BLOCK_SIZE)
    x_mask = x_offs < GROUP_SIZE
    tl.store(x_ptr + pid * stride_x0 + x_offs * stride_x1, x.to(OUTPUT_DTYPE), mask=x_mask)





def amax_quantize_pack(x: torch.Tensor, group_size: int, bits: int):
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    ori_shape = x.shape
    x2 = x.view(-1, group_size)
    num_groups = x2.shape[0]

    packed = torch.empty((num_groups, NWORDS), dtype=torch.int32, device=x.device)
    scalar = torch.empty(num_groups, dtype=torch.float16, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(group_size)
    grid = (num_groups,)
    amax_quantize_pack_kernel[grid](
        x2, packed, scalar,
        x2.stride(0), x2.stride(1),
        packed.stride(0), packed.stride(1),
        BITS=bits, VPW=VPW, NWORDS=NWORDS,
        BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size, QMAX=QMAX,
    )

    return packed, scalar, ori_shape



def amax_dequantize_unpack(packed: torch.Tensor, scalar: torch.Tensor, ori_shape, group_size: int, bits: int, output_dtype):
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    x = torch.empty(ori_shape, dtype=output_dtype, device=packed.device)
    x2 = x.view(-1, group_size)
    num_groups = x2.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(group_size)
    grid = (num_groups,)
    amax_dequantize_unpack_kernel[grid](
        packed, scalar, x2,
        packed.stride(0), packed.stride(1),
        x2.stride(0), x2.stride(1),
        BITS=bits, VPW=VPW, NWORDS=NWORDS,
        BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size, QMAX=QMAX,
        OUTPUT_DTYPE=output_dtype,
    )

    return x
