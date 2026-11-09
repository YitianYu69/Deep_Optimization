"""Autotuned Triton FP8 GEMM with TMA operand loads."""

import torch
import inspect
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor 

from ..quantization.tensor import QuantizedTensor
from ._cuda_launch import device_metadata
from ._gemm_launch import gemm_launch_plan, selected_config, launch_selected
from ._triton_autotune import (
    AUTOTUNE_KEY, AUTOTUNE_TILES, AUTOTUNE_STAGES, GROUP_SIZE_M_CHOICES, SPLIT_K_CHOICES,
    prune_gemm_configs, split_k_limit, DescriptorKernel,
    grouped_tile_coordinates,
)


_grouped_tile_coordinates = triton.jit(grouped_tile_coordinates)
_cache_options = ({"cache_results": True}
                  if "cache_results" in inspect.signature(triton.autotune).parameters else {})


def _benchmark_gemm(kernel_call, quantiles):
    """Rank short kernels without Python launch gaps or per-call L2 flushing."""
    from triton.testing import do_bench_cudagraph

    # The benchmark uses a separate stream; packing/scale copies on the caller's
    # stream must finish first. Only the already-prepared GEMM is captured.
    torch.cuda.synchronize()
    return do_bench_cudagraph(kernel_call, rep=20, quantiles=quantiles)


@triton.jit
def _reduce_split_k(partials_ptr, output_ptr, bias_ptr, M, N,
                    OUTPUT_STRIDE: tl.constexpr, SPLIT_K: tl.constexpr,
                    HAS_BIAS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows, cols = offsets // N, offsets % N
    splits = tl.arange(0, SPLIT_K)
    addresses = (splits[:, None] * M * OUTPUT_STRIDE
                 + rows[None, :] * OUTPUT_STRIDE + cols[None, :])
    partials = tl.load(partials_ptr + addresses, mask=offsets[None, :] < M * N, other=0.0)
    values = tl.sum(partials, axis=0)
    if HAS_BIAS:
        values += tl.load(bias_ptr + cols, mask=offsets < M * N, other=0.0).to(tl.float32)
    tl.store(output_ptr + rows * OUTPUT_STRIDE + cols, values, mask=offsets < M * N)


def _finish_gemm(args, exception=None):
    """Benchmark post-hook; the wrapper also calls it after the winning launch.

    Triton's autotuner invokes its post-hook only during benchmarking. Including
    the reduction here makes split and unsplit timings directly comparable.
    """
    if exception is None and args["SPLIT_K"] > 1:
        _reduce_split_k[(triton.cdiv(args["M"] * args["N"], 256),)](
            args["partials_ptr"], args["output_ptr"], args["bias_ptr"], args["M"], args["N"],
            OUTPUT_STRIDE=args["OUTPUT_STRIDE"], SPLIT_K=args["SPLIT_K"],
            HAS_BIAS=args["HAS_BIAS"], BLOCK_SIZE=256, num_warps=4, num_stages=1,
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                       "NUM_STAGES": stages, "SPLIT_K": split, "GROUP_SIZE_M": group},
                      num_warps=warps, num_stages=stages)
        for bm, bn, bk, warps in AUTOTUNE_TILES
        for stages in AUTOTUNE_STAGES
        for split in SPLIT_K_CHOICES
        for group in GROUP_SIZE_M_CHOICES
    ],
    key=list(AUTOTUNE_KEY),
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    do_bench=_benchmark_gemm,
    post_hook=_finish_gemm,
    **_cache_options,
)
@DescriptorKernel
@triton.jit
def _gemm_nt_kernel(
    a_desc, b_desc, c_desc,
    a_scale_inv_ptr, b_scale_inv_ptr,
    bias_ptr,
    partials_ptr, output_ptr,
    M, N, K,
    A_BLOCK_ROWS: tl.constexpr, A_BLOCK_COLS: tl.constexpr,
    B_BLOCK_ROWS: tl.constexpr, B_BLOCK_COLS: tl.constexpr,
    A_SCALE_COLS: tl.constexpr, B_SCALE_COLS: tl.constexpr,
    TENSORWISE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    OUTPUT_FORMAT: tl.constexpr,
    A_FP8_FORMAT: tl.constexpr, B_FP8_FORMAT: tl.constexpr,
    DEVICE_INDEX: tl.constexpr, DEVICE_CAPABILITY: tl.constexpr,
    NUM_SMS: tl.constexpr, OUTPUT_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    NUM_STAGES: tl.constexpr, SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_ROW_STRIDE: tl.constexpr, B_ROW_STRIDE: tl.constexpr,
):

    pid_m, pid_n = _grouped_tile_coordinates(
        tl.program_id(0), tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N), GROUP_SIZE_M)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in tl.range(tl.program_id(1), tl.cdiv(K, BLOCK_K), SPLIT_K, num_stages=NUM_STAGES):
        start_k = k_tile * BLOCK_K

        a_tile = a_desc.load([start_m, start_k])
        b_tile = b_desc.load([start_n, start_k])

        if TENSORWISE:
            acc = tl.dot(a_tile, b_tile.T, acc, out_dtype=tl.float32)
        elif NATIVE_FP8:
            partial = tl.dot(a_tile, b_tile.T, out_dtype=tl.float32)

            a_scale_offset = (rows // A_BLOCK_ROWS) * A_SCALE_COLS + start_k // A_BLOCK_COLS
            b_scale_offset = (cols // B_BLOCK_ROWS) * B_SCALE_COLS + start_k // B_BLOCK_COLS

            a_scale_inv = tl.load(a_scale_inv_ptr + a_scale_offset, mask=rows<M, other=0.0)
            b_scale_inv = tl.load(b_scale_inv_ptr + b_scale_offset, mask=cols<N, other=0.0)

            scaled_partial = (partial * a_scale_inv[:, None]) * b_scale_inv[None, :]
            acc = acc + scaled_partial
        else:
            # Transposing rowwise blocks can make scales vary at every K.
            # Apply those scales before reduction without requantizing FP8.
            ks = start_k + tl.arange(0, BLOCK_K)
            a_scale_offset = ((rows // A_BLOCK_ROWS)[:, None] * A_SCALE_COLS
                              + (ks // A_BLOCK_COLS)[None, :])
            b_scale_offset = ((cols // B_BLOCK_ROWS)[:, None] * B_SCALE_COLS
                              + (ks // B_BLOCK_COLS)[None, :])
            a_scales = tl.load(a_scale_inv_ptr + a_scale_offset,
                               mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
            b_scales = tl.load(b_scale_inv_ptr + b_scale_offset,
                               mask=(cols[:, None] < N) & (ks[None, :] < K), other=0.0)
            a_values = a_tile.to(tl.float32) * a_scales
            b_values = b_tile.to(tl.float32) * b_scales
            acc = tl.dot(a_values, b_values.T, acc, out_dtype=tl.float32,
                         input_precision="tf32x3")
    
    if TENSORWISE:
        a_scale_inv = tl.load(a_scale_inv_ptr)
        b_scale_inv = tl.load(b_scale_inv_ptr)
        acc = (acc * a_scale_inv) * b_scale_inv

    if HAS_BIAS and SPLIT_K == 1:
        cols = start_n + tl.arange(0, BLOCK_N)
        bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    
    if SPLIT_K == 1:
        c_desc.store([start_m, start_n], acc.to(OUTPUT_DTYPE))
    else:
        # Each CTA owns its partial tile; every valid element is overwritten on
        # every invocation. No clearing or nondeterministic atomics are needed.
        addresses = (tl.program_id(1) * M * OUTPUT_STRIDE
                     + rows[:, None] * OUTPUT_STRIDE + cols[None, :])
        tl.store(partials_ptr + addresses, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))



@triton.jit
def _pack_rows(source_ptr, target_ptr, ROWS, COLS, PADDED_COLS, STRIDE_M, STRIDE_N,
                BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = offsets // PADDED_COLS, offsets % PADDED_COLS
    values = tl.load(source_ptr + rows * STRIDE_M + cols * STRIDE_N,
                     mask=(rows < ROWS) & (cols < COLS), other=0.0)
    tl.store(target_ptr + offsets, values, mask=offsets < ROWS * PADDED_COLS)


def _pack_fp8_rows(data: torch.Tensor, block_k: int) -> torch.Tensor:
    """
    Make FP8 rows contiguous and pad K with zeros
    """

    rows, cols = data.shape
    padded_cols = triton.cdiv(cols, block_k) * block_k

    if (data.stride(1) == 1 and data.stride(0) >= cols
            and data.stride(0) % block_k == 0 and data.data_ptr() % 16 == 0):
        return data
    
    packed = torch.empty(
        (rows, padded_cols),
        device=data.device, dtype=data.dtype
    )
    # One strided copy/padding kernel writes every byte; aligned copies need
    # neither a preliminary zero-fill nor a separate torch copy launch.
    _pack_rows[(triton.cdiv(rows * padded_cols, 1024),)](
        data, packed, rows, cols, padded_cols, *data.stride(), BLOCK=1024)
    return packed




@torch.no_grad()
def gemm_nt(
    a: QuantizedTensor,
    b: QuantizedTensor,
    *,
    out_dtype: torch.dtype = torch.float32,
    bias: torch.Tensor | None = None,
    num_stages: int | str = "auto",
) -> torch.Tensor:

    """
    Compute A @ B.t with independent FP8 formats and TMA loads/stores.

    Scale blocks aligned along K use native FP8 dot products. Other layouts
    dequantize tiles to FP32 before a TF32x3 dot, retaining the stored FP8
    values and their scales without additional FP8 rounding.
    """
    
    if a.data.device != b.data.device:
        raise ValueError("Operands must be on the same device")
    if a.data.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("This backend requires an Nvidia CUDA GPU")
    device_index = a.data.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability, num_sms = device_metadata(device_index)
    if capability < 90:
        raise ValueError("This backend requires compute cabaility >= 9")
    if a.data.shape[1] != b.data.shape[1]:
        raise ValueError("Operands must have the same reduction dimension")
    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in AUTOTUNE_STAGES):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")
    
    output_types = {
        torch.float32 : tl.float32,
        torch.bfloat16 : tl.bfloat16,
        torch.float16 : tl.float16
    }
    if out_dtype not in output_types:
        raise ValueError("Output dtype must be FP32, BF16, or FP16")

    m, k = a.data.shape
    n, _ = b.data.shape
    
    if bias is not None:
        if bias.device != a.data.device:
            raise ValueError("Bias must be on the operands' device")
        if tuple(bias.shape) != (n,):
            raise ValueError("Bias must have shape [N]")
        if bias.dtype not in output_types:
            raise TypeError("Bias must be FP32, BF16, or FP16")

    with torch.cuda.device(a.data.device):
        # TMA needs 16-byte row strides, not whole-tile storage. Logical
        # descriptor bounds zero-fill edge loads and discard edge stores.
        # All candidates now benchmark exactly the same production layout.
        a_packed = _pack_fp8_rows(a.data, 16)
        b_packed = _pack_fp8_rows(b.data, 16)

        # The kernel indexes scale grids as flat row-major arrays, including
        # when callers supply a valid QuantizedTensor with strided scales.
        a_scale_inv = a.scale_inv.contiguous()
        b_scale_inv = b.scale_inv.contiguous()

        plan = gemm_launch_plan(m, n, k, a.block_shape, b.block_shape,
                                str(a.data.dtype), str(b.data.dtype), str(out_dtype), device_index,
                                capability, num_sms, a_packed.stride(0), b_packed.stride(0), num_stages,
                                str(bias.dtype) if bias is not None else None)
        padded_n = plan.output_stride
        cached = selected_config(_gemm_nt_kernel, plan.cache_key)
        bm, bn, bk = ((cached.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
                      if cached is not None else (64, 64, 32))
        # Fresh descriptors bind this call's storage; cached layouts retain no
        # tensors. Explicit logical shapes avoid creating operand slice views.
        a_desc = TensorDescriptor(a_packed, [m, k], list(a_packed.stride()), [bm, bk])
        b_desc = TensorDescriptor(b_packed, [n, k], list(b_packed.stride()), [bn, bk])

        output_storage = torch.empty(
            (m, padded_n), dtype=out_dtype, device=a.data.device
        )
        output = output_storage[:, :n]
        c_desc = TensorDescriptor(output_storage, [m, n], list(output_storage.stride()), [bm, bn])

        bias_arg = output_storage if bias is None else bias.contiguous()
        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])

        launch_args = {**plan.arguments, "OUTPUT_DTYPE": output_types[out_dtype]}
        # A dummy FP32 pointer keeps the implicit dtype key identical with and
        # without scratch. Cached unsplit winners allocate no partial buffer;
        # cached split winners need only their chosen number of planes.
        splits = cached.kwargs["SPLIT_K"] if cached is not None else split_k_limit(m, n, k)
        partials = (torch.empty((splits, m, padded_n), device=a.data.device, dtype=torch.float32)
                    if splits > 1 else a_scale_inv)
        positional = (a_desc, b_desc, c_desc, a_scale_inv, b_scale_inv,
                      bias_arg, partials, output_storage, m, n, k)
        if cached is None:
            _gemm_nt_kernel[grid](*positional, **launch_args)
            cached = selected_config(_gemm_nt_kernel, plan.cache_key)
            if cached is None:
                raise RuntimeError("Triton winner is missing from the per-key autotune cache")
        else:
            launch_selected(_gemm_nt_kernel, cached, positional, launch_args, grid(cached.kwargs))
        _gemm_nt_kernel.cache[plan.cache_key] = cached
        if cached.kwargs["SPLIT_K"] > 1:
            _finish_gemm({**launch_args, **cached.kwargs,
                          "partials_ptr": partials, "output_ptr": output_storage,
                          "bias_ptr": bias_arg, "M": m, "N": n})
    
    return output
