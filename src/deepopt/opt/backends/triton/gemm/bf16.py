"""BF16 tensor-core GEMMs, ordered epilogues and group-aware producer packing."""

import torch
import triton
import triton.language as tl

from ..activation.math import _evaluate
from ..quantization.activation.affine import _affine_decode, _affine_quantize_pack
from ..._shared.bf16_gemm import (
    BF16GemmConfig, BF16GemmKey, bf16_gemm_candidates, bf16_gemm_selection,
    bf16_source_version, legacy_bf16_config,
)
from ..._shared.gemm_diagnostics import gemm_launch_is_recording, note_gemm_launch
from ..._shared.bf16_resources import launch_admitted


@triton.jit
def _bf16_gemm(A, B, Bias, Y, Saved, DInput, Steps, Minima, Residual, DSteps, DMinima,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               ASM: tl.constexpr, ASK: tl.constexpr, BSN: tl.constexpr, BSK: tl.constexpr,
               BIAS: tl.constexpr, BIAS_STRIDE: tl.constexpr, KIND: tl.constexpr, APPROX: tl.constexpr, SAVE: tl.constexpr,
               DERIVATIVE: tl.constexpr, DAPPROX: tl.constexpr,
               DSM: tl.constexpr, DSN: tl.constexpr,
               DPACKED: tl.constexpr, DGROUP: tl.constexpr, DWORDS: tl.constexpr, DBITS: tl.constexpr,
               RESIDUAL: tl.constexpr, RSM: tl.constexpr, RSN: tl.constexpr,
               PACKED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr,
               BITS: tl.constexpr, SOURCE_N: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               ORDER: tl.constexpr, GROUP_M: tl.constexpr,
               Accum=None, ACCUMULATE: tl.constexpr = False, FINAL: tl.constexpr = True):
    pid = tl.program_id(0).to(tl.int64)
    blocks_m, blocks_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    if ORDER == "column_major":
        block_m, block_n = pid % blocks_m, pid // blocks_m
    elif ORDER == "row_major":
        block_m, block_n = pid // blocks_n, pid % blocks_n
    else:
        span = GROUP_M * blocks_n
        first_m = (pid // span) * GROUP_M
        group_rows = tl.minimum(blocks_m - first_m, GROUP_M)
        group_local = pid % span
        block_m, block_n = first_m + group_local % group_rows, group_local // group_rows
    rows = block_m * BM + tl.arange(0, BM)
    cols = block_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK).to(tl.int64)
    if ACCUMULATE:
        acc = tl.load(Accum + rows[:, None] * N + cols[None, :],
                      (rows[:, None] < M) & (cols[None, :] < N), other=0)
    else:
        acc = tl.full((BM, BN), 0, tl.float32)
    for panel in range(tl.cdiv(K, BK)):
        reduction = panel * BK + kk
        left = tl.load(A + rows[:, None] * ASM + reduction[None, :] * ASK,
                       (rows[:, None] < M) & (reduction[None, :] < K), other=0).to(tl.bfloat16)
        valid = (reduction[:, None] < K) & (cols[None, :] < N)
        if PACKED:
            # The saved X is logically [K,N]; consume X.T without a full decode.
            logical = reduction[:, None].to(tl.int64) * SOURCE_N + cols[None, :]
            groups = logical // GROUP
            within = logical % GROUP
            word = tl.load(B + groups * WORDS + within // (32 // BITS), valid, other=0).to(tl.uint32)
            codes = ((word >> ((within % (32 // BITS)) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
            step = tl.load(Steps + groups, valid, other=0.0)
            minimum = tl.load(Minima + groups, valid, other=0.0)
            right = _affine_decode(codes, step, minimum, BITS, 0.0).to(tl.bfloat16)
            right = tl.where(valid, right, 0.0).to(tl.bfloat16)
        else:
            right = tl.load(B + cols[None, :] * BSN + reduction[:, None] * BSK,
                            valid, other=0).to(tl.bfloat16)
        acc = tl.dot(left, right, acc)
    if not FINAL:
        tl.store(Accum + rows[:, None] * N + cols[None, :], acc,
                 (rows[:, None] < M) & (cols[None, :] < N))
    else:
        if BIAS:
            bias = tl.load(Bias + cols * BIAS_STRIDE, cols < N, other=0).to(tl.bfloat16).to(tl.float32)
            acc = acc + bias[None, :]
        value = acc.to(tl.bfloat16)
        if RESIDUAL:
            residual = tl.load(Residual + rows[:, None] * RSM + cols[None, :] * RSN,
                               (rows[:, None] < M) & (cols[None, :] < N), other=0)
            value = (value.to(tl.float32) + residual.to(tl.float32)).to(tl.bfloat16)
        if SAVE:
            tl.store(Saved + rows[:, None] * N + cols[None, :], value,
                     (rows[:, None] < M) & (cols[None, :] < N))
        if KIND != "none":
            activated, unused = _evaluate(value.to(tl.float32), KIND, APPROX)
            value = activated.to(tl.bfloat16)
        if DERIVATIVE != "none" or DPACKED:
            if DPACKED:
                logical = rows[:, None] * N + cols[None, :]
                groups, within = logical // DGROUP, logical % DGROUP
                valid = (rows[:, None] < M) & (cols[None, :] < N)
                word = tl.load(DInput + groups * DWORDS + within // (32 // DBITS), valid, other=0).to(tl.uint32)
                codes = ((word >> ((within % (32 // DBITS)) * DBITS)) & ((1 << DBITS) - 1)).to(tl.float32)
                local = _affine_decode(codes, tl.load(DSteps + groups, valid, other=0.0),
                                        tl.load(DMinima + groups, valid, other=0.0), DBITS, 0.0)
            else:
                source = tl.load(DInput + rows[:, None] * DSM + cols[None, :] * DSN,
                                 (rows[:, None] < M) & (cols[None, :] < N), other=0).to(tl.float32)
                unused, local = _evaluate(source, DERIVATIVE, DAPPROX)
            value = (value.to(tl.float32) * local).to(tl.bfloat16)
        tl.store(Y + rows[:, None] * N + cols[None, :], value,
                 (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _bf16_prepare(X, Y, Payload, Steps, Minima, DPayload, DSteps, DMinima, Errors, COUNT: tl.constexpr,
                  GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                  KIND: tl.constexpr, APPROX: tl.constexpr, BLOCK: tl.constexpr,
                  SAVE_D: tl.constexpr, CHECK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    offsets = group * GROUP + lane
    valid = (lane < GROUP) & (offsets < COUNT)
    value = tl.load(X + offsets, valid, other=0).to(tl.float32)
    if KIND != "none":
        value, coefficient = _evaluate(value, KIND, APPROX)
    value = value.to(tl.bfloat16)
    tl.store(Y + offsets, value, valid)
    words, step, minimum = _affine_quantize_pack(value.to(tl.float32), valid, BITS, BLOCK)
    indices = tl.arange(0, BLOCK // (32 // BITS))
    tl.store(Payload + group * WORDS + indices, words, indices < WORDS)
    tl.store(Steps + group, step)
    tl.store(Minima + group, minimum)
    if SAVE_D:
        words, step, minimum = _affine_quantize_pack(coefficient, valid, BITS, BLOCK)
        tl.store(DPayload + group * WORDS + indices, words, indices < WORDS)
        tl.store(DSteps + group, step)
        tl.store(DMinima + group, minimum)
        if CHECK:
            bad = valid & (~(tl.abs(coefficient) < float("inf")))
            tl.store(Errors + group, tl.max(bad.to(tl.int32), 0))


def product(a, b, *, bias=None, activation=None, derivative_input=None,
            derivative_activation=None, packed_b=None, tile=64, num_stages=3,
            return_preactivation=False, residual=None, derivative_packed=None,
            kernel_config=None, role="unspecified", packed_route="direct", panel_size=1024,
            _accumulator=None, _accumulate=False, _final=True, _output=None, _route=None, bf16_policy=None):
    if packed_route == "panel":
        from .panel import panel_product
        return panel_product(a, packed_b, panel_size=panel_size, bias=bias, activation=activation,
            derivative_input=derivative_input, derivative_activation=derivative_activation,
            derivative_packed=derivative_packed, residual=residual,
            tile=tile, num_stages=num_stages, kernel_config=kernel_config, role=role,
            return_preactivation=return_preactivation, bf16_policy=bf16_policy)
    if packed_route != "direct":
        raise ValueError("packed_route must be direct or panel")
    if kernel_config is not None and not isinstance(kernel_config, BF16GemmConfig):
        raise TypeError("BF16 kernel_config must be an immutable BF16GemmConfig")
    config = kernel_config if kernel_config is not None else legacy_bf16_config(tile, num_stages)
    m, k = a.shape
    n = b.shape[0] if packed_b is None else packed_b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16) if _output is None else _output
    saved = torch.empty_like(output) if return_preactivation else output
    rhs = b if packed_b is None else packed_b.data
    dummy = output
    with torch.cuda.device(a.device):
        args = (
            a, rhs, bias if bias is not None else dummy, output, saved,
            derivative_packed.data if derivative_packed is not None else derivative_input if derivative_input is not None else dummy,
            packed_b.step if packed_b is not None else dummy,
            packed_b.minimum if packed_b is not None else dummy,
            residual if residual is not None else dummy,
            derivative_packed.step if derivative_packed is not None else dummy,
            derivative_packed.minimum if derivative_packed is not None else dummy,
            m, n, k, *a.stride(), *(b.stride() if b is not None else (0, 0)),
            bias is not None, bias.stride(0) if bias is not None else 1,
            *(activation or ("none", "none")), return_preactivation,
            *(derivative_activation or ("none", "none")),
            *(derivative_input.stride() if derivative_input is not None else (0, 0)),
            derivative_packed is not None, derivative_packed.group_size if derivative_packed is not None else 1,
            derivative_packed.data.shape[1] if derivative_packed is not None else 1,
            derivative_packed.bits if derivative_packed is not None else 2,
            residual is not None, *(residual.stride() if residual is not None else (0, 0)),
            packed_b is not None, packed_b.group_size if packed_b is not None else 1,
            packed_b.data.shape[1] if packed_b is not None else 1,
            packed_b.bits if packed_b is not None else 2, n,
        )
        grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),)
        route = _route or ("direct" if packed_b is not None else "dense")
        arguments = dict(config.kernel_arguments, Accum=_accumulator,
                         ACCUMULATE=_accumulate, FINAL=_final)
        if not m or not n:
            return (output, saved) if return_preactivation else output
        artifact, admission = launch_admitted(_bf16_gemm, grid, args, arguments,
            device=a.device, role=role, route=route, config=config.as_dict())
    if gemm_launch_is_recording():
        selection = bf16_gemm_selection(_selection_key(a, b, packed_b, role,
            bias, activation, return_preactivation, residual, derivative_input,
            derivative_activation, derivative_packed, bf16_policy), kernel_config, tile=tile, num_stages=num_stages)
        candidates = bf16_gemm_candidates(selection.key)
        note_gemm_launch(
            kernel="deepopt.opt.backends.triton.gemm.bf16._bf16_gemm",
            backend="triton", selected_config=config.as_dict(), selection=selection.as_dict(),
            resource_admission=admission, route=route,
            candidate_rejections=[{"config": item.config.as_dict(), "reason": item.rejection_reason}
                                  for item in candidates if item.rejection_reason],
            unmeasured_candidate_count=sum(item.rejection_reason is None for item in candidates),
            resources={"registers_per_thread": admission["registers_per_thread"],
                       "shared_memory_bytes": admission["shared_memory_bytes"],
                       "spills": admission["spills"], "occupancy": None,
                       "explicit_workspace_bytes": 0},
        )
    return (output, saved) if return_preactivation else output


def _selection_key(a, b, packed_b, role, bias, activation, saved, residual,
                   derivative_input, derivative_activation, derivative_packed, bf16_policy=None):
    """Construct diagnostics only on demand; never cache an invocation tensor."""
    from deepopt.kernel_policy import current_kernel_policy
    from deepopt._bf16_policy import resolve_bf16_policy

    def layout(value):
        return None if value is None else (
            tuple(value.shape), tuple(value.stride()), str(value.dtype), value.storage_offset())

    def codec(value, orientation):
        return ("dense",) if value is None else (
            "affine_flat", orientation, value.bits, value.group_size, tuple(value.shape),
            layout(value.data), layout(value.step), layout(value.minimum))

    m, k = a.shape
    n = b.shape[0] if packed_b is None else packed_b.shape[1]
    return BF16GemmKey(role, m, n, k, str(a.device), torch.cuda.get_device_capability(a.device),
        str(a.dtype), str(b.dtype if packed_b is None else packed_b.data.dtype), tuple(a.stride()),
        tuple(b.stride()) if b is not None else (0, 0), codec(packed_b, "source_KN_right_KN"),
        (layout(bias), tuple(activation) if activation is not None else None, saved,
         layout(residual), layout(derivative_input),
         tuple(derivative_activation) if derivative_activation is not None else None,
         codec(derivative_packed, "output_MN")),
        torch.are_deterministic_algorithms_enabled(),
        ("bf16_multiplicands_fp32_accumulation", "ordered_bf16_epilogue_rounding", "fp_fusion_disabled",
         ("allow_bf16_reduced_precision_reduction", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
         ("deterministic_warn_only", torch.is_deterministic_algorithms_warn_only_enabled())),
        current_kernel_policy().fingerprint,
        (str(torch.__version__), torch.version.cuda, str(triton.__version__)), bf16_source_version(),
        a_storage_offset=a.storage_offset(), b_storage_offset=b.storage_offset() if b is not None else 0,
        bf16_policy=tuple(resolve_bf16_policy(bf16_policy).as_dict().items()))


def prepare(value, *, activation, compression, derivative_compression=None):
    from deepopt.opt.quantization.activation import PackedActivation
    from deepopt.opt._execution import finite_checks_enabled

    group, bits = compression["group_size"], compression["bits"]
    groups = triton.cdiv(value.numel(), group)
    words = triton.cdiv(group, 32 // bits)
    payload = torch.empty((groups, words), device=value.device, dtype=torch.int32)
    step = torch.empty(groups, device=value.device, dtype=torch.float32)
    minimum = torch.empty_like(step)
    output = torch.empty_like(value, dtype=torch.bfloat16)
    saved = derivative_compression is not None
    dpayload = torch.empty_like(payload) if saved else payload
    dstep, dminimum = (torch.empty_like(step), torch.empty_like(step)) if saved else (step, minimum)
    checked = saved and finite_checks_enabled()
    errors = torch.empty(groups, device=value.device, dtype=torch.int32) if checked else payload
    block = max(32 // bits, triton.next_power_of_2(group))
    with torch.cuda.device(value.device):
        _bf16_prepare[(groups,)](
            value, output, payload, step, minimum, dpayload, dstep, dminimum, errors,
            value.numel(), group, words, bits, *(activation or ("none", "none")), block, saved, checked,
            num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False,
        )
    if checked and errors.any().item():
        raise ValueError("Activation derivative compression requires finite coefficients")
    result = output, PackedActivation(payload, step, minimum, tuple(value.shape), group, bits=bits)
    return (*result, PackedActivation(dpayload, dstep, dminimum, tuple(value.shape), group, bits=bits)) if saved else result
