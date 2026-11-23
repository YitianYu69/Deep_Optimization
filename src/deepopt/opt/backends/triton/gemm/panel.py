"""Explicit bounded K-panel decode. Scratch belongs to this invocation only."""

import torch
from dataclasses import replace
import triton
import triton.language as tl

from ..quantization.activation.affine import _affine_decode
from ..._shared.bf16_resources import launch_admitted, bf16_scratch_requirements
from ..._shared.gemm_diagnostics import note_gemm_launch, gemm_launch_is_recording, observe_bf16_scratch


@triton.jit
def _decode_panel(Data, Step, Minimum, Output, START: tl.constexpr, COUNT: tl.constexpr,
                  N: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr,
                  BITS: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row, col = index // N, index % N
    logical = (START + row) * N + col
    group, within = logical // GROUP, logical % GROUP
    valid = index < COUNT * N
    word = tl.load(Data + group * WORDS + within // (32 // BITS), valid, other=0).to(tl.uint32)
    code = ((word >> ((within % (32 // BITS)) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
    value = _affine_decode(code, tl.load(Step + group, valid, other=0),
                           tl.load(Minimum + group, valid, other=0), BITS, 0.0)
    # Native right operand is [N, panel_K], matching a @ b.T.
    tl.store(Output + col * STRIDE + row, value.to(tl.bfloat16), valid)


def panel_product(a, packed, *, panel_size=1024, **options):
    from .bf16 import product
    from ..._shared.bf16_gemm import legacy_bf16_config

    if packed is None or packed.block_shape is not None:
        raise ValueError("Panel decoding requires a flat packed operand")
    config = options.get("kernel_config") or legacy_bf16_config(options["tile"], options["num_stages"])
    if type(panel_size) is not int or panel_size <= 0 or panel_size % config.block_k:
        raise ValueError("panel_size must be a positive multiple of GEMM block_k")
    k, n = packed.shape
    m = a.shape[0]
    capacity = min(panel_size, k)
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if not m or not n:
        return (output, torch.empty_like(output)) if options.get("return_preactivation") else output
    # K=0 still needs bias/activation epilogues on a zero product.
    if not k:
        return product(a, a.new_empty((n, 0)), **options)
    with torch.cuda.device(a.device):
        decoded = torch.empty((n, capacity), device=a.device, dtype=torch.bfloat16)
        accumulator = torch.empty((m, n), device=a.device, dtype=torch.float32)
        if gemm_launch_is_recording():
            observe_bf16_scratch({"decoded_panel": decoded, "fp32_accumulator": accumulator})
        admissions = []
        for start in range(0, k, panel_size):
            count = min(panel_size, k - start)
            _, admission = launch_admitted(_decode_panel, (triton.cdiv(count * n, 256),),
                (packed.data, packed.step, packed.minimum, decoded),
                dict(START=start, COUNT=count, N=n, GROUP=packed.group_size,
                     WORDS=packed.data.shape[1], BITS=packed.bits, STRIDE=capacity, BLOCK=256,
                     enable_fp_fusion=False), device=a.device, role=options["role"],
                route="panel_decode", config={"panel_size": panel_size})
            admissions.append(admission)
            final = start + count == k
            current = dict(options)
            if not final:
                for key in ("bias", "activation", "derivative_input", "derivative_activation",
                            "derivative_packed", "residual"):
                    current[key] = None
                current["return_preactivation"] = False
            result = product(a[:, start:start + count], decoded[:, :count],
                _accumulator=accumulator, _accumulate=start > 0, _final=final,
                _output=output, _route="panel", **current)
        note_gemm_launch(route="panel", panel_size=panel_size, panel_count=triton.cdiv(k, panel_size),
                         scratch_bytes=decoded.numel() * 2 + accumulator.numel() * 4,
                         scratch_lifetime="invocation_stream_ordered", decode_admissions=admissions,
                         scratch_requirements=bf16_scratch_requirements(m, n, k, route="panel", panel_size=panel_size),
                         qualification="experimental_unmeasured", automatic_promotion=False)
        if gemm_launch_is_recording():
            from .bf16 import _selection_key
            from ..._shared.bf16_gemm import bf16_gemm_selection

            key = _selection_key(a, None, packed, options["role"], options.get("bias"),
                options.get("activation"), options.get("return_preactivation", False),
                options.get("residual"), options.get("derivative_input"),
                options.get("derivative_activation"), options.get("derivative_packed"), options.get("bf16_policy"))
            key = replace(key, decode_route="panel", panel_size=panel_size)
            selection = bf16_gemm_selection(key, options.get("kernel_config"),
                                            tile=options["tile"], num_stages=options["num_stages"])
            note_gemm_launch(selection=selection.as_dict())
    return result
