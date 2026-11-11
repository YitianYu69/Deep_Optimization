"""FP8 weight gradients segmented at gathered-source scale boundaries."""

import triton
import triton.language as tl

from .common import _load_operand, _load_scale, _scale_chunk_length, _coordinates, _tile_coordinates


@triton.jit
def _blockwise_weight_gradient_kernel(
    a, b, a_scale, b_scale, output,
    BATCH: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    VALID_I0: tl.constexpr, VALID_I1: tl.constexpr, VALID_I2: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    DD: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    CI: tl.constexpr, CO: tl.constexpr, GROUPS: tl.constexpr,
    A_RS: tl.constexpr, A_CS: tl.constexpr, A_BR: tl.constexpr, A_BC: tl.constexpr,
    A_SRS: tl.constexpr, A_SCS: tl.constexpr,
    B_RS: tl.constexpr, B_CS: tl.constexpr, B_BR: tl.constexpr, B_BC: tl.constexpr,
    B_SRS: tl.constexpr, B_SCS: tl.constexpr,
    TENSORWISE: tl.constexpr, INDEX64: tl.constexpr, SPLIT_K: tl.constexpr, STRIPE_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    kernel_volume: tl.constexpr = KD * KH * KW
    stripes_per_row: tl.constexpr = tl.cdiv(OW, STRIPE_W)
    tap_group, pm, pn = _tile_coordinates(CO, CI, BLOCK_M, BLOCK_N)
    if INDEX64:
        tap_group, pm, pn = tap_group.to(tl.int64), pm.to(tl.int64), pn.to(tl.int64)
    # Fold taps into grid X instead of imposing CUDA's smaller grid-Z limit
    # on otherwise valid large spatial kernels.
    group, kernel_pos = tap_group // kernel_volume, tap_group % kernel_volume
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    channels_out, channels_in = group * CO + rows, group * CI + cols
    kd, kh, kw = _coordinates(kernel_pos, KH, KW)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # A CTA owns one spatial kernel tap. Within an output-width stripe both
    # matrix-row addresses advance monotonically, making scale boundaries
    # computable without scanning scales or constructing image patches.
    for stripe_index in range(tl.program_id(1), BATCH * OD * OH * stripes_per_row, SPLIT_K):
        if INDEX64:
            stripe = stripe_index.to(tl.int64)
        else:
            stripe = stripe_index
        output_row = stripe // stripes_per_row
        batch = output_row // (OD * OH)
        od, oh = output_row // OH % OD, output_row % OH
        id_, ih = od * SD - PD + kd * DD, oh * SH - PH + kh * DH
        # Clip padding out before segmentation. Structural SAME right-padding
        # is also excluded, even if an affine decoder reconstructed it nonzero.
        start = tl.maximum(stripe % stripes_per_row * STRIPE_W,
                           tl.maximum(0, tl.cdiv(PW - kw * DW, SW)))
        end = tl.minimum(tl.minimum(OW, (stripe % stripes_per_row + 1) * STRIPE_W),
                         tl.maximum(0, tl.cdiv(VALID_I2 + PW - kw * DW, SW)))
        if (id_ >= 0) & (id_ < VALID_I0) & (ih >= 0) & (ih < VALID_I1):
            while start < end:
                grad_row = output_row * OW + start
                source_row = batch * (ID * IH * IW) + (id_ * IH + ih) * IW + start * SW - PW + kw * DW
                length = _scale_chunk_length(grad_row, 1, A_BR, source_row, SW, B_BR,
                                             tl.minimum(BLOCK_K, end - start))
                offsets = tl.arange(0, BLOCK_K)
                av = _load_operand(a, grad_row + offsets[None, :], channels_out[:, None],
                                   (rows[:, None] < CO) & (offsets[None, :] < length), A_RS, A_CS)
                bv = _load_operand(b, source_row + offsets[:, None] * SW, channels_in[None, :],
                                   (offsets[:, None] < length) & (cols[None, :] < CI), B_RS, B_CS)
                partial = tl.dot(av, bv, out_dtype=tl.float32)
                sa = _load_scale(a_scale, grad_row, channels_out, rows < CO, A_BR, A_BC, A_SRS, A_SCS)
                sb = _load_scale(b_scale, source_row, channels_in, cols < CI, B_BR, B_BC, B_SRS, B_SCS)
                acc += (partial * sa[:, None]) * sb[None, :]
                start += length

    tl.store(output + tl.program_id(1) * (GROUPS * CO * CI * kernel_volume)
             + channels_out[:, None] * (CI * kernel_volume) + cols[None, :] * kernel_volume + kernel_pos,
             acc, mask=(rows[:, None] < CO) & (cols[None, :] < CI))
