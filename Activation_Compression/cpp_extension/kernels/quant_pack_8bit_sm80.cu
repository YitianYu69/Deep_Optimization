#include "quant_common.h"


//
// 8-BIT FUSED QUANTIZER — SM80 (A100 / Ampere)
//
// • group_size = 512
// • 8-bit quant (0..255)
// • 8 ballots per warp (bit 0..7)
// • 16 warps → 128 ballots per group
// • Streaming loads: ld.global.cs.f32
// • Shared-memory min/max reduction
// • XORSHIFT stochastic rounding
// • Fully fused, no atomics, no branches
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_8bit_sm80_kernel(
    const float* __restrict__ input,     // [N * groups * 512]
    int32_t* __restrict__ output,        // [N * groups * 128]
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int lanes_per_warp  = 32;
    constexpr int warps_per_group = 16;
    constexpr int BITS            = 8;   // 8-bit quant

    // ----------------------------------------------------------
    // Thread indexing
    // ----------------------------------------------------------
    const int tid     = threadIdx.x;      // 0..511
    const int warp_id = tid >> 5;         // 0..15
    const int lane    = tid & 31;         // 0..31

    const int global_group = blockIdx.x;  // range: 0..N*groups-1
    const int n = global_group / groups;
    const int g = global_group % groups;

    // ----------------------------------------------------------
    // RNG init
    // ----------------------------------------------------------
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b9u) ^
        (tid          * 0x85ebca6bu);

    // ----------------------------------------------------------
    // Streaming load (Ampere best path)
    // ----------------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.cs.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // ----------------------------------------------------------
    // Local min/max
    // ----------------------------------------------------------
    float local_min = x;
    float local_max = x;

    // ----------------------------------------------------------
    // Reduce across 512 threads
    // ----------------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;   // 255 / denom

    // ----------------------------------------------------------
    // Noise (stochastic rounding)
    // ----------------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // ----------------------------------------------------------
    // 8-bit Quantization (0..255)
    // ----------------------------------------------------------
    int q = quantize_val<8>(x, minv, scale, noise);

    // ----------------------------------------------------------
    // Warp ballots: extract bit-planes b0..b7
    // ----------------------------------------------------------
    uint32_t b0 = ptx_ballot(q & 1);
    uint32_t b1 = ptx_ballot((q >> 1) & 1);
    uint32_t b2 = ptx_ballot((q >> 2) & 1);
    uint32_t b3 = ptx_ballot((q >> 3) & 1);
    uint32_t b4 = ptx_ballot((q >> 4) & 1);
    uint32_t b5 = ptx_ballot((q >> 5) & 1);
    uint32_t b6 = ptx_ballot((q >> 6) & 1);
    uint32_t b7 = ptx_ballot((q >> 7) & 1);

    // ----------------------------------------------------------
    // Store: 128 ballots per group
    //   out[group_base + warp_id * 8 + bit]
    // ----------------------------------------------------------
    if (lane == 0)
    {
        const int out_base =
            global_group * (warps_per_group * BITS) +
            warp_id * BITS;

        output[out_base + 0] = b0;
        output[out_base + 1] = b1;
        output[out_base + 2] = b2;
        output[out_base + 3] = b3;
        output[out_base + 4] = b4;
        output[out_base + 5] = b5;
        output[out_base + 6] = b6;
        output[out_base + 7] = b7;
    }
}
