#include "quant_common.h"

//
// 4-BIT FUSED QUANTIZER — SM89 (Ada / Lovelace)
//
// • group_size = 512 (16 warps)
// • 4-bit quant → 4 ballots per warp
// • 64 int32 words per group
// • Ampere/Ada streaming load: ld.global.cs.f32
// • Fully fused pipeline, zero atomics
// • XORSHIFT stochastic rounding
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_4bit_sm89_kernel(
    const float* __restrict__ input,     // [N * groups * 512]
    int32_t* __restrict__ output,        // [N * groups * 64]
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int warps_per_group = 16;
    constexpr int lanes_per_warp  = 32;
    constexpr int BITS            = 4;   // 4-bit quant (0..15)

    // ----------------------------------------------------------
    // Thread & group indexing
    // ----------------------------------------------------------
    const int tid     = threadIdx.x;       // 0..511
    const int warp_id = tid >> 5;          // 0..15
    const int lane    = tid & 31;          // 0..31

    const int global_group = blockIdx.x;   // identifies (n,g)
    const int n = global_group / groups;
    const int g = global_group % groups;

    // ----------------------------------------------------------
    // RNG init — xorshift32
    // ----------------------------------------------------------
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b9u) ^
        (tid          * 0x85ebca6bu);

    // ----------------------------------------------------------
    // Ada-optimized streaming load (best for single-use loads)
    // ----------------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.cs.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // ----------------------------------------------------------
    // Local min/max initialization
    // ----------------------------------------------------------
    float local_min = x;
    float local_max = x;

    // ----------------------------------------------------------
    // Reduce min/max over 512 threads
    // ----------------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;   // 15 / denom

    // ----------------------------------------------------------
    // Stochastic rounding
    // ----------------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // ----------------------------------------------------------
    // Quantize to 4 bits (0..15)
    // ----------------------------------------------------------
    int q = quantize_val<4>(x, minv, scale, noise);

    // ----------------------------------------------------------
    // Ballot pack: 4 bits → 4 ballots per warp
    // ----------------------------------------------------------
    uint32_t b0 = ptx_ballot(q & 1);
    uint32_t b1 = ptx_ballot((q >> 1) & 1);
    uint32_t b2 = ptx_ballot((q >> 2) & 1);
    uint32_t b3 = ptx_ballot((q >> 3) & 1);

    // ----------------------------------------------------------
    // Store results
    // Output layout per group:
    //   16 warps × 4 ballots = 64 int32 values
    // ----------------------------------------------------------
    if (lane == 0)
    {
        const int out_base =
            global_group * (warps_per_group * BITS)
          + warp_id * BITS;

        output[out_base + 0] = b0;
        output[out_base + 1] = b1;
        output[out_base + 2] = b2;
        output[out_base + 3] = b3;
    }
}
