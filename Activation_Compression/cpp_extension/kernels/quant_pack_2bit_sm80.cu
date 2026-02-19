#include "quant_common.h"

//
// 2-BIT FUSED QUANTIZER — SM80 (A100 / Ampere)
//
// Fully fused pipeline:
//  - Compute group min/max (512 threads)
//  - Compute per-group scale
//  - XORSHIFT stochastic rounding
//  - Quantize to 2 bits (0..3)
//  - Warp ballot pack (2 ballots per warp)
//  - Write output
//
// Output size per group:
//   warps_per_group = 16
//   bits = 2
//   => 16 warps * 2 ballots = 32 uint32 words per group
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_2bit_sm80_kernel(
    const float* __restrict__ input,    // [N * groups * 512]
    int32_t* __restrict__ output,       // packed bits
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int lanes_per_warp  = 32;
    constexpr int warps_per_group = 16;
    constexpr int BITS            = 2;

    // -----------------------------------------------
    // Thread & group indexing
    // -----------------------------------------------
    const int tid     = threadIdx.x;         // 0..511
    const int warp_id = tid >> 5;            // 0..15
    const int lane    = tid & 31;            // 0..31

    const int global_group = blockIdx.x;     // 0..N*groups
    const int n = global_group / groups;
    const int g = global_group % groups;

    // -----------------------------------------------
    // RNG init
    // -----------------------------------------------
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b1u) ^
        (tid * 0x85ebca6bu);

    // -----------------------------------------------
    // Load input (one DRAM pass)
    // -----------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.cs.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // -----------------------------------------------
    // Local min/max
    // -----------------------------------------------
    float local_min = x;
    float local_max = x;

    // -----------------------------------------------
    // Reduce min/max across 512 threads
    // -----------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;
    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;   // = 3 / denom

    // -----------------------------------------------
    // Stochastic rounding
    // -----------------------------------------------
    float noise = xorshift32_uniform(rng);

    // -----------------------------------------------
    // Quantize to 2 bits
    // -----------------------------------------------
    int q = quantize_val<2>(x, minv, scale, noise); // 0..3

    // -----------------------------------------------
    // Ballot packing
    // Two ballots: bit-0 and bit-1
    // -----------------------------------------------
    uint32_t b0 = ptx_ballot(q & 1);
    uint32_t b1 = ptx_ballot((q >> 1) & 1);

    // -----------------------------------------------
    // Store phase
    // per-group output layout:
    //   output[group_idx * (warps_per_group * BITS) + warp_id * BITS + bit_idx]
    //
    //   where bit_idx = 0 or 1
    // -----------------------------------------------
    if (lane == 0)
    {
        int out_base = global_group * (warps_per_group * BITS)
                     + warp_id * BITS;

        output[out_base + 0] = b0;
        output[out_base + 1] = b1;
    }
}
