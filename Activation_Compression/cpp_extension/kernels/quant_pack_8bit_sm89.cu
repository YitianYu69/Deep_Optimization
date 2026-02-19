#include "quant_common.h"

//
// 8-BIT FUSED QUANTIZER — SM89 (Ada / Lovelace)
//
// - Fully fused kernel (one DRAM pass)
// - 512 threads per group
// - 8 ballots per warp → 128 ballots per group
// - ld.global.cs.f32 streaming loads
// - xorshift32 RNG
// - zero branches, zero atomics
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_8bit_sm89_kernel(
    const float* __restrict__ input,     // [N * groups * 512]
    int32_t* __restrict__ output,        // [N * groups * 128]
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int lanes_per_warp  = 32;
    constexpr int warps_per_group = 16;
    constexpr int BITS            = 8;

    // --------------------------------------------------------
    // Thread & warp indexing
    // --------------------------------------------------------
    const int tid     = threadIdx.x;      // 0..511
    const int warp_id = tid >> 5;         // 0..15
    const int lane    = tid & 31;         // 0..31

    const int global_group = blockIdx.x;  // 0..N*groups-1
    const int n = global_group / groups;
    const int g = global_group % groups;

    // --------------------------------------------------------
    // Fast RNG (xorshift)
    // --------------------------------------------------------
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b9u) ^
        (tid * 0x85ebca6bu);

    // --------------------------------------------------------
    // Ada-optimized streaming load (".cs")
    // --------------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.cs.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // --------------------------------------------------------
    // Local min/max initialization
    // --------------------------------------------------------
    float local_min = x;
    float local_max = x;

    // --------------------------------------------------------
    // Shared-memory min/max reduction (512 threads)
    // --------------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;  // = 255 / denom

    // --------------------------------------------------------
    // Stochastic rounding noise
    // --------------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // --------------------------------------------------------
    // Quantize to 8 bits (0..255)
    // --------------------------------------------------------
    int q = quantize_val<8>(x, minv, scale, noise);

    // --------------------------------------------------------
    // Warp ballot packing: 8 bit-planes
    // --------------------------------------------------------
    uint32_t b0 = ptx_ballot(q & 1);
    uint32_t b1 = ptx_ballot((q >> 1) & 1);
    uint32_t b2 = ptx_ballot((q >> 2) & 1);
    uint32_t b3 = ptx_ballot((q >> 3) & 1);
    uint32_t b4 = ptx_ballot((q >> 4) & 1);
    uint32_t b5 = ptx_ballot((q >> 5) & 1);
    uint32_t b6 = ptx_ballot((q >> 6) & 1);
    uint32_t b7 = ptx_ballot((q >> 7) & 1);

    // --------------------------------------------------------
    // Store ballots: 128 ints per group
    // Layout:
    //   out[groupBase + warp_id * 8 + bit]
    // --------------------------------------------------------
    if (lane == 0)
    {
        const int out_base =
            global_group * (warps_per_group * BITS)
          + warp_id * BITS;

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
