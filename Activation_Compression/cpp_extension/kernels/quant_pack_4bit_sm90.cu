#include "quant_common.h"

//
// 4-BIT FUSED QUANTIZER — SM90 (H100 / Hopper)
//
// Features:
//   • Warp-specialized Hopper kernel
//   • Streaming loads: ld.global.nc.f32
//   • 512-thread group reduction (min/max)
//   • Per-group scale = 15 / (max - min)
//   • XORSHIFT stochastic rounding
//   • 4-bit quantization → 4 ballots per warp
//   • 128-bit vector stores for maximum global bandwidth
//   • Output = 64 int32 ballots per group
//   • Zero atomics, zero control divergence
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_4bit_sm90_kernel(
    const float* __restrict__ input,     // [N * groups * 512]
    int32_t* __restrict__ output,        // [N * groups * 64]
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int warps_per_group = 16;
    constexpr int lanes_per_warp  = 32;
    constexpr int BITS            = 4;   // 4-bit quantization

    // -------------------------------------------------------
    // Thread & group indexing
    // -------------------------------------------------------
    const int tid     = threadIdx.x;     // 0..511
    const int warp_id = tid >> 5;        // 0..15
    const int lane    = tid & 31;        // 0..31

    const int global_group = blockIdx.x;
    const int n = global_group / groups;
    const int g = global_group % groups;

    // -------------------------------------------------------
    // RNG init
    // -------------------------------------------------------
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b9u) ^
        (tid * 0x85ebca6bu);

    // -------------------------------------------------------
    // Hopper streaming load (L1-bypass)
    // -------------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.nc.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // -------------------------------------------------------
    // Local min/max
    // -------------------------------------------------------
    float local_min = x;
    float local_max = x;

    // -------------------------------------------------------
    // Shared-memory group reduction (512 threads)
    // -------------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;   // 15 / denom

    // -------------------------------------------------------
    // Stochastic rounding
    // -------------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // -------------------------------------------------------
    // Quantize to 4-bit (0..15)
    // -------------------------------------------------------
    int q = quantize_val<4>(x, minv, scale, noise);

    // -------------------------------------------------------
    // Hopper-optimized ballots (bit0..bit3)
    // -------------------------------------------------------
    uint32_t b0, b1, b2, b3;

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; }"
        : "=r"(b0) : "r"(q & 1)
    );

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; }"
        : "=r"(b1) : "r"((q >> 1) & 1)
    );

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; }"
        : "=r"(b2) : "r"((q >> 2) & 1)
    );

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; }"
        : "=r"(b3) : "r"((q >> 3) & 1)
    );

    // -------------------------------------------------------
    // Write-back phase (128-bit stores)
    //
    // Warp groups of 4:
    //    warp 0 writes ballots for warps 0,1,2,3
    //    warp 4 writes ballots for warps 4,5,6,7
    //    ...
    //
    // Each warp produces 4 ballots (b0..b3)
    // Combined = 16 ballots per warp-group
    //
    // Vectorized stores use:
    //    st.global.v4.u32 {x,y,z,w}
    //
    // -------------------------------------------------------
    if (lane == 0)
    {
        const int warp_out  = warp_id;
        const int out_base  = global_group * (warps_per_group * BITS);

        // warp groups: 0,4,8,12 perform the write
        if ((warp_out & 3) == 0)
        {
            // Fetch ballots from 4 consecutive warps
            uint32_t b0_0 = b0, b1_0 = b1, b2_0 = b2, b3_0 = b3;

            uint32_t b0_1 = __shfl_sync(0xffffffffu, b0, 0);
            uint32_t b1_1 = __shfl_sync(0xffffffffu, b1, 0);
            uint32_t b2_1 = __shfl_sync(0xffffffffu, b2, 0);
            uint32_t b3_1 = __shfl_sync(0xffffffffu, b3, 0);

            uint32_t b0_2 = __shfl_sync(0xffffffffu, b0, 0);
            uint32_t b1_2 = __shfl_sync(0xffffffffu, b1, 0);
            uint32_t b2_2 = __shfl_sync(0xffffffffu, b2, 0);
            uint32_t b3_2 = __shfl_sync(0xffffffffu, b3, 0);

            uint32_t b0_3 = __shfl_sync(0xffffffffu, b0, 0);
            uint32_t b1_3 = __shfl_sync(0xffffffffu, b1, 0);
            uint32_t b2_3 = __shfl_sync(0xffffffffu, b2, 0);
            uint32_t b3_3 = __shfl_sync(0xffffffffu, b3, 0);

            // Vector stores
            int store_idx = out_base + warp_out * BITS;

            // First 128-bit block
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + store_idx),
                  "r"(b0_0), "r"(b1_0), "r"(b2_0), "r"(b3_0)
            );

            // Second 128-bit block
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + store_idx + 4),
                  "r"(b0_1), "r"(b1_1), "r"(b2_1), "r"(b3_1)
            );

            // Third block
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + store_idx + 8),
                  "r"(b0_2), "r"(b1_2), "r"(b2_2), "r"(b3_2)
            );

            // Fourth block
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + store_idx + 12),
                  "r"(b0_3), "r"(b1_3), "r"(b2_3), "r"(b3_3)
            );
        }
    }
}
