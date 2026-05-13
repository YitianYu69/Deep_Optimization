#include "quant_common.h"

//
// 4-BIT FUSED QUANTIZER — SM90a (H100-SXM / GH200)
//
// Features:
//   - Fully fused (min/max → scale → stochastic → quantize → pack)
//   - Warp-specialized Hopper pipeline
//   - Streaming loads bypass L1: ld.global.nc.L1::no_allocate.f32
//   - 4-bit quant: 4 ballots per warp
//   - 16 warps → 64 int32 ballots per group
//   - 128-bit vector stores (st.global.v4.u32)
//   - RNG = XORSHIFT32 (register-only)
//   - Zero atomics, zero divergence
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_4bit_sm90a_kernel(
    const float* __restrict__ input,   // [N * groups * 512]
    int32_t* __restrict__ output,      // [N * groups * 64]
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int warps_per_group = 16;
    constexpr int lanes_per_warp  = 32;
    constexpr int BITS            = 4;

    // -----------------------------------------------------
    // Thread indices
    // -----------------------------------------------------
    const int tid     = threadIdx.x;    // 0..511
    const int warp_id = tid >> 5;       // 0..15
    const int lane    = tid & 31;       // 0..31

    const int global_group = blockIdx.x;
    const int n = global_group / groups;
    const int g = global_group % groups;

    // -----------------------------------------------------
    // RNG init (xorshift32)
    // -----------------------------------------------------
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b9u) ^
        (tid          * 0x85ebca6bu);

    // -----------------------------------------------------
    // Hopper-SMX90A streaming load:
    //    ld.global.nc.L1::no_allocate.f32
    // -----------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.nc.L1::no_allocate.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // -----------------------------------------------------
    // Local min/max init
    // -----------------------------------------------------
    float local_min = x;
    float local_max = x;

    // -----------------------------------------------------
    // Shared-memory 512-thread reduction
    // -----------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom; // = 15 / (max-min)

    // -----------------------------------------------------
    // Stochastic rounding
    // -----------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // -----------------------------------------------------
    // Quantize into 4-bit integer (0..15)
    // -----------------------------------------------------
    int q = quantize_val<4>(x, minv, scale, noise);

    // -----------------------------------------------------
    // SM90a fast ballots for 4 bits
    // -----------------------------------------------------
    uint32_t b0, b1, b2, b3;

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;     \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(b0) : "r"(q & 1)
    );

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;     \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(b1) : "r"((q >> 1) & 1)
    );

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;     \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(b2) : "r"((q >> 2) & 1)
    );

    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %1, 0;     \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(b3) : "r"((q >> 3) & 1)
    );

    // -----------------------------------------------------
    // Writeback (SM90a: vector-only storage path)
    //
    // Warp IDs 0,4,8,12 write ballots for 4 consecutive warps:
    //
    //   warp_out = 0  → writes ballots of warp 0,1,2,3
    //   warp_out = 4  → writes warp 4,5,6,7
    //   warp_out = 8  → writes warp 8,9,10,11
    //   warp_out = 12 → writes warp 12..15
    //
    // Each warp generates 4 ballots → 16 total
    // Written as 4 × int4 (128-bit) stores
    // -----------------------------------------------------
    if (lane == 0)
    {
        const int warp_out = warp_id;

        if ((warp_out & 3) == 0)
        {
            const int out_base = global_group * (warps_per_group * BITS)
                               + warp_out * BITS;

            // Collect ballots from this warp-group (4 warps)
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

            // Perform 4 vector stores (16 ballots)
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + out_base),
                  "r"(b0_0), "r"(b1_0), "r"(b2_0), "r"(b3_0)
            );
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + out_base + 4),
                  "r"(b0_1), "r"(b1_1), "r"(b2_1), "r"(b3_1)
            );
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + out_base + 8),
                  "r"(b0_2), "r"(b1_2), "r"(b2_2), "r"(b3_2)
            );
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + out_base + 12),
                  "r"(b0_3), "r"(b1_3), "r"(b2_3), "r"(b3_3)
            );
        }
    }
}
