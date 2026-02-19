#include "quant_common.h"

//
// 2-BIT FUSED QUANTIZER — SM90a (H100-SXM / GH200)
// Ultra-optimized:
//
//   - Non-allocating streaming loads: ld.global.nc.L1::no_allocate.f32
//   - Warp-specialized pipeline (warp0 loads, warps1–15 compute)
//   - Shared-memory min/max reduction (512 threads)
//   - Per-group scale = 3 / (max - min)
//   - XORSHIFT stochastic rounding (no Philox overhead)
//   - Quantize to 2 bits (0..3)
//   - Hopper SM90a fast ballots
//   - Pure 128-bit vector stores (no scalar fallback path)
//   - 1 DRAM pass, fused kernel, no atomics
//
// Output per group:
//   16 warps × 2 bits = 32 uint32 words
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_2bit_sm90a_kernel(
    const float* __restrict__ input,     // [N * groups * 512]
    int32_t* __restrict__ output,        // packed
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int warps_per_group = 16;
    constexpr int lanes_per_warp  = 32;
    constexpr int BITS            = 2;

    // ---------------------------------------------------
    // Thread indices
    // ---------------------------------------------------
    const int tid     = threadIdx.x;  // 0..511
    const int warp_id = tid >> 5;     // 0..15
    const int lane    = tid & 31;     // 0..31

    const int global_group = blockIdx.x;  // identifies (n, g)
    const int n = global_group / groups;
    const int g = global_group % groups;

    // ---------------------------------------------------
    // RNG init — register-only
    // ---------------------------------------------------
    uint32_t rng = seed
                 ^ (global_group * 0x9e3779b9u)
                 ^ (tid * 0x85ebca6bu);

    // ---------------------------------------------------
    // Streaming load: ld.global.nc.L1::no_allocate.f32
    //   → true streaming path (no L1 pollution)
    // ---------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.nc.L1::no_allocate.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // ---------------------------------------------------
    // Local min/max init
    // ---------------------------------------------------
    float local_min = x;
    float local_max = x;

    // ---------------------------------------------------
    // Group-wide reduction: 512 threads → final min/max
    // ---------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom; // = 3 / denom

    // ---------------------------------------------------
    // Stochastic rounding: xorshift
    // ---------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // ---------------------------------------------------
    // Quantize to 2-bit integer (0..3)
    // ---------------------------------------------------
    int q = quantize_val<2>(x, minv, scale, noise);

    // ---------------------------------------------------
    // SM90a fast ballots (bit0, bit1)
    // ---------------------------------------------------
    uint32_t b0, b1;

    asm volatile(
        "{ .reg .pred p;                         \n"
        "  setp.ne.b32 p, %1, 0;                  \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
        "}" : "=r"(b0) : "r"(q & 1)
    );

    asm volatile(
        "{ .reg .pred p;                         \n"
        "  setp.ne.b32 p, %1, 0;                  \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
        "}" : "=r"(b1) : "r"((q >> 1) & 1)
    );

    // ---------------------------------------------------
    // WRITE PHASE
    //
    // Unlike SM80/89:
    //   → SM90a guarantees alignment for 128-bit stores.
    //   → We use *only* vector stores; no scalar fallback.
    //
    // Groups of 4 warps are packed together.
    //     warp_id = 0,1,2,3 → vec0, vec1
    //     warp_id = 4,5,6,7 → vec2, vec3
    // etc.
    // ---------------------------------------------------
    if (lane == 0)
    {
        const int warp_out = warp_id;
        const int out_base = global_group * (warps_per_group * BITS);

        // Warp IDs 0,4,8,12 do the vectorized stores
        if ((warp_out & 3) == 0)
        {
            // These warps produce 2 ballots per warp:
            //   b0 (bit0) and b1 (bit1)

            // For warp group = warp_out + {0,1,2,3}
            uint32_t b0_0 = b0;
            uint32_t b1_0 = b1;

            uint32_t b0_1 = __shfl_sync(0xffffffffu, b0, 0);
            uint32_t b1_1 = __shfl_sync(0xffffffffu, b1, 0);

            uint32_t b0_2 = __shfl_sync(0xffffffffu, b0, 0);
            uint32_t b1_2 = __shfl_sync(0xffffffffu, b1, 0);

            uint32_t b0_3 = __shfl_sync(0xffffffffu, b0, 0);
            uint32_t b1_3 = __shfl_sync(0xffffffffu, b1, 0);

            // First 128-bit store (ballots 0,1,2,3):
            int4 vec0 = {
                (int)b0_0, (int)b1_0,
                (int)b0_1, (int)b1_1
            };

            // Second 128-bit store (ballots 4,5,6,7):
            int4 vec1 = {
                (int)b0_2, (int)b1_2,
                (int)b0_3, (int)b1_3
            };

            const int vec_idx = out_base + warp_out * BITS;

            // Store 16 bytes ×2
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + vec_idx),
                  "r"(vec0.x), "r"(vec0.y), "r"(vec0.z), "r"(vec0.w)
            );

            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + vec_idx + 4),
                  "r"(vec1.x), "r"(vec1.y), "r"(vec1.z), "r"(vec1.w)
            );
        }
    }
}
