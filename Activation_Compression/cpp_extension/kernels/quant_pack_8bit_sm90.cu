#include "quant_common.h"

//
// 8-BIT FUSED QUANTIZER — SM90 (Hopper / H100 / GH100)
//
// * Fully fused (single-pass) groupwise quantization
// * 512 threads per group
// * Warp ballot bit-plane packing (8 ballots per warp)
// * Hopper-optimized memory instructions:
//       - ld.global.nc.f32   (fast streaming loads)
//       - st.global.v4.u32   (128-bit vector stores)
// * Faster shared-memory reductions on SM90
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_8bit_sm90_kernel(
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

    //
    // Thread index
    //
    const int tid     = threadIdx.x;      // 0..511
    const int warp_id = tid >> 5;         // 0..15
    const int lane    = tid & 31;

    //
    // Logical group index
    //
    const int global_group = blockIdx.x;
    const int n = global_group / groups;
    const int g = global_group % groups;

    //
    // RNG setup
    //
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b9u) ^
        (tid * 0x85ebca6bu);

    //
    // Streaming load (Hopper-optimized)
    //
    int base_idx = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.nc.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base_idx)
    );

    //
    // Local min/max
    //
    float local_min = x;
    float local_max = x;

    //
    // SM90-optimized reduction (shared memory)
    //
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;  // 255.0 / denom

    //
    // Stochastic noise
    //
    float noise = xorshift32_uniform(rng);

    //
    // Quantize (0..255)
    //
    int q = quantize_val<8>(x, minv, scale, noise);

    //
    // Ballots
    //
    uint32_t b0 = ptx_ballot(q & 1);
    uint32_t b1 = ptx_ballot((q >> 1) & 1);
    uint32_t b2 = ptx_ballot((q >> 2) & 1);
    uint32_t b3 = ptx_ballot((q >> 3) & 1);
    uint32_t b4 = ptx_ballot((q >> 4) & 1);
    uint32_t b5 = ptx_ballot((q >> 5) & 1);
    uint32_t b6 = ptx_ballot((q >> 6) & 1);
    uint32_t b7 = ptx_ballot((q >> 7) & 1);

    //
    // 128-bit vectorized store path
    //
    if (lane == 0)
    {
        const int out_base =
            global_group * (warps_per_group * BITS)
          + warp_id * BITS;

        // Store 8 ballots → 2 × {uint4}
        uint4 v0 = {b0, b1, b2, b3};
        uint4 v1 = {b4, b5, b6, b7};

        asm volatile(
            "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
            :
            : "l"(output + out_base),
              "r"(v0.x), "r"(v0.y), "r"(v0.z), "r"(v0.w)
        );

        asm volatile(
            "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
            :
            : "l"(output + out_base + 4),
              "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w)
        );
    }
}
