#include "quant_common.h"

extern "C" __global__
__launch_bounds__(512)
void quant_pack_8bit_sm90a_kernel(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int warps_per_group = 16;
    constexpr int BITS            = 8;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    const int global_group = blockIdx.x;
    const int n = global_group / groups;
    const int g = global_group % groups;

    // Per-thread fast RNG
    uint32_t rng =
        seed ^
        (global_group * 0x9e3779b9u) ^
        (tid * 0x85ebca6bu * 3);

    int base_idx = global_group * group_size + tid;
    float x;

    //
    // SM90a-safe load (no .nt)
    //
    asm volatile(
        "ld.global.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base_idx)
    );

    // Local accumulators
    float local_min = x;
    float local_max = x;

    // 512-thread optimized reduction
    group_minmax_512(local_min, local_max);

    float minv  = local_min;
    float maxv  = local_max;
    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;

    float noise = xorshift32_uniform(rng);

    int q = quantize_val<8>(x, minv, scale, noise);

    // Ballots
    uint32_t b0 = ptx_ballot(q & 1);
    uint32_t b1 = ptx_ballot((q >> 1) & 1);
    uint32_t b2 = ptx_ballot((q >> 2) & 1);
    uint32_t b3 = ptx_ballot((q >> 3) & 1);
    uint32_t b4 = ptx_ballot((q >> 4) & 1);
    uint32_t b5 = ptx_ballot((q >> 5) & 1);
    uint32_t b6 = ptx_ballot((q >> 6) & 1);
    uint32_t b7 = ptx_ballot((q >> 7) & 1);

    if (lane == 0) {
        const int out_base =
            global_group * (warps_per_group * BITS) +
            warp_id * BITS;

        uint4 v0 = {b0, b1, b2, b3};
        uint4 v1 = {b4, b5, b6, b7};

        //
        // SM90a-safe store (no .nt, no .nc)
        //
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
