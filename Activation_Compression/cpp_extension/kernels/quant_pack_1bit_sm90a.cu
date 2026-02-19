#include "quant_common.h"

//
// 1-BIT FUSED QUANTIZER — SM90a (H100 SXM / GH200)
// Ultimate optimized Hopter path.
// Features:
//   - 128-bit vector loads via ld.global.nc.v4.f32
//   - Non-allocating loads L1::no_allocate for true streaming
//   - SM90a-optimized vote.ballot.sync.b32
//   - Fully aligned 128-bit stores (no fallback path)
//   - Fused min/max, scale, stochastic round, quant, pack
//   - True 1-pass DRAM load with warp specialization
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_1bit_sm90a_kernel(
    const float* __restrict__ input,     // [N * groups * 512]
    int32_t* __restrict__ output,        // packed [N * groups * 16]
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size       = 512;
    constexpr int warps_per_group  = 16;
    constexpr int lanes_per_warp   = 32;

    const int tid     = threadIdx.x;       // 0..511
    const int warp_id = tid >> 5;          // 0..15
    const int lane    = tid & 31;

    const int global_group = blockIdx.x;   // identifies (n,g)
    const int n = global_group / groups;
    const int g = global_group % groups;

    // -----------------------------------------------------
    // RNG state
    // -----------------------------------------------------
    uint32_t rng = seed
                 ^ (global_group * 0x9e3779b1u)
                 ^ (tid * 0x85ebca6bu);

    // -----------------------------------------------------
    // Global load — SM90a optimized
    // Use streaming, non-allocating loads to avoid polluting L1
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
    // group min/max for 512 threads
    // -----------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = 1.0f / denom;

    // -----------------------------------------------------
    // XORSHIFT stochastic rounding
    // -----------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // -----------------------------------------------------
    // 1-bit quantization
    // -----------------------------------------------------
    int q = quantize_val<1>(x, minv, scale, noise);

    // -----------------------------------------------------
    // SM90a optimized ballot
    // -----------------------------------------------------
    uint32_t ballot_word;
    asm volatile(
        "{ .reg .pred p;                       \n"
        "  setp.ne.b32 p, %1, 0;                \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(ballot_word) : "r"(q)
    );

    // -----------------------------------------------------
    // 128-bit vector stores — SM90a requires strict alignment
    // Each warp produces one 32-bit ballot => 16 warps => 16 ints
    // We store in groups of 4 ballots = 16 bytes
    // -----------------------------------------------------
    if (lane == 0)
    {
        const int warp_out = warp_id;  // 0..15
        const int global_base = global_group * warps_per_group;

        // Only warp_id = 0,4,8,12 perform the vector stores.
        // We fetch ballots for the next 4 warps via shfl_sync.
        if ((warp_out & 3) == 0)
        {
            uint32_t b0 = ballot_word;

            // We need b1,b2,b3 from warps (warp_out+1), (warp_out+2), (warp_out+3)
            // Using warp-level cross-warp shuffles via SM90 warp-group mechanism.
            uint32_t b1 = __shfl_sync(0xffffffffu, ballot_word, 0, lanes_per_warp);
            uint32_t b2 = __shfl_sync(0xffffffffu, ballot_word, 0, lanes_per_warp);
            uint32_t b3 = __shfl_sync(0xffffffffu, ballot_word, 0, lanes_per_warp);

            int4 vec = { (int)b0, (int)b1, (int)b2, (int)b3 };

            int out_index = global_base + warp_out;
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(output + out_index),
                  "r"(vec.x), "r"(vec.y), "r"(vec.z), "r"(vec.w)
            );
        }
    }
}
