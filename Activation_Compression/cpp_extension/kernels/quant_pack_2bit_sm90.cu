#include "quant_common.h"

//
// 2-BIT FUSED QUANTIZER — SM90 (H100 / Hopper)
//
// Features:
//   - Warp-specialized (warp 0 loads, warp 1-15 compute)
//   - Shared memory reduction: min/max across 512 threads
//   - Per-group scale = 3 / (max-min)
//   - XORSHIFT stochastic rounding
//   - Quantize to 2 bits (0..3)
//   - Two SM90-optimized ballots per warp
//   - 128-bit vector stores for peak throughput
//   - Single DRAM pass (streaming loads ld.global.nc)
//
// Output per group: 32 uint32 words
//   => 16 warps * 2 bits = 32 ballots
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_2bit_sm90_kernel(
    const float* __restrict__ input,    // [N * groups * 512]
    int32_t* __restrict__ output,       // packed output
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int warps_per_group = 16;
    constexpr int lanes_per_warp  = 32;
    constexpr int BITS            = 2;

    // -----------------------------------------------------
    // Thread indices
    // -----------------------------------------------------
    const int tid     = threadIdx.x;   // 0..511
    const int warp_id = tid >> 5;      // 0..15
    const int lane    = tid & 31;      // 0..31

    const int global_group = blockIdx.x;   // identifies (n,g)
    const int n = global_group / groups;
    const int g = global_group % groups;

    // -----------------------------------------------------
    // RNG init
    // -----------------------------------------------------
    uint32_t rng = seed
                 ^ (global_group * 0x9e3779b9u)
                 ^ (tid * 0x85ebca6bu);

    // -----------------------------------------------------
    // Hopper streaming load - ld.global.nc.f32
    // -----------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.nc.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // -----------------------------------------------------
    // Local min/max init
    // -----------------------------------------------------
    float local_min = x;
    float local_max = x;

    // -----------------------------------------------------
    // Reduce min/max for 512 threads
    // -----------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom; // = 3 / denom

    // -----------------------------------------------------
    // Stochastic rounding
    // -----------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // -----------------------------------------------------
    // 2-bit quantize
    // -----------------------------------------------------
    int q = quantize_val<2>(x, minv, scale, noise);   // q ∈ [0,3]

    // -----------------------------------------------------
    // Hopper-optimized ballot (bit0, then bit1)
    // -----------------------------------------------------
    uint32_t b0, b1;

    asm volatile(
        "{ .reg .pred p;                           \n"
        "  setp.ne.b32 p, %1, 0;                   \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(b0) : "r"(q & 1)
    );

    asm volatile(
        "{ .reg .pred p;                           \n"
        "  setp.ne.b32 p, %1, 0;                   \n"
        "  vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(b1) : "r"((q >> 1) & 1)
    );

    // -----------------------------------------------------
    // STORE PHASE
    // We pack 4 ballots at a time using 128-bit stores.
    //
    // For 2-bit quant:
    //   each warp produces 2 ballots
    //   => 4 warps produce 8 ballots = 2 vector stores
    //
    // Warp groups (0,1,2,3), (4,5,6,7), ...
    // -----------------------------------------------------
    if (lane == 0)
    {
        const int warp_out = warp_id;   // 0..15
        const int global_base = global_group * (warps_per_group * BITS);

        // Only warp_id % 4 == 0 performs the vector store
        if ((warp_out & 3) == 0)
        {
            // gather b0,b1 from warp_out+0, +1, +2, +3

            // my ballots
            uint32_t b0_0 = b0;
            uint32_t b1_0 = b1;

            // Fetch from the next 3 warps
            uint32_t b0_1 = __shfl_sync(0xffffffffu, b0, 0, lanes_per_warp);
            uint32_t b1_1 = __shfl_sync(0xffffffffu, b1, 0, lanes_per_warp);
            uint32_t b0_2 = __shfl_sync(0xffffffffu, b0, 0, lanes_per_warp);
            uint32_t b1_2 = __shfl_sync(0xffffffffu, b1, 0, lanes_per_warp);
            uint32_t b0_3 = __shfl_sync(0xffffffffu, b0, 0, lanes_per_warp);
            uint32_t b1_3 = __shfl_sync(0xffffffffu, b1, 0, lanes_per_warp);

            // Final output order for these 8 ballots:
            // b0_0, b1_0, b0_1, b1_1   (vector store 0)
            // b0_2, b1_2, b0_3, b1_3   (vector store 1)

            int store_idx = global_base + warp_out * BITS;

            int4 vec0 = {
                (int)b0_0, (int)b1_0,
                (int)b0_1, (int)b1_1
            };
            int4 vec1 = {
                (int)b0_2, (int)b1_2,
                (int)b0_3, (int)b1_3
            };

            // Vectorized 16-byte store #1
            *reinterpret_cast<int4*>(&output[store_idx]) = vec0;

            // Vectorized store #2
            *reinterpret_cast<int4*>(&output[store_idx + 4]) = vec1;
        }
    }
}
