#include "quant_common.h"

extern "C" __global__
void quant_pack_1bit_sm80_kernel(
    const float* __restrict__ input,    // [N * groups * 512]
    int32_t* __restrict__ output,       // packed bits
    int N,
    int groups,
    uint32_t seed)
{
    // -------------------------------------------------
    // Thread + group indexing
    // -------------------------------------------------
    const int group_size = 512;
    const int warps_per_group = 16;   // 512 / 32
    const int tid = threadIdx.x;      // 0..511
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int global_group = blockIdx.x;   // group index [0..N*groups)
    const int n = global_group / groups;
    const int g = global_group % groups;

    uint32_t rng = seed ^ (global_group * 1315423911u + tid * 2654435761u);

    // -------------------------------------------------
    // Load data (single pass)
    // -------------------------------------------------
    int base = (global_group * group_size);
    float x = __ldg(&input[base + tid]);

    // local min/max init
    float local_min = x;
    float local_max = x;

    // Compute group min/max using shared memory + warp reductions
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;
    float denom = maxv - minv + 1e-6f;
    float scale = 1.0f / denom;  // for 1-bit, max value is 1

    // -------------------------------------------------
    // Stochastic rounding
    // -------------------------------------------------
    float noise = xorshift32_uniform(rng); // [0,1)

    // quantize to 1 bit: value ∈ {0,1}
    int q = quantize_val<1>(x, minv, scale, noise);

    // -------------------------------------------------
    // Warp ballot packing
    // -------------------------------------------------
    uint32_t ballot = ptx_ballot(q);

    // Only lane 0 stores
    if (lane == 0) {
        // For 1 bit:
        // Each warp outputs 1 int32
        // Output layout:
        // [global_group * warps_per_group + warp_id]
        output[global_group * warps_per_group + warp_id] = ballot;
    }
}
