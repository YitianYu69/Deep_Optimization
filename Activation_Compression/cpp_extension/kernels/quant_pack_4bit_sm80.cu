#include "quant_common.h"

//
// 4-BIT FUSED QUANTIZER — SM80 (A100 / Ampere)
//
// Features:
//   • Fully fused: min/max → scale → stochastic → quantize → pack
//   • group_size = 512
//   • XORSHIFT RNG
//   • 4 bits per value = 4 ballots per warp
//   • 16 warps × 4 ballots = 64 int32 outputs per group
//   • Ampere-optimized streaming loads (ld.global.cs.f32)
//   • Zero-branch, zero-atomic design
//

extern "C" __global__
__launch_bounds__(512)
void quant_pack_4bit_sm80_kernel(
    const float* __restrict__ input,    // [N * groups * 512]
    int32_t* __restrict__ output,       // [N * groups * 64]
    int N,
    int groups,
    uint32_t seed)
{
    constexpr int group_size      = 512;
    constexpr int lanes_per_warp  = 32;
    constexpr int warps_per_group = 16;
    constexpr int BITS            = 4;      // 4-bit quant → 16 levels

    // -------------------------------------------------------------
    // Thread & group indexing
    // -------------------------------------------------------------
    const int tid     = threadIdx.x;        // 0..511
    const int warp_id = tid >> 5;           // 0..15
    const int lane    = tid & 31;           // 0..31

    const int global_group = blockIdx.x;    // 0..N*groups
    const int n = global_group / groups;
    const int g = global_group % groups;

    // -------------------------------------------------------------
    // RNG: register-only xorshift32
    // -------------------------------------------------------------
    uint32_t rng = seed
                 ^ (global_group * 0x9e3779b9u)
                 ^ (tid * 0x85ebca6bu);

    // -------------------------------------------------------------
    // Ampere streaming load: ld.global.cs.f32
    //   Best for single-use loads on SM80
    // -------------------------------------------------------------
    const int base = global_group * group_size + tid;

    float x;
    asm volatile(
        "ld.global.cs.f32 %0, [%1];\n"
        : "=f"(x)
        : "l"(input + base)
    );

    // -------------------------------------------------------------
    // Local min/max initialization
    // -------------------------------------------------------------
    float local_min = x;
    float local_max = x;

    // -------------------------------------------------------------
    // Reduce over 512 threads → group-wide min/max
    // -------------------------------------------------------------
    group_minmax_512(local_min, local_max);

    float minv = local_min;
    float maxv = local_max;

    float denom = maxv - minv + 1e-6f;
    float scale = float((1 << BITS) - 1) / denom;   // 15 / denom

    // -------------------------------------------------------------
    // Stochastic rounding noise
    // -------------------------------------------------------------
    float noise = xorshift32_uniform(rng);

    // -------------------------------------------------------------
    // Quantize value to 4 bits (0..15)
    // -------------------------------------------------------------
    int q = quantize_val<4>(x, minv, scale, noise);

    // -------------------------------------------------------------
    // Warp ballot packing: 4 ballots per warp
    //
    // b0 = bit 0 of 32 values
    // b1 = bit 1
    // b2 = bit 2
    // b3 = bit 3
    // -------------------------------------------------------------
    uint32_t b0 = ptx_ballot(q & 1);
    uint32_t b1 = ptx_ballot((q >> 1) & 1);
    uint32_t b2 = ptx_ballot((q >> 2) & 1);
    uint32_t b3 = ptx_ballot((q >> 3) & 1);

    // -------------------------------------------------------------
    // Store results — lane0 does global write
    //
    // Output layout (per group):
    //   16 warps * 4 ballots = 64 int32 outputs
    //
    // Index:
    //   out[group * (16*4) + warp_id*4 + bit]
    // -------------------------------------------------------------
    if (lane == 0)
    {
        const int out_base =
            global_group * (warps_per_group * BITS)
          + warp_id * BITS;

        output[out_base + 0] = b0;
        output[out_base + 1] = b1;
        output[out_base + 2] = b2;
        output[out_base + 3] = b3;
    }
}
