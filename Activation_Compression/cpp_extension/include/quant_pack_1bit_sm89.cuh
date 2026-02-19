#pragma once
#include "quant_common.h"
#include "vector_load.h"

// ================================================================
//  1-BIT PACK — SM89 — 128-element group (4 vals per thread × 32)
// ================================================================
template<typename scalar_t>
__global__ void quant_pack_1bit_sm89_kernel(
    const scalar_t* __restrict__ input,
    int32_t* __restrict__ output,
    scalar_t* __restrict__ scaler,
    scalar_t* __restrict__ minv_out,
    int N,
    int groups,
    uint32_t seed)
{
    const int tid = threadIdx.x;    // 0..31
    const int global_group = blockIdx.x;

    constexpr int group_size = 128;        // 128 elements per group
    constexpr int vals_per_thread = 4;     // vectorized load
    constexpr int threads = 32;

    // Base index for this group
    const int base_idx = global_group * group_size;

    // ============================================================
    //  Vector load (128-bit) → 4 scalar FP32 values
    // ============================================================
    using Loader = Vec128<scalar_t>;
    using V = typename Loader::vec_t;

    // Load 16 bytes per thread
    V v = Loader::load(input + base_idx + tid * vals_per_thread);

    float x[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
        x[i] = Loader::lane(v, i);

    // ============================================================
    //  Local min/max for thread’s 4 values
    // ============================================================
    float local_min = fminf(fminf(x[0], x[1]), fminf(x[2], x[3]));
    float local_max = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));

    // ============================================================
    //  Warp-wide min/max (1 warp = 32 threads)
    // ============================================================
    local_min = warp_min(local_min);
    local_max = warp_max(local_max);

    float minv = __shfl_sync(0xffffffff, local_min, 0);
    float maxv = __shfl_sync(0xffffffff, local_max, 0);

    float scale = ((1 << 1) - 1) / (maxv - minv + 1e-6f);

    // ============================================================
    //  Quantize 4 values → pack 4 bits in thread_bits
    // ============================================================
    uint32_t thread_bits = 0;

    uint32_t rng = seed
                 ^ (global_group * 0x9e3779b1u)
                 ^ (tid * 0x85ebca6bu);

#pragma unroll
    for (int i = 0; i < 4; i++) {
        float noise = xorshift32_uniform(rng);
        int q = quantize_val<1>(x[i], minv, scale, noise);
        thread_bits |= (q & 1) << i;
    }

    // ============================================================
    //  Warp ballot: 32 threads → 32 bits
    // ============================================================
    // Each thread contributes ONE BIT: the bit0 of its 4-bit pack.
    // q0 pack → ballot 0
    uint32_t out0 = __ballot_sync(0xffffffff, (thread_bits >> 0) & 1);
    
    // q1 pack → ballot 1
    uint32_t out1 = __ballot_sync(0xffffffff, (thread_bits >> 1) & 1);
    
    // q2 pack → ballot 2
    uint32_t out2 = __ballot_sync(0xffffffff, (thread_bits >> 2) & 1);
    
    // q3 pack → ballot 3
    uint32_t out3 = __ballot_sync(0xffffffff, (thread_bits >> 3) & 1);
    
    int base = global_group * 4;

    if (tid == 0) {
        uint4 store_val = { out0, out1, out2, out3 };
        *reinterpret_cast<uint4*>(&output[base]) = store_val;
    }

    // ============================================================
    //  Write scaler/min (per group)
    // ============================================================
    if (tid == 0) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            scaler[global_group]   = scale;
            minv_out[global_group] = minv;
        }
        else if constexpr (std::is_same<scalar_t, half>::value) {
            scaler[global_group]   = __float2half(scale);
            minv_out[global_group] = __float2half(minv);
        }
        else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
            scaler[global_group]   = __float2bfloat16(scale);
            minv_out[global_group] = __float2bfloat16(minv);
        }
        else if constexpr (std::is_same<scalar_t, c10::Float8_e4m3fn>::value) {
            scaler[global_group]   = c10::Float8_e4m3fn(scale);
            minv_out[global_group] = c10::Float8_e4m3fn(minv);
        }
        else if constexpr (std::is_same<scalar_t, c10::Float8_e5m2>::value) {
            scaler[global_group]   = c10::Float8_e5m2(scale);
            minv_out[global_group] = c10::Float8_e5m2(minv);
        }
    }
}
