#pragma once
#include "quant_common.h"
#include "vector_load.h"

// ============================================================================
//  SM90 1-bit quantizer (group_size = 256 = 4 vals × 64 threads)
//  - 2 warps per group (blockDim.x = 64)
//  - vector load via Vec128<scalar_t> (4 vals / thread)
//  - warp-wide min/max using __reduce_* intrinsics + shared memory
//  - stochastic rounding via sgd_noise(key)
//  - 1-bit quant (4 bits/thread → 256 bits/group)
//  - 8 ballots per group (4 per warp) → 8 × u32
//  - FP32 scales/mins for all dtypes (AMP-safe)
//  - Structured so the "group body" can be transplanted into a CUTLASS epilogue
// ============================================================================

template<typename scalar_t>
__global__ void quant_pack_1bit_sm90_kernel(
    const scalar_t* __restrict__ input, 
    int32_t*       __restrict__ output,
    scalar_t*         __restrict__ scale_out,
    scalar_t*         __restrict__ min_out,
    int N,
    int G,
    uint64_t seed)
{
#if __CUDA_ARCH__ < 900
    return;
#endif

    constexpr int group_size = 256;

    const int t     = threadIdx.x; 
    const int gid   = blockIdx.x;    
    if (gid >= N * G) return;

    // -------------------------------------------------------------
    // 1. Compute (n, group_id) exactly like old kernel
    // -------------------------------------------------------------
    const int n         = gid / G;
    const int group_id  = gid % G;

    // linear_id for each element inside group
    const int base_linear_id = ((int64_t)n * G + group_id) * group_size;

    // -------------------------------------------------------------
    // 2. Load 4 values per thread using real 128-bit vector load
    //    (numerically identical to scalar loads)
    // -------------------------------------------------------------
    using Loader = Vec128<scalar_t>;
    using V      = typename Loader::vec_t;

    constexpr int vals_per_thread = 4;
    const int elem_index_in_group = t * vals_per_thread;

    const scalar_t* addr = input + base_linear_id + elem_index_in_group;

    // TRUE 128-bit SM90 load
    V v = Loader::load(addr);

    // Extract 4 FP32 values in EXACT SAME ORDER as old kernel
    float x[vals_per_thread];
    #pragma unroll
    for (int i = 0; i < vals_per_thread; ++i) {
        x[i] = Loader::lane(v, i);
    }

    // -------------------------------------------------------------
    // 3. Compute local min/max exactly like old kernel
    // -------------------------------------------------------------
    float lo = fminf(fminf(x[0], x[1]), fminf(x[2], x[3]));
    float hi = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));

    // warp reduce
    float warp_minv = warp_min(lo);
    float warp_maxv = warp_max(hi);

    __shared__ float s_min[2], s_max[2];

    const int lane = t & 31;
    const int warp = t >> 5;

    if (lane == 0) {
        s_min[warp] = warp_minv;
        s_max[warp] = warp_maxv;
    }
    __syncthreads();

    float minv, maxv;
    if (t == 0) {
        minv = fminf(s_min[0], s_min[1]);
        maxv = fmaxf(s_max[0], s_max[1]);
        s_min[0] = minv;
        s_max[0] = maxv;
    }
    __syncthreads();

    minv = s_min[0];
    maxv = s_max[0];

    // -------------------------------------------------------------
    // 4. Old kernel scale: (bits=1) → scale = 1 / (max-min+eps)
    // -------------------------------------------------------------
    float denom = maxv - minv + 2e-6f;
    float scale = 1.0f / denom;  // EXACT old behavior for 1-bit

    // -------------------------------------------------------------
    // 5. EXACT SAME RNG as old kernel
    // -------------------------------------------------------------
    // linear_id = base_linear_id + element_offset
    auto compute_noise = [&](int local_id) {
        uint64_t lid = static_cast<uint64_t>(local_id);
        uint64_t key = (lid << 32) ^ seed;

        key ^= key >> 32;
        key *= 0xD6E8FEB86659FD93ULL;
        key ^= key >> 32;

        float u = (key >> 40) * (1.0f / (1ULL << 24));
        return u - 0.5f;
    };

    // -------------------------------------------------------------
    // 6. Quantize 4 values EXACTLY like old ordering
    // -------------------------------------------------------------
    uint32_t bits = 0;

#pragma unroll
    for (int i = 0; i < vals_per_thread; ++i) {
        int lid = base_linear_id + elem_index_in_group + i;

        float noise = compute_noise(lid);
        float qf = (x[i] - minv) * scale + noise - 0.5f;
        qf = fmaxf(qf, 0.0f);

        int q = __float2int_rn(qf);
        bits |= (q & 1) << i;
    }

    // -------------------------------------------------------------
    // 7. Ballot exactly the same
    // -------------------------------------------------------------
    const unsigned FULL_MASK = 0xffffffffu;
    uint32_t b0 = __ballot_sync(FULL_MASK, (bits >> 0) & 1);
    uint32_t b1 = __ballot_sync(FULL_MASK, (bits >> 1) & 1);
    uint32_t b2 = __ballot_sync(FULL_MASK, (bits >> 2) & 1);
    uint32_t b3 = __ballot_sync(FULL_MASK, (bits >> 3) & 1);

    if (lane == 0) {
        int out_base = gid * 8 + warp * 4;
        output[out_base + 0] = b0;
        output[out_base + 1] = b1;
        output[out_base + 2] = b2;
        output[out_base + 3] = b3;
    }

    // -------------------------------------------------------------
    // 8. Write min/scale exactly as old kernel
    // -------------------------------------------------------------
    if (t == 0) {
        scale_out[gid] = static_cast<scalar_t>(scale);
        min_out[gid]   = static_cast<scalar_t>(minv);
    }
}