#pragma once
#include "quant_common.h"
#include "vector_load.h"

template<typename scalar_t, int bits>
__global__ void dequant_unpack_sm90_kernel(
    const int32_t* __restrict__ input,    // [N*groups*8*bits]
    scalar_t* __restrict__ unpack,        // [N*groups*256]
    const scalar_t* __restrict__ scaler,  // [N*groups]
    const scalar_t* __restrict__ ema_minv,    // [N*groups]
    int N,
    int groups,
    int batchSize)
{


    #if __CUDA_ARCH__ < 900
        return;
    #endif

    const int t    = threadIdx.x;    // 0..63
    const int lane = t & 31;         // 0..31
    const int warp = t >> 5;         // 0..1
    const int gid  = blockIdx.x;     // group index

    if (gid >= N * groups)
        return;

    constexpr int group_size      = 256;
    constexpr int vals_per_thread = 4;

    // -------------------------------------------------------
    // 1. Load scale + min (fp32 reliable)
    // -------------------------------------------------------
    float s = static_cast<float>(scaler[gid]);
    float m = static_cast<float>(ema_minv[gid]);

    // -------------------------------------------------------
    // 2. Vector load 4 int32 ballots at once (16 bytes)
    //
    // This matches the pack kernel *exactly*.
    // -------------------------------------------------------
    const int base_word = gid * 8 * bits + warp * 4 * bits;

    using Loader = Vec128<int32_t>;
    using V      = typename Loader::vec_t;

    const int32_t* addr = input + base_word;

    V v, v_2, v_3, v_4, v_5, v_6, v_7, v_8;

    if constexpr (bits == 1) {
        v = Loader::load(addr);
    }
    else if constexpr (bits == 2) {
        v = Loader::load(addr);
        v_2 = Loader::load(addr + 4);

    } else if constexpr (bits == 3) {
        v = Loader::load(addr);
        v_2 = Loader::load(addr + 4);
        v_3 = Loader::load(addr + 8);

    } else if constexpr (bits == 4) {
        v = Loader::load(addr);
        v_2 = Loader::load(addr + 4);
        v_3 = Loader::load(addr + 8);
        v_4 = Loader::load(addr + 12);
    } else if constexpr (bits == 8) {
        v = Loader::load(addr);
        v_2 = Loader::load(addr + 4);
        v_3 = Loader::load(addr + 8);
        v_4 = Loader::load(addr + 12);
        v_5 = Loader::load(addr + 16);
        v_6 = Loader::load(addr + 20);
        v_7 = Loader::load(addr + 24);
        v_8 = Loader::load(addr + 28);
    }


    // v.x = input[g*8 + warp*4 + 0]
    // v.y = input[g*8 + warp*4 + 1]
    // v.z = input[g*8 + warp*4 + 2]
    // v.w = input[g*8 + warp*4 + 3]

    int32_t b0 = Loader::lane(v, 0);
    int32_t b1 = Loader::lane(v, 1);
    int32_t b2 = Loader::lane(v, 2);
    int32_t b3 = Loader::lane(v, 3);

    int32_t b01 = 0, b11 = 0, b21 = 0, b31 = 0;
    int32_t b02 = 0, b12 = 0, b22 = 0, b32 = 0;
    int32_t b03 = 0, b13 = 0, b23 = 0, b33 = 0;
    int32_t b04 = 0, b14 = 0, b24 = 0, b34 = 0;
    int32_t b05 = 0, b15 = 0, b25 = 0, b35 = 0;
    int32_t b06 = 0, b16 = 0, b26 = 0, b36 = 0;
    int32_t b07 = 0, b17 = 0, b27 = 0, b37 = 0;

    if constexpr (bits == 2) {
        b01 = Loader::lane(v_2, 0);
        b11 = Loader::lane(v_2, 1);
        b21 = Loader::lane(v_2, 2);
        b31 = Loader::lane(v_2, 3);
    } else if constexpr (bits == 3) {
        b01 = Loader::lane(v_2, 0);
        b11 = Loader::lane(v_2, 1);
        b21 = Loader::lane(v_2, 2);
        b31 = Loader::lane(v_2, 3);

        b02 = Loader::lane(v_3, 0);
        b12 = Loader::lane(v_3, 1);
        b22 = Loader::lane(v_3, 2);
        b32 = Loader::lane(v_3, 3);

    } else if constexpr (bits == 4) {
        b01 = Loader::lane(v_2, 0);
        b11 = Loader::lane(v_2, 1);
        b21 = Loader::lane(v_2, 2);
        b31 = Loader::lane(v_2, 3);

        b02 = Loader::lane(v_3, 0);
        b12 = Loader::lane(v_3, 1);
        b22 = Loader::lane(v_3, 2);
        b32 = Loader::lane(v_3, 3);

        b03 = Loader::lane(v_4, 0);
        b13 = Loader::lane(v_4, 1);
        b23 = Loader::lane(v_4, 2);
        b33 = Loader::lane(v_4, 3);
    } else if constexpr (bits == 8) {
        b01 = Loader::lane(v_2, 0);
        b11 = Loader::lane(v_2, 1);
        b21 = Loader::lane(v_2, 2);
        b31 = Loader::lane(v_2, 3);

        b02 = Loader::lane(v_3, 0);
        b12 = Loader::lane(v_3, 1);
        b22 = Loader::lane(v_3, 2);
        b32 = Loader::lane(v_3, 3);

        b03 = Loader::lane(v_4, 0);
        b13 = Loader::lane(v_4, 1);
        b23 = Loader::lane(v_4, 2);
        b33 = Loader::lane(v_4, 3);

        b04 = Loader::lane(v_5, 0);
        b14 = Loader::lane(v_5, 1);
        b24 = Loader::lane(v_5, 2);
        b34 = Loader::lane(v_5, 3);

        b05 = Loader::lane(v_6, 0);
        b15 = Loader::lane(v_6, 1);
        b25 = Loader::lane(v_6, 2);
        b35 = Loader::lane(v_6, 3);

        b06 = Loader::lane(v_7, 0);
        b16 = Loader::lane(v_7, 1);
        b26 = Loader::lane(v_7, 2);
        b36 = Loader::lane(v_7, 3);

        b07 = Loader::lane(v_8, 0);
        b17 = Loader::lane(v_8, 1);
        b27 = Loader::lane(v_8, 2);
        b37 = Loader::lane(v_8, 3);
    }

    // -------------------------------------------------------
    // 3. Extract 1-bit per value
    // -------------------------------------------------------
    int q0 = (b0 >> lane) & 1;
    int q1 = (b1 >> lane) & 1;
    int q2 = (b2 >> lane) & 1;
    int q3 = (b3 >> lane) & 1;

    if constexpr (bits == 2) {
        q0 |= (((b01 >> lane) & 1) << 1);
        q1 |= (((b11 >> lane) & 1) << 1);
        q2 |= (((b21 >> lane) & 1) << 1);
        q3 |= (((b31 >> lane) & 1) << 1);
        
    } else if constexpr (bits == 3) {
        q0 |= (((b01 >> lane) & 1u) << 1);
        q1 |= (((b11 >> lane) & 1u) << 1);
        q2 |= (((b21 >> lane) & 1u) << 1);
        q3 |= (((b31 >> lane) & 1u) << 1);

        q0 |= (((b02 >> lane) & 1u) << 2);
        q1 |= (((b12 >> lane) & 1u) << 2);
        q2 |= (((b22 >> lane) & 1u) << 2);
        q3 |= (((b32 >> lane) & 1u) << 2);

    } else if constexpr (bits == 4) {
        q0 |= (((b01 >> lane) & 1) << 1);
        q1 |= (((b11 >> lane) & 1) << 1);
        q2 |= (((b21 >> lane) & 1) << 1);
        q3 |= (((b31 >> lane) & 1) << 1);

        q0 |= (((b02 >> lane) & 1) << 2);
        q1 |= (((b12 >> lane) & 1) << 2);
        q2 |= (((b22 >> lane) & 1) << 2);
        q3 |= (((b32 >> lane) & 1) << 2);

        q0 |= (((b03 >> lane) & 1) << 3);
        q1 |= (((b13 >> lane) & 1) << 3);
        q2 |= (((b23 >> lane) & 1) << 3);
        q3 |= (((b33 >> lane) & 1) << 3);
    } else if constexpr (bits == 8) {
        q0 |= (((b01 >> lane) & 1) << 1);
        q1 |= (((b11 >> lane) & 1) << 1);
        q2 |= (((b21 >> lane) & 1) << 1);
        q3 |= (((b31 >> lane) & 1) << 1);

        q0 |= (((b02 >> lane) & 1) << 2);
        q1 |= (((b12 >> lane) & 1) << 2);
        q2 |= (((b22 >> lane) & 1) << 2);
        q3 |= (((b32 >> lane) & 1) << 2);

        q0 |= (((b03 >> lane) & 1) << 3);
        q1 |= (((b13 >> lane) & 1) << 3);
        q2 |= (((b23 >> lane) & 1) << 3);
        q3 |= (((b33 >> lane) & 1) << 3);

        q0 |= (((b04 >> lane) & 1) << 4);
        q1 |= (((b14 >> lane) & 1) << 4);
        q2 |= (((b24 >> lane) & 1) << 4);
        q3 |= (((b34 >> lane) & 1) << 4);

        q0 |= (((b05 >> lane) & 1) << 5);
        q1 |= (((b15 >> lane) & 1) << 5);
        q2 |= (((b25 >> lane) & 1) << 5);
        q3 |= (((b35 >> lane) & 1) << 5);

        q0 |= (((b06 >> lane) & 1) << 6);
        q1 |= (((b16 >> lane) & 1) << 6);
        q2 |= (((b26 >> lane) & 1) << 6);
        q3 |= (((b36 >> lane) & 1) << 6);
        
        q0 |= (((b07 >> lane) & 1) << 7);
        q1 |= (((b17 >> lane) & 1) << 7);
        q2 |= (((b27 >> lane) & 1) << 7);
        q3 |= (((b37 >> lane) & 1) << 7);
    }

    // -------------------------------------------------------
    // 4. Dequantize back to float
    //
    // q ∈ [0, (1<<bits)-1]
    // x = q/scale + minv
    //
    //  x = q/scale + minv
    // -------------------------------------------------------
    // const float alpha = sqrt(batchSize / 8.0f);

    float v0 = (q0 * (1.f / s) + m);
    float v1 = (q1 * (1.f / s) + m);
    float v2 = (q2 * (1.f / s) + m);
    float v3 = (q3 * (1.f / s) + m);
    // float v0 = (q0 * (1.f / s));
    // float v1 = (q1 * (1.f / s));
    // float v2 = (q2 * (1.f / s));
    // float v3 = (q3 * (1.f / s));

    // v0 = lightReshape<scalar_t>(v0) * alpha;
    // v1 = lightReshape<scalar_t>(v1) * alpha;
    // v2 = lightReshape<scalar_t>(v2) * alpha;
    // v3 = lightReshape<scalar_t>(v3) * alpha;

    // -------------------------------------------------------
    // 5. Store 4 output values
    // Mapping identical to pack kernel:
    //
    // element_id = gid*256 + warp*128 + lane*4 + {0,1,2,3}
    // -------------------------------------------------------
    int out_base = gid * group_size + warp * 128 + lane * 4;

    unpack[out_base + 0] = static_cast<scalar_t>(v0);
    unpack[out_base + 1] = static_cast<scalar_t>(v1);
    unpack[out_base + 2] = static_cast<scalar_t>(v2);
    unpack[out_base + 3] = static_cast<scalar_t>(v3);

}

