#pragma once 
#include "quant_common.h" 
#include "vector_load.h" 


// ============================================================================ 
// SM90 1-bit quantizer (group_size = 256 = 4 vals × 64 threads) 
// - 2 warps per group (blockDim.x = 64) 
// - vector load via Vec128<scalar_t> (4 vals / thread) 
// - warp-wide min/max using __reduce_* intrinsics + shared memory 
// - stochastic rounding via sgd_noise(key) 
// - 1-bit quant (4 bits/thread → 256 bits/group) 
// - 8 ballots per group (4 per warp) → 8 × u32 
// - FP32 scales/mins for all dtypes (AMP-safe) 
// - Structured so the "group body" can be transplanted into a CUTLASS epilogue 
// ============================================================================ 

template<typename scalar_t, int bits> 
__global__ void quant_pack_1bit_sm90_kernel(const scalar_t* __restrict__ input, // [N * groups * 256] 
                                            int32_t* __restrict__ output, // packed output: [N * groups * 8] 
                                            scalar_t* __restrict__ scaler, // scale: [N * groups] (fp32) 
                                            int N, 
                                            int groups, 
                                            uint32_t seed,
                                            scalar_t* __restrict__ min,
                                            float beta) { 
    #if __CUDA_ARCH__ < 900 
    // SM90-only kernel; guard just in case 
        return; 
    #endif 
    // ------------------------------------------------------------------------ 
    // Thread / warp / group indices 
    // ------------------------------------------------------------------------ 
    const int t = threadIdx.x; // 0..63 
    const int lane = t & 31; // lane within warp (0..31) 
    const int warp = t >> 5; // warp 0 or 1 
    const int gid = blockIdx.x; // group index: 0..N*groups-1 
    if (gid >= N * groups) 
        return; 
    
    constexpr int group_size = 256; 
    constexpr int vals_per_thread = 4; // 64 * 4 = 256 
    const int group_base = gid * group_size; 
    
    // ------------------------------------------------------------------------ 
    // Typedefs for vectorized load 
    // ------------------------------------------------------------------------ 
    using Loader = Vec128<scalar_t>; 
    using V = typename Loader::vec_t; 
    
    
    // ------------------------------------------------------------------------ 
    // Shared memory for 2-warp min/max reduction 
    // ------------------------------------------------------------------------ 
    __shared__ float s_min[2]; 
    __shared__ float s_max[2];
    
    // ------------------------------------------------------------------------ 
    // 1) Vector load: each thread loads 4 contiguous elements 
    // ------------------------------------------------------------------------ 
    const scalar_t* addr = input + group_base + t * vals_per_thread; 
    V v = Loader::load(addr); 
    

    float x[vals_per_thread]; 
    #pragma unroll 
    for (int i = 0; i < vals_per_thread; ++i) {
        x[i] = Loader::lane(v, i); 
    }
    

    // ------------------------------------------------------------------------ 
    // 2) Local min/max over the 4 values in this thread 
    // ------------------------------------------------------------------------ 
    float lo = fminf(fminf(x[0], x[1]), fminf(x[2], x[3])); 
    float hi = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));

    
    // ------------------------------------------------------------------------ 
    // 3) Warp-wide reduction (per-warp min/max) 
    // ------------------------------------------------------------------------ 
    float warp_minv = warp_min(lo);  // float → float 
    float warp_maxv = warp_max(hi); // float → float 

    
    if (lane == 0) { 
        s_min[warp] = warp_minv; 
        s_max[warp] = warp_maxv; 
    } 
    __syncthreads(); 

    
    // ------------------------------------------------------------------------ 
    // 4) Block-wide reduction across the 2 warps (256 elements total) 
    // ------------------------------------------------------------------------ 
    float minv, maxv;
    if (t == 0) { 
        float m0 = s_min[0]; 
        float m1 = s_min[1]; 
        float M0 = s_max[0]; 
        float M1 = s_max[1]; 
        
        minv = fminf(m0, m1); 
        maxv = fmaxf(M0, M1); 

        s_min[0] = minv;
        s_max[0] = maxv;

        min[gid] = static_cast<scalar_t>(minv);
    } 
    __syncthreads(); 
    
    minv = s_min[0]; 
    maxv = s_max[0];
    

    float denom = maxv - minv + 2e-6f; // For general Bits, scale = ((1 << Bits) - 1) / denom. 

    // Here Bits=1, so this is just 1 / denom but written in a fusion-friendly way. 
    float scale = float((1 << bits) - 1) / denom; 
    // float scale = 1.0f / sumv;
    
    // ------------------------------------------------------------------------ 
    // 5) Stochastic rounding + 1-bit quantization (4 bits/thread) 
    // ------------------------------------------------------------------------ 
    uint32_t prePackedBits = 0;
    #pragma unroll 
    for (int i = 0; i < vals_per_thread; ++i) { 
        // Global element index within the whole tensor 
        const int elem_idx = group_base + t * vals_per_thread + i;

        // 64-bit key → stable per element (fusion-friendly; you can reuse this 
        // pattern in a GEMM epilogue since it's purely index-based). 
        
        uint64_t key = (static_cast<uint64_t>(elem_idx) << 32) ^ static_cast<uint64_t>(seed); 
        float noise = sgd_noise(key); // from quant_common.h 
        
        int q = quantize_val<bits>(x[i], minv, scale, noise); 
        
        if constexpr (bits == 1) {
            prePackedBits |= (q & 1u) << i; // pack 4 bits in low nibble 
        } else if constexpr (bits == 2) {
            prePackedBits |= (q & 1u) << i; 
            prePackedBits |= ((q >> 1) & 1u) << (4 + i);
        } else if constexpr (bits == 3) {
            prePackedBits |= (q & 1u) << i;
            prePackedBits |= ((q >> 1) & 1u) << (4 + i);
            prePackedBits |= ((q >> 2) & 1u) << (8 + i);
        } else if constexpr (bits == 4) {
            prePackedBits |= (q & 1u) << i;
            prePackedBits |= ((q >> 1) & 1u) << (4 + i);
            prePackedBits |= ((q >> 2) & 1u) << (8 + i);
            prePackedBits |= ((q >> 3) & 1u) << (12 + i);
        } else if constexpr (bits == 8) {
            prePackedBits |= (q & 1u) << i;
            prePackedBits |= ((q >> 1) & 1u) << (4 + i);
            prePackedBits |= ((q >> 2) & 1u) << (8 + i);
            prePackedBits |= ((q >> 3) & 1u) << (12 + i);
            prePackedBits |= ((q >> 4) & 1u) << (16 + i);
            prePackedBits |= ((q >> 5) & 1u) << (20 + i);
            prePackedBits |= ((q >> 6) & 1u) << (24 + i);
            prePackedBits |= ((q >> 7) & 1u) << (28 + i);
        }
    } 
        
        
    // ------------------------------------------------------------------------ 
    // 6) Warp-local ballots (each warp handles 128 elements → 4 u32 words) 
    // ------------------------------------------------------------------------ 
    auto p = bits_pack(prePackedBits, 0);

    Pack4 p2, p3, p4, p5, p6, p7, p8;
    if constexpr (bits == 2) {
        p2 = bits_pack(prePackedBits, 4);
    } else if constexpr (bits == 3) {
        p2 = bits_pack(prePackedBits, 4);
        p3 = bits_pack(prePackedBits, 8);
    } else if constexpr (bits == 4) {
        p2 = bits_pack(prePackedBits, 4);
        p3 = bits_pack(prePackedBits, 8);
        p4 = bits_pack(prePackedBits, 12);
    } else if constexpr (bits == 8) {
        p2 = bits_pack(prePackedBits, 4);
        p3 = bits_pack(prePackedBits, 8);
        p4 = bits_pack(prePackedBits, 12);
        p5 = bits_pack(prePackedBits, 16);
        p6 = bits_pack(prePackedBits, 20);
        p7 = bits_pack(prePackedBits, 24);
        p8 = bits_pack(prePackedBits, 28);
    }
    
    // ------------------------------------------------------------------------ 
    // 7) Store packed bits 
    // 
    // Each group outputs 8 x u32: 
    // - warp 0: [b0, b1, b2, b3] for elements 0..127 
    // - warp 1: [b0, b1, b2, b3] for elements 128..255 
    // 
    // Layout: output[group * 8 + (warp * 4 + 0..3)] 
    // ------------------------------------------------------------------------ 
    if (lane == 0) { 
        const int out_base = gid * (8 * bits) + warp * 4 * bits; 
        uint4 pack = { p.b0, p.b1, p.b2, p.b3 }; 
        *reinterpret_cast<uint4*>(&output[out_base]) = pack; 

        uint4 pack2, pack3, pack4, pack5, pack6, pack7;

        if constexpr (bits == 2) {
            pack = {p2.b0, p2.b1, p2.b2, p2.b3};
            *reinterpret_cast<uint4*>(&output[out_base + 4]) = pack;
        } else if constexpr (bits == 3) {
            pack = {p2.b0, p2.b1, p2.b2, p2.b3};
            pack2 = {p3.b0, p3.b1, p3.b2, p3.b3};

            *reinterpret_cast<uint4*>(&output[out_base + 4]) = pack;
            *reinterpret_cast<uint4*>(&output[out_base + 8]) = pack2;
        } else if constexpr (bits == 4) {
            pack = {p2.b0, p2.b1, p2.b2, p2.b3};
            pack2 = {p3.b0, p3.b1, p3.b2, p3.b3};
            pack3 = {p4.b0, p4.b1, p4.b2, p4.b3};

            *reinterpret_cast<uint4*>(&output[out_base + 4]) = pack;
            *reinterpret_cast<uint4*>(&output[out_base + 8]) = pack2;
            *reinterpret_cast<uint4*>(&output[out_base + 12]) = pack3;
        } else if constexpr (bits == 8) {
            pack = {p2.b0, p2.b1, p2.b2, p2.b3};
            pack2 = {p3.b0, p3.b1, p3.b2, p3.b3};
            pack3 = {p4.b0, p4.b1, p4.b2, p4.b3};
            pack4 = {p5.b0, p5.b1, p5.b2, p5.b3};
            pack5 = {p6.b0, p6.b1, p6.b2, p6.b3};
            pack6 = {p7.b0, p7.b1, p7.b2, p7.b3};
            pack7 = {p8.b0, p8.b1, p8.b2, p8.b3};

            *reinterpret_cast<uint4*>(&output[out_base + 4]) = pack;
            *reinterpret_cast<uint4*>(&output[out_base + 8]) = pack2;
            *reinterpret_cast<uint4*>(&output[out_base + 12]) = pack3;
            *reinterpret_cast<uint4*>(&output[out_base + 16]) = pack4;
            *reinterpret_cast<uint4*>(&output[out_base + 20]) = pack5;
            *reinterpret_cast<uint4*>(&output[out_base + 24]) = pack6;
            *reinterpret_cast<uint4*>(&output[out_base + 28]) = pack7;
        }
    } 
    
    // ------------------------------------------------------------------------ 
    // 8) Store FP32 scale/min (once per group) 
    // ------------------------------------------------------------------------ 
    if (t == 0) { 
        scaler[gid] = static_cast<scalar_t>(scale);
    }

}