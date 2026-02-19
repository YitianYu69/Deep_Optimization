#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <stdint.h>

#define FULL_MASK 0xffffffffu

// ======================================================
//  ARCH CHECK
// ======================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#error "This quantizer requires Volta or newer."
#endif

// ======================================================
//  FAST XORSHIFT32 RNG — 100% in registers, warp-safe
// ======================================================
__device__ __forceinline__ float xorshift32_uniform(uint32_t &state) {
    // xorshift32
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state <<  5;
    uint32_t v = state & 0x00FFFFFF;  // 24 bits for mantissa
    return float(v) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float sgd_noise(uint64_t x) {
    // wyhash64 step
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ULL;
    x ^= x >> 32;

    // uniform [0,1)
    float u = (x >> 40) * (1.0f / (1ULL << 24));

    // convert to [-0.5, +0.5]
    return u - 0.5f;
}

// ======================================================
//  Warp-level min/max reductions
// ======================================================
__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off));
    return x;
}

__device__ __forceinline__ float warp_min(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        x = fminf(x, __shfl_xor_sync(0xffffffffu, x, off));
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>=1) {
        x += __shfl_down_sync(0xffffffffu, x, off);
    }
    return x;
}

// ======================================================
//  Shared-memory group reduction (group = 512 threads)
// ======================================================
__device__ __forceinline__
void group_minmax_512(float &minv, float &maxv) {
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    // Shared scratch for 16 warps
    __shared__ float shm_min[16];
    __shared__ float shm_max[16];

    // Warp-level reduction
    float w_min = warp_min(minv);
    float w_max = warp_max(maxv);

    if (lane == 0) {
        shm_min[warpId] = w_min;
        shm_max[warpId] = w_max;
    }
    __syncthreads();

    // warp 0 final reduction
    if (warpId == 0) {
        float fm = (lane < 16) ? shm_min[lane] : 1e30f;
        float fM = (lane < 16) ? shm_max[lane] : -1e30f;

        fm = warp_min(fm);
        fM = warp_max(fM);

        if (lane == 0) {
            shm_min[0] = fm;
            shm_max[0] = fM;
        }
    }
    __syncthreads();

    minv = shm_min[0];
    maxv = shm_max[0];
}

// ======================================================
//  FAST BITPACK: inline PTX ballot (faster on SM90)
// ======================================================
__device__ __forceinline__
uint32_t ptx_ballot(int pred) {
#if __CUDA_ARCH__ >= 900
    // SM90 optimized ballot
    uint32_t out;
    asm volatile (
        "{ .reg .pred p;        \n"
        "setp.ne.b32 p, %1, 0;  \n"
        "vote.ballot.sync.b32 %0, p, 0xffffffff; \n"
        "}" : "=r"(out) : "r"(pred)
    );
    return out;
#else
    return __ballot_sync(0xffffffffu, pred);
#endif
}

template<typename scalar_t>
__device__ __forceinline__
float signf(scalar_t val) {
    return copysignf(1.0f, static_cast<float>(val));
}

// ======================================================
//  VECTOR STORE (128-bit) — use on SM80+
// ======================================================
__device__ __forceinline__
void store128(int32_t* base, int idx, int4 v) {
    *reinterpret_cast<int4*>(&base[idx]) = v;
}

// ======================================================
// Template: quantize function per bit-width
// ======================================================
template<int bits>
__device__ __forceinline__
int quantize_val(float x, float minv, float scale, float noise) {
    float qf = (x - minv) * scale + noise - 0.5f;
    int q = __float2int_rn(qf);
    q = max(0, min(q, (1<<bits)-1));

    return q;
}

__device__ __forceinline__
int quantize_1_bit(float x, float scale, float noise) {
    // float qf = x * scale + noise - 0.5f;
    float temp_q = signf(x);
    int q = __float2int_rn(temp_q);
    return q;
}

// ======================================================
// Template: Pack bits
// ======================================================
struct Pack4 {
    uint32_t b0, b1, b2, b3;
};


__device__ __forceinline__
Pack4 bits_pack(uint32_t prePackedBits, int offs) {
    Pack4 p;
    p.b0 = __ballot_sync(FULL_MASK, (prePackedBits >> (0 + offs)) & 1u);
    p.b1 = __ballot_sync(FULL_MASK, (prePackedBits >> (1 + offs)) & 1u);
    p.b2 = __ballot_sync(FULL_MASK, (prePackedBits >> (2 + offs)) & 1u);
    p.b3 = __ballot_sync(FULL_MASK, (prePackedBits >> (3 + offs)) & 1u);

    return p;
}

// ======================================================
// Template: Activation Smoother
// ======================================================
template<typename scalar_t>
__device__ __forceinline__
scalar_t lightReshape(float a) {
    float beta = 0.02f;
    return a * rsqrtf(1.0f + beta * a * a); 
}


// ======================================================
// Template: Sm89 loading
// ======================================================
template<typename scalar_t>
__device__ __forceinline__
float sm89StreamLoad(const scalar_t* addr);

template<>
__device__ __forceinline__
float sm89StreamLoad<float>(const float* addr) {
    float out;
    asm volatile ("ld.global.cs.f32 %0, [%1];\n"
                  : "=f"(out)
                  : "l"(addr)
    );
    return out;
}

template<>
__device__ __forceinline__
float sm89StreamLoad<c10::Half>(const c10::Half* addr) {
    unsigned short tmp;
    asm volatile ("ld.global.cs.u16 %0, [%1];\n"
                  : "=h"(tmp)
                  : "l"(addr)
    );
    __half out = *reinterpret_cast<const __half*>(&tmp);
    return __half2float(out);
}

template<>
__device__ __forceinline__
float sm89StreamLoad<c10::BFloat16>(const c10::BFloat16* addr) {
    unsigned short tmp;
    asm volatile ("ld.global.cs.u16 %0, [%1];\n"
                  : "=h"(tmp)
                  : "l"(addr)
    );
    __nv_bfloat16 out = *reinterpret_cast<const __nv_bfloat16*>(&tmp);
    return __bfloat162float(out);
}


template<>
__device__ __forceinline__
float sm89StreamLoad<double>(const double* addr) {
    double tmp;
    asm volatile("ld.global.cs.f64 %0, [%1];\n"
                 : "=d"(tmp)
                 : "l"(addr));
    return (float)tmp;
}

template<>
__device__ __forceinline__
float sm89StreamLoad<at::Float8_e4m3fn>(const at::Float8_e4m3fn* addr)
{
    uint32_t tmp32;
    asm volatile(
        "ld.global.cs.u8 %0, [%1];\n"
        : "=r"(tmp32)
        : "l"(addr)
    );

    uint8_t byte = static_cast<uint8_t>(tmp32);
    at::Float8_e4m3fn fp8_val(byte);
    return static_cast<float>(fp8_val);
}

template<>
__device__ __forceinline__
float sm89StreamLoad<at::Float8_e5m2>(const at::Float8_e5m2* addr)
{
    uint32_t tmp32;
    asm volatile(
        "ld.global.cs.u8 %0, [%1];\n"
        : "=r"(tmp32)
        : "l"(addr)
    );

    uint8_t byte = static_cast<uint8_t>(tmp32);
    at::Float8_e5m2 fp8_val(byte);
    return static_cast<float>(fp8_val);
}