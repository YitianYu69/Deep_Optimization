#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <ATen/ATen.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
// 1. DEFAULT SAFE LOAD MACROS (used on SM80, SM89, PTX compile mode)
////////////////////////////////////////////////////////////////////////////////

// FP32 / FP8 vector load fallback (16 bytes)
#define SM90_LOAD_16B(r0,r1,r2,r3,ptr) do {            \
    const uint4 v = *reinterpret_cast<const uint4*>(ptr); \
    r0 = v.x; r1 = v.y; r2 = v.z; r3 = v.w;             \
} while(0)

// FP16/BF16 fallback (8 bytes)
#define SM90_LOAD_8B(r0,r1,ptr) do {                   \
    const uint2 v = *reinterpret_cast<const uint2*>(ptr); \
    r0 = v.x; r1 = v.y;                                \
} while(0)

// int32
#define SM90_LOAD_INT4(r0,r1,r2,r3,ptr) do { \
    const uint4 v = *reinterpret_cast<const uint4*>(ptr); \
    r0 = v.x; r1 = v.y; r2 = v.z; r3 = v.w; \
} while(0)


////////////////////////////////////////////////////////////////////////////////
// 2. HOPPER SM90 ACCELERATED LOADS (SASS-only, guarded by __CUDA_ARCH__)
////////////////////////////////////////////////////////////////////////////////
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#undef SM90_LOAD_16B
#undef SM90_LOAD_8B
#undef SM90_LOAD_INT4

// 16B → L2::128B streaming load (best for FP32 and FP8)
#define SM90_LOAD_16B(r0,r1,r2,r3,ptr)                  \
    asm volatile(                                       \
        "ld.global.nc.L1::no_allocate.L2::128B.v4.b32 " \
        "{%0,%1,%2,%3}, [%4];"                          \
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)        \
        : "l"(ptr));

// 8B → L2::128B load (still good for FP16/BF16 even with waste)
#define SM90_LOAD_8B(r0,r1,ptr)                         \
    asm volatile(                                       \
        "ld.global.nc.L1::no_allocate.L2::128B.v2.b32 " \
        "{%0,%1}, [%2];"                                \
        : "=r"(r0), "=r"(r1)                            \
        : "l"(ptr));


// int4
#define SM90_LOAD_INT4(r0,r1,r2,r3,ptr)                 \
    asm volatile(                                       \
        "ld.global.nc.L1::no_allocate.L2::128B.v4.b32"  \
        "{%0,%1,%2,%3}, [%4];"                          \
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)        \
        : "l"(ptr));                          

#endif  // __CUDA_ARCH__ >= 900


////////////////////////////////////////////////////////////////////////////////
// 3. Vec128 Template Declaration
////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Vec128;   // Primary template


////////////////////////////////////////////////////////////////////////////////
// 4. FP32 Loader — 16 bytes
////////////////////////////////////////////////////////////////////////////////
template<>
struct Vec128<float> {
    using vec_t = float4;

    __device__ __forceinline__
    static vec_t load(const float* addr) {
        uint32_t r0,r1,r2,r3;
        SM90_LOAD_16B(r0,r1,r2,r3,addr);

        union { float4 f4; uint32_t u32[4]; } u;
        u.u32[0] = r0; u.u32[1] = r1;
        u.u32[2] = r2; u.u32[3] = r3;
        return u.f4;
    }

    __device__ __forceinline__
    static float lane(const vec_t& v, int i) {
        return ((const float*)&v)[i];
    }
};


////////////////////////////////////////////////////////////////////////////////
// 5. FP16 Loader — 8 bytes → converts to 4 floats
////////////////////////////////////////////////////////////////////////////////
template<>
struct Vec128<c10::Half> {
    using vec_t = uint2;

    __device__ __forceinline__
    static vec_t load(const c10::Half* addr) {
        uint32_t r0,r1;
        SM90_LOAD_8B(r0,r1,addr);
        return uint2{r0,r1};
    }

    __device__ __forceinline__
    static float lane(const vec_t& v, int i) {
        uint16_t raw =
            (i < 2) ?  ((v.x >> (16*i)) & 0xFFFFu)
                    : ((v.y >> (16*(i-2))) & 0xFFFFu);

        return __half2float(__ushort_as_half(raw));
    }
};


////////////////////////////////////////////////////////////////////////////////
// 6. BF16 Loader — 8 bytes → converts to 4 floats
////////////////////////////////////////////////////////////////////////////////
template<>
struct Vec128<c10::BFloat16> {
    using vec_t = uint2;

    __device__ __forceinline__
    static vec_t load(const c10::BFloat16* addr) {
        uint32_t r0,r1;
        SM90_LOAD_8B(r0,r1,addr);
        return uint2{r0,r1};
    }

    __device__ __forceinline__
    static float lane(const vec_t& v, int i) {
        // Extract raw 16-bit BF16 bits
        uint16_t raw =
            (i < 2) ?  ((v.x >> (16*i)) & 0xFFFFu)
                    : ((v.y >> (16*(i-2))) & 0xFFFFu);

        // Correct bit reinterpretation (CUDA 12-safe)
        union {
            uint16_t u16;
            __nv_bfloat16 bf;
        } u;
        u.u16 = raw;

        return __bfloat162float(u.bf);
    }
};


////////////////////////////////////////////////////////////////////////////////
// 7. FP8 (e4m3fn) Loader — 16 bytes
////////////////////////////////////////////////////////////////////////////////
template<>
struct Vec128<at::Float8_e4m3fn> {
    using vec_t = uint4;

    __device__ __forceinline__
    static vec_t load(const at::Float8_e4m3fn* addr) {
        uint32_t r0,r1,r2,r3;
        SM90_LOAD_16B(r0,r1,r2,r3,addr);
        return uint4{r0,r1,r2,r3};
    }

    __device__ __forceinline__
    static float lane(const vec_t& v, int i) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        return float(at::Float8_e4m3fn(p[i]));
    }
};


////////////////////////////////////////////////////////////////////////////////
// 8. FP8 (e5m2) Loader — 16 bytes
////////////////////////////////////////////////////////////////////////////////
template<>
struct Vec128<at::Float8_e5m2> {
    using vec_t = uint4;

    __device__ __forceinline__
    static vec_t load(const at::Float8_e5m2* addr) {
        uint32_t r0,r1,r2,r3;
        SM90_LOAD_16B(r0,r1,r2,r3,addr);
        return uint4{r0,r1,r2,r3};
    }

    __device__ __forceinline__
    static float lane(const vec_t& v, int i) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        return float(at::Float8_e5m2(p[i]));
    }
};


////////////////////////////////////////////////////////////////////////////////
// Vec128<int32_t> — 16B load on SM90
////////////////////////////////////////////////////////////////////////////////
template<>
struct Vec128<int32_t> {
    using vec_t = int4;

    __device__ __forceinline__
    static vec_t load(const int32_t *addr) {
        uint32_t r0, r1, r2, r3;
        SM90_LOAD_INT4(r0, r1, r2, r3, addr);
        
        int4 out;
        out.x = static_cast<int32_t>(r0);
        out.y = static_cast<int32_t>(r1);
        out.z = static_cast<int32_t>(r2);
        out.w = static_cast<int32_t>(r3);
        return out;
    }

    __device__ __forceinline__
    static int lane(const vec_t& v, int i) {
        return ((const int*)&v)[i];
    }
};
