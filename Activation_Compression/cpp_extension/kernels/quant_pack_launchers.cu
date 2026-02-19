#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <cuda_runtime.h>
#include "quant_pack_dispatch.h"

#include "quant_pack_1bit_sm89.cuh"
#include "quant_pack_1bit_sm90.cuh"

// ===============================================================
// External CUDA kernel signatures
// (These must match your .cu file kernel names exactly)
// ===============================================================

// ========== 1-bit ==========
extern "C" __global__ void quant_pack_1bit_sm80_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_1bit_sm90a_kernel(const float*, int32_t*, int, int, uint32_t);

// ========== 2-bit ==========
extern "C" __global__ void quant_pack_2bit_sm80_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_2bit_sm89_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_2bit_sm90_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_2bit_sm90a_kernel(const float*, int32_t*, int, int, uint32_t);

// ========== 4-bit ==========
extern "C" __global__ void quant_pack_4bit_sm80_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_4bit_sm89_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_4bit_sm90_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_4bit_sm90a_kernel(const float*, int32_t*, int, int, uint32_t);

// ========== 8-bit ==========
extern "C" __global__ void quant_pack_8bit_sm80_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_8bit_sm89_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_8bit_sm90_kernel (const float*, int32_t*, int, int, uint32_t);
extern "C" __global__ void quant_pack_8bit_sm90a_kernel(const float*, int32_t*, int, int, uint32_t);



// ===============================================================
// Launch helper (same for all kernels)
// ===============================================================
static inline void launch(dim3 grid, dim3 block,
                          const float* input,
                          int32_t* output,
                          int N, int G, uint32_t seed,
                          void (*kernel)(const float*, int32_t*, int, int, uint32_t))
{
    kernel<<<grid, block>>>(input, output, N, G, seed);
}



// ===============================================================
// 1-bit wrappers
// ===============================================================
void launch_pack_1bit_sm80(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_1bit_sm80_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        N, G, seed);
}

void launch_pack_1bit_sm89(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(32);

    // auto dtype = in.scalar_type();
    // if (dtype == at::kFloat8_e4m3fn) {
    //     quant_pack_1bit_sm89_kernel<c10::Float8_e4m3fn><<<grid, block>>>(
    //         in.data_ptr<c10::Float8_e4m3fn>(),
    //         out.data_ptr<int32_t>(),
    //         scaler.view({N*G}).data_ptr<c10::Float8_e4m3fn>(),
    //         min.view({N*G}).data_ptr<c10::Float8_e4m3fn>(),
    //         N, G, seed
    //     );
    // } else if (dtype == at::kFloat8_e5m2) {
    //     quant_pack_1bit_sm89_kernel<c10::Float8_e5m2><<<grid, block>>>(
    //         in.data_ptr<c10::Float8_e5m2>(),
    //         out.data_ptr<int32_t>(),
    //         scaler.view({N*G}).data_ptr<c10::Float8_e5m2>(),
    //         min.view({N*G}).data_ptr<c10::Float8_e5m2>(),
    //         N, G, seed
    //     );
    // } else {
    //     AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
    //                                     in.scalar_type(),
    //                                     "pack_1bit_sm89",
    //                                     [&] {
    //         quant_pack_1bit_sm89_kernel<scalar_t><<<grid, block>>>(
    //             in.data_ptr<scalar_t>(),
    //             out.data_ptr<int32_t>(),
    //             scaler.view({N*G}).data_ptr<scalar_t>(),
    //             min.view({N*G}).data_ptr<scalar_t>(),
    //             N, G, seed);
    //     });
    // }
}

// ========================================
// Explicit template instantiations
// ========================================
template void launch_pack_1bit_sm90<1>(
    torch::Tensor, torch::Tensor, torch::Tensor, int, int, uint32_t, torch::Tensor, float);

template void launch_pack_1bit_sm90<2>(
    torch::Tensor, torch::Tensor, torch::Tensor,  int, int, uint32_t, torch::Tensor, float);

template void launch_pack_1bit_sm90<3>(
    torch::Tensor, torch::Tensor, torch::Tensor, int, int, uint32_t, torch::Tensor, float);

template void launch_pack_1bit_sm90<4>(
    torch::Tensor, torch::Tensor, torch::Tensor, int, int, uint32_t, torch::Tensor, float);

template void launch_pack_1bit_sm90<8>(
    torch::Tensor, torch::Tensor, torch::Tensor, int, int, uint32_t, torch::Tensor, float);

template<int Bits>
void launch_pack_1bit_sm90(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, int N, int G, uint32_t seed, torch::Tensor min, float beta) {
    dim3 grid(N * G), block(64);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (in.scalar_type()) {

        case at::ScalarType::Float: {
            quant_pack_1bit_sm90_kernel<float, Bits><<<grid, block, 0, stream>>>(
                in.data_ptr<float>(),
                out.data_ptr<int32_t>(),
                scaler.view({N*G}).data_ptr<float>(),
                N, G, seed, min.data_ptr<float>(), beta
            );
            break;
        }

        case at::ScalarType::Half: {
            quant_pack_1bit_sm90_kernel<c10::Half, Bits><<<grid, block, 0, stream>>>(
                in.data_ptr<c10::Half>(),
                out.data_ptr<int32_t>(),
                scaler.view({N*G}).data_ptr<c10::Half>(),
                N, G, seed, min.data_ptr<c10::Half>(), beta
            );
            break;
        }

        case at::ScalarType::BFloat16: {
            quant_pack_1bit_sm90_kernel<c10::BFloat16, Bits><<<grid, block, 0, stream>>>(
                in.data_ptr<c10::BFloat16>(),
                out.data_ptr<int32_t>(),
                scaler.view({N*G}).data_ptr<c10::BFloat16>(),
                N, G, seed, min.data_ptr<c10::BFloat16>(), beta
            );
            break;
        }

        case at::ScalarType::Float8_e4m3fn: {
            quant_pack_1bit_sm90_kernel<at::Float8_e4m3fn, Bits><<<grid, block, 0, stream>>>(
                in.data_ptr<at::Float8_e4m3fn>(),
                out.data_ptr<int32_t>(),
                scaler.view({N*G}).data_ptr<at::Float8_e4m3fn>(),
                N, G, seed, min.data_ptr<at::Float8_e4m3fn>(), beta
            );
            break;
        }

        case at::ScalarType::Float8_e5m2: {
            quant_pack_1bit_sm90_kernel<at::Float8_e5m2, Bits><<<grid, block, 0, stream>>>(
                in.data_ptr<at::Float8_e5m2>(),
                out.data_ptr<int32_t>(),
                scaler.view({N*G}).data_ptr<at::Float8_e5m2>(),
                N, G, seed, min.data_ptr<at::Float8_e5m2>(), beta
            );
            break;
        }

        default:
            TORCH_CHECK(false,
                "pack_1bit_sm90 does not support dtype: ",
                in.scalar_type());
    }
}

void launch_pack_1bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_1bit_sm90a_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        N, G, seed);
}



// ===============================================================
// 2-bit wrappers
// ===============================================================
void launch_pack_2bit_sm80(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_2bit_sm80_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        N, G, seed);
}

void launch_pack_2bit_sm89(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_2bit_sm89_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        N, G, seed);
}

void launch_pack_2bit_sm90(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_2bit_sm90_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        N, G, seed);
}

void launch_pack_2bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_2bit_sm90a_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        N, G, seed);
}



// ===============================================================
// 4-bit wrappers
// ===============================================================
void launch_pack_4bit_sm80(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_4bit_sm80_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}

void launch_pack_4bit_sm89(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_4bit_sm89_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}

void launch_pack_4bit_sm90(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_4bit_sm90_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}

void launch_pack_4bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_4bit_sm90a_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}



// ===============================================================
// 8-bit wrappers
// ===============================================================
void launch_pack_8bit_sm80(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_8bit_sm80_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}

void launch_pack_8bit_sm89(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_8bit_sm89_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}

void launch_pack_8bit_sm90(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_8bit_sm90_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}

void launch_pack_8bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed) {
    dim3 grid(N * G), block(512);
    quant_pack_8bit_sm90a_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<int32_t>(), N, G, seed);
}
