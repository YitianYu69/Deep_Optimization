#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <cuda_runtime.h>

#include "dequant_unpack_1bit_sm90.cuh"


template<int bits>
void launch_unpack_sm90(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize) {
    dim3 grid(N * G), block(64);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (scaler.scalar_type()) {

        case at::ScalarType::Float: {
            dequant_unpack_sm90_kernel<float, bits><<<grid, block, 0, stream>>>(
                in.data_ptr<int32_t>(),
                out.data_ptr<float>(),
                scaler.data_ptr<float>(),
                ema_min.data_ptr<float>(),
                N, G, batchSize
            );
            break;
        }

        case at::ScalarType::Half: {
            dequant_unpack_sm90_kernel<c10::Half, bits><<<grid, block, 0, stream>>>(
                in.data_ptr<int32_t>(),
                out.data_ptr<c10::Half>(),
                scaler.data_ptr<c10::Half>(),
                ema_min.data_ptr<c10::Half>(),
                N, G, batchSize
            );
            break;
        }

        case at::ScalarType::BFloat16: {
            dequant_unpack_sm90_kernel<c10::BFloat16, bits><<<grid, block, 0, stream>>>(
                in.data_ptr<int32_t>(),
                out.data_ptr<c10::BFloat16>(),
                scaler.data_ptr<c10::BFloat16>(),
                ema_min.data_ptr<c10::BFloat16>(),
                N, G, batchSize
            );

            break;
        }

        case at::ScalarType::Float8_e4m3fn: {
            dequant_unpack_sm90_kernel<at::Float8_e4m3fn, bits><<<grid, block, 0, stream>>>(
                in.data_ptr<int32_t>(),
                out.data_ptr<at::Float8_e4m3fn>(),
                scaler.data_ptr<at::Float8_e4m3fn>(),
                ema_min.data_ptr<at::Float8_e4m3fn>(),
                N, G, batchSize
            );

            break;
        }

        case at::ScalarType::Float8_e5m2: {
            dequant_unpack_sm90_kernel<at::Float8_e5m2, bits><<<grid, block, 0, stream>>>(
                in.data_ptr<int32_t>(),
                out.data_ptr<at::Float8_e5m2>(),
                scaler.data_ptr<at::Float8_e5m2>(),
                ema_min.data_ptr<at::Float8_e5m2>(),
                N, G, batchSize
            );

            break;
        }

        default:
            TORCH_CHECK(false,
                "unpack does not support current dtype!");
    }
}



template void launch_unpack_sm90<1>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
template void launch_unpack_sm90<2>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
template void launch_unpack_sm90<3>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
template void launch_unpack_sm90<4>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
template void launch_unpack_sm90<8>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);