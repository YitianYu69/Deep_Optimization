#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#include <cuda_runtime.h>
#include "vector_load.h"



template<typename scalar_t>
__global__
void copy(const scalar_t* __restrict__ input,
          scalar_t* __restrict__ fake_unpack,
          int N) {

    const int gid = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = 2;

    constexpr int group_size = 256;
    const int group_base = gid * group_size;
    const int vals_per_thread = 4;


    if (gid >= N * group_size) return;


    using Loader = Vec128<scalar_t>; 
    using V = typename Loader::vec_t; 

    const scalar_t* addr = input + group_base + t * vals_per_thread; 
    V v = Loader::load(addr); 
    

    float x[vals_per_thread]; 
    #pragma unroll 
    for (int i = 0; i < vals_per_thread; ++i) {
        x[i] = Loader::lane(v, i); 
    }

    fake_unpack[gid * group_size + warp * 128 + lane * 4 + 0] = x[0];
    fake_unpack[gid * group_size + warp * 128 + lane * 4 + 1] = x[1];
    fake_unpack[gid * group_size + warp * 128 + lane * 4 + 2] = x[2];
    fake_unpack[gid * group_size + warp * 128 + lane * 4 + 3] = x[3];
}


void copy_launch(torch::Tensor input, torch::Tensor fake_unpack, int N, int G) {
    dim3 grid(N * G), block(64);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (input.scalar_type()) {

        case at::ScalarType::Float: {
            copy<float><<<grid, block, 0, stream>>>(
                input.data_ptr<float>(),
                fake_unpack.data_ptr<float>(),
                N
            );
            break;
        } 

        case at::ScalarType::Half: {
            copy<c10::Half><<<grid, block, 0, stream>>>(
                input.data_ptr<c10::Half>(),
                fake_unpack.data_ptr<c10::Half>(),
                N
            );
            break;
        }

        case at::ScalarType::BFloat16: {
            copy<c10::BFloat16><<<grid, block, 0, stream>>>(
                input.data_ptr<c10::BFloat16>(),
                fake_unpack.data_ptr<c10::BFloat16>(),
                N
            );
            break;
        }

        case at::ScalarType::Float8_e4m3fn: {
            copy<at::Float8_e4m3fn><<<grid, block, 0, stream>>>(
                input.data_ptr<at::Float8_e4m3fn>(),
                fake_unpack.data_ptr<at::Float8_e4m3fn>(),
                N
            );
            break;
        }

        case at::ScalarType::Float8_e5m2: {
            copy<at::Float8_e5m2><<<grid, block, 0, stream>>>(
                input.data_ptr<at::Float8_e5m2>(),
                fake_unpack.data_ptr<at::Float8_e5m2>(),
                N
            );
            break;
        }

    }
}