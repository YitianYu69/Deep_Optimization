#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>       
#include <c10/cuda/CUDAException.h>    
#include <cfloat>

#include <utility>


#define MASK 0xffffffffu


__forceinline__ __device__ std::pair<float, float> warpReduce(float val1, float val2) {
    val1 = fmaxf(val1, __shfl_down_sync(MASK, val1, 16));
    val1 = fmaxf(val1, __shfl_down_sync(MASK, val1, 8));
    val1 = fmaxf(val1, __shfl_down_sync(MASK, val1, 4));
    val1 = fmaxf(val1, __shfl_down_sync(MASK, val1, 2));
    val1 = fmaxf(val1, __shfl_down_sync(MASK, val1, 1));

    val2 = fminf(val2, __shfl_down_sync(MASK, val2, 16));
    val2 = fminf(val2, __shfl_down_sync(MASK, val2, 8));
    val2 = fminf(val2, __shfl_down_sync(MASK, val2, 4));
    val2 = fminf(val2, __shfl_down_sync(MASK, val2, 2));
    val2 = fminf(val2, __shfl_down_sync(MASK, val2, 1));

    return std::make_pair(val1, val2);
}

__device__ std::pair<float, float> maxMinBlockReduce(float val1, float val2) {
    __shared__ float shmem_max[32];
    __shared__ float shmem_min[32];

    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    float val_max = -FLT_MAX;
    float val_min = FLT_MAX;
    auto val_temp = warpReduce(val1, val2);
    if (lane == 0) {
        shmem_max[warpId] = val_temp.first;
        shmem_min[warpId] = val_temp.second;
    }
    __syncthreads();


    float block_max = -FLT_MAX;
    float block_min = FLT_MAX;
    if (warpId == 0) {
        int numWarps = (blockDim.x + 31) / 32;
        float local_max = (lane < numWarps) ? shmem_max[lane] : -FLT_MAX;
        float local_min = (lane < numWarps) ? shmem_min[lane] : FLT_MAX;
        auto finalPair = warpReduce(local_max, local_min);
        block_max = finalPair.first;
        block_min = finalPair.second;
    }
    // __syncthreads();
    
    float max = __shfl_sync(MASK, block_max, 0);
    float min = __shfl_sync(MASK, block_min, 0);
    return std::make_pair(max, min);
}



template <typename scalar_t>
__global__ void maxmin_reduce_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ out_max,
                                     scalar_t* __restrict__ out_min,
                                     int B, int N, int D) {

    // int n = blockIdx.x;
    int n = blockIdx.y * N + blockIdx.x;

    int d = threadIdx.x;

    if (blockIdx.y >= B || blockIdx.x >= N || d >= D) return;

    float v_max = -FLT_MAX;
    float v_min = FLT_MAX;
    for (int tid = d; tid < D; tid += blockDim.x) {
        float data = __ldg(&input[n * D + tid]);
        v_max = fmaxf(v_max, data);
        v_min = fminf(v_min, data);
    }
    // for (int tid = d; tid < N; tid += blockDim.x) {
    //     v  = fmax(v, __ldg(&data[tid * D + n]));
    // }

    auto v = maxMinBlockReduce(v_max, v_min);
    if (d == 0) {
        out_max[n] = v.first;
        out_min[n] = v.second;        
    }
}


__global__ void compute_scale(const float* __restrict__ input,
                              const float* __restrict__ max,
                              const float* __restrict__ min,
                              float* __restrict__ scale,
                              int bits,
                              int B,
                              int N) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * N) return;
    size_t demo = max[i] - min[i] + (float)2e-6;
    scale[i] = (float)((1 << bits) - 1) / demo;
}



std::tuple<torch::Tensor, torch::Tensor> min_max_kernel(torch::Tensor data) {
    TORCH_CHECK(data.is_cuda(), "data must be CUDA");
    TORCH_CHECK(data.dim() == 3, "data must be [B, N, D]");

    const int B = data.size(0);
    const int N = data.size(1);
    const int D = data.size(2);

    auto opts = data.options().dtype(data.dtype()).device(data.device());
    torch::Tensor max_t = torch::empty({B, N}, opts);
    torch::Tensor min_t = torch::empty({B, N}, opts);

    dim3 grid(N, B);
    int threads = 512;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
        data.scalar_type(), "maxmin_reduce_kernel", [&] {
            maxmin_reduce_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                data.data_ptr<scalar_t>(),
                max_t.view({B * N}).data_ptr<scalar_t>(),
                min_t.view({B * N}).data_ptr<scalar_t>(),
                B, N, D
            );
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(min_t, max_t);
}

