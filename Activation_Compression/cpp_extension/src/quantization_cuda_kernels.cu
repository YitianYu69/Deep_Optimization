#include <torch/extension.h>
#include <torch/torch.h> 
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <curand_kernel.h>
#include <mutex>

#define BLOCK_Y_DIM_MAX ((((int64_t)(1)) << 16) - 1)

using torch::Tensor;

// ======================================================
// Compute scaling factors
// ======================================================
template<typename scalar_t>
__global__ void compute_scale_mixed_precision_kernel(
    const scalar_t* __restrict__ min,
    const scalar_t* __restrict__ max,
    scalar_t* __restrict__ scale,
    const int bits,
    int64_t N,
    int64_t num_groups)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * num_groups) return;
    scalar_t denom = max[i] - min[i] + 2e-6;
    scale[i] = (((1 << bits) - 1)) / denom;
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
// Pack kernel — warp-ballot version (no atomics)
// ======================================================
template <typename scalar_t>
__global__ void pack_mixed_precision_kernel(
    const scalar_t* __restrict__ data,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ min,
    int32_t* __restrict__ packed,
    const int bits,
    uint64_t seed,
    uint64_t offset,
    int64_t N,
    int64_t num_groups,
    int64_t group_size,
    int64_t block_idx_y_base)
{
    const int64_t n = blockIdx.y + block_idx_y_base;
    const int group_id = blockIdx.x;
    const int tid = threadIdx.x;
    if (n >= N || group_id >= num_groups || tid >= group_size)
        return;

    const unsigned lane = tid & 31;
    const unsigned warp_id = tid >> 5;
    const unsigned FULL_MASK = 0xffffffffu;
    const int warps_per_group = (group_size + 31) / 32;

    // load scale/min
    float s = static_cast<float>(__ldg(&scale[n * num_groups + group_id]));
    float m = static_cast<float>(__ldg(&min[n * num_groups + group_id]));
    // float s = __ldg(&scale[n * num_groups + group_id]);
    // float m = __ldg(&min[n * num_groups + group_id]);

    // RNG (Philox)
    int64_t linear_id = ((int64_t)n * num_groups + group_id) * group_size + tid;
    // curandStatePhilox4_32_10_t state;
    // curand_init(seed, linear_id, offset, &state);
    // float noise = curand_uniform(&state);
    // float noise = 1.0;
    uint64_t key = ((uint64_t)linear_id << 32) ^ (uint64_t)seed;
    float noise = sgd_noise(key);

    // quantize
    const float val = static_cast<float>(__ldg(&data[linear_id]));
    float qf = fmaxf((val - m) * s + noise - 0.5f, 0.0f);
    int q = __float2int_rn(qf);

    // warp-ballot bit-packing (direct global store)
    int64_t base_idx = ((int64_t)n * num_groups + group_id) *
                       warps_per_group * bits +
                       warp_id * bits;

    #pragma unroll
    for (int b = 0; b < bits; ++b) {
        unsigned pred = (q >> b) & 1u;
        unsigned ballot = __ballot_sync(FULL_MASK, pred);

        if (lane == 0) {
            packed[base_idx + b] = ballot;
        }
    }
}

// ======================================================
// Host launcher for pack
// ======================================================
std::pair<Tensor, Tensor> pack_mixed_precision_cuda(
    Tensor data,
    Tensor min,
    Tensor max,
    int bits,
    bool stochastic)
{
    TORCH_CHECK(data.is_cuda(), "data must be CUDA tensor");
    TORCH_CHECK(min.is_cuda() && max.is_cuda(),
                "min/max must be CUDA tensors");

    const int64_t N = data.size(0);
    const int64_t num_groups = data.size(1);
    const int64_t group_size = data.size(2);
    const int warps_per_group = (group_size + 31) / 32;

    auto options_int =
        torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    Tensor packed = torch::empty({N * num_groups * warps_per_group * bits},
                                 options_int);

    auto options =
        torch::TensorOptions().dtype(data.dtype()).device(data.device());
    Tensor scale = torch::empty({N, num_groups, 1}, options);

    // Compute scale
    int threads = 256;
    int blocks = (N * num_groups + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                    scale.scalar_type(),
                                    "compute_scale_mixed_precision", ([&] {
        compute_scale_mixed_precision_kernel<scalar_t>
            <<<blocks, threads>>>(
                min.data_ptr<scalar_t>(), max.data_ptr<scalar_t>(),
                scale.data_ptr<scalar_t>(), bits, N, num_groups);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    // RNG setup
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(
        at::cuda::detail::getDefaultCUDAGenerator());

    uint64_t seed, offset;
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 3)
    auto philox = gen->philox_cuda_state(N * num_groups * group_size);
    seed = philox.seed_.val;
    offset = philox.offset_.val;
#else
    std::lock_guard<std::mutex> lock(gen->mutex_);
    std::pair<uint64_t, uint64_t> engine_inputs =
        gen->philox_engine_inputs(N * num_groups * group_size);
    seed = engine_inputs.first;
    offset = engine_inputs.second;
#endif

    // launch
    const int64_t logical_block_y_dim = N;
    for (int64_t block_idx_y_base = 0;
         block_idx_y_base < logical_block_y_dim;
         block_idx_y_base += BLOCK_Y_DIM_MAX)
    {
        dim3 blocksPerGrid(
            num_groups,
            std::min(logical_block_y_dim - block_idx_y_base,
                     BLOCK_Y_DIM_MAX),
            1);
        dim3 threadsPerBlock(group_size, 1, 1);

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                        data.scalar_type(),
                                        "pack_mixed_precision", ([&] {
            pack_mixed_precision_kernel<scalar_t>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    data.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    min.data_ptr<scalar_t>(),
                    packed.data_ptr<int32_t>(),
                    bits, seed, offset,
                    N, num_groups, group_size, block_idx_y_base);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
    }

    return std::make_pair(packed, scale);
}

// ======================================================
// Unpack kernel (warp-shuffle inverse of ballot)
// ======================================================
template <typename scalar_t>
__global__ void unpack_mixed_precision_kernel(
    const int32_t* __restrict__ data,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ min,
    scalar_t* __restrict__ unpacked,
    const int bits,
    int64_t N,
    int64_t num_groups,
    int64_t group_size,
    int64_t block_idx_y_base)
{
    const int64_t n = blockIdx.y + block_idx_y_base;
    const int group_id = blockIdx.x;
    const int tid = threadIdx.x;
    if (n >= N || group_id >= num_groups || tid >= group_size)
        return;

    unsigned lane = tid & 31;
    unsigned warp_id = tid >> 5;
    const unsigned FULL_MASK = 0xffffffffu;
    const int warps_per_group = (group_size + 31) / 32;

    float s = static_cast<float>(scale[n * num_groups + group_id]);
    float m = static_cast<float>(min[n * num_groups + group_id]);

    int64_t base_idx = ((int64_t)n * num_groups + group_id) *
                       warps_per_group * bits +
                       warp_id * bits;

    int val = 0;
    #pragma unroll
    for (int b = 0; b < bits; ++b) {
        unsigned word = 0;
        if (lane == 0)
            word = reinterpret_cast<const unsigned*>(data)[base_idx + b];
        word = __shfl_sync(FULL_MASK, word, 0);
        unsigned bit_val = (word >> lane) & 1u;
        val |= (bit_val << b);
    }

    int64_t id = (n * num_groups + group_id) * group_size + tid;
    unpacked[id] = static_cast<scalar_t>(val / s + m);
}

// ======================================================
// Host launcher for unpack
// ======================================================
Tensor unpack_mixed_precision_cuda(
    Tensor data,
    int bits,
    Tensor scale,
    Tensor min,
    int64_t N,
    int64_t num_groups,
    int64_t group_size)
{
    TORCH_CHECK(data.is_cuda(), "data must be CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA tensor");


    auto options =
        torch::TensorOptions().dtype(scale.dtype()).device(data.device());
    Tensor unpacked =
        torch::empty({N, num_groups, group_size}, options);


    const int64_t logical_block_y_dim = N;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int64_t block_idx_y_base = 0;
         block_idx_y_base < logical_block_y_dim;
         block_idx_y_base += BLOCK_Y_DIM_MAX)
    {
        dim3 blocksPerGrid(
            num_groups,
            std::min(logical_block_y_dim - block_idx_y_base,
                     BLOCK_Y_DIM_MAX),
            1);
        dim3 threadsPerBlock(group_size, 1, 1);

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                        scale.scalar_type(),
                                        "unpack_mixed_precision", ([&] {
            unpack_mixed_precision_kernel<scalar_t>
                <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    data.data_ptr<int32_t>(),
                    scale.data_ptr<scalar_t>(),
                    min.data_ptr<scalar_t>(),
                    unpacked.data_ptr<scalar_t>(),
                    bits, N, num_groups, group_size,
                    block_idx_y_base);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
    }

    return unpacked;
}
