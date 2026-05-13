#include <torch/extension.h>
#include <cuda_runtime.h>

#include <tuple>

#include "dequant_unpack_dispatch.h"


#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), "It must be CUDA Tensor")


template<int bits>
void dequant_unpack_bits(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize) {
    CHECK_INPUT(in);

    launch_unpack_sm90<bits>(in, out, scaler, ema_min, N, G, batchSize);    
}


// =======================================================================================
// Python API
// =======================================================================================
void unpack1(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize) {
    dequant_unpack_bits<1>(in, out, scaler, ema_min, N, G, batchSize);
}

void unpack2(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize) {
    dequant_unpack_bits<2>(in, out, scaler, ema_min, N, G, batchSize);
}

void unpack3(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize) {
    dequant_unpack_bits<3>(in, out, scaler, ema_min,  N, G, batchSize);
}

void unpack4(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize) {
    dequant_unpack_bits<4>(in, out, scaler, ema_min, N, G, batchSize);
}

void unpack8(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize) {
    dequant_unpack_bits<8>(in, out, scaler, ema_min, N, G, batchSize);
}