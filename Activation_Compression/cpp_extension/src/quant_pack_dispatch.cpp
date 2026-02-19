#include <torch/extension.h>
#include <cuda_runtime.h>

#include <tuple>

#include "quant_pack_dispatch.h"
#include "copy.h"

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DIM(x) TORCH_CHECK(x.dim() == 3, #x " must be [N, groups, 64]")

// =======================================================================================
// Core dispatch helper
// =======================================================================================
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 

template<int Bits>
void quant_pack_bits(torch::Tensor input, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, float beta) {
    CHECK_INPUT(input);
    CHECK_DIM(input);
    TORCH_CHECK(input.size(2) == 256, "group_size must be 256");

    int N = input.size(0);
    int G = input.size(1);

    // // ballot output count
    // constexpr int warps = 8;   // 512 threads → 16 warps
    // constexpr int B = Bits;     // bits per value

    // int out_elems = N * G * (warps * B);

    // auto out = torch::empty({out_elems},
    //     torch::dtype(torch::kInt32).device(input.device()));
    
    // auto scaler = torch::empty({N * G}, torch::dtype(input.dtype()).device(input.device()));
    // auto min = torch::empty({N * G}, torch::dtype(input.dtype()).device(input.device()));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, input.get_device());

    uint32_t seed = 0x12345678u;

    // -------------------------------------------------------------
    // 1-bit dispatch
    // -------------------------------------------------------------
    if constexpr (Bits == 1 || Bits == 2 || Bits == 3 || Bits == 4 || Bits == 8) {
        if (prop.major == 9 && prop.minor == 0) launch_pack_1bit_sm90<Bits>(input, out, scaler, N, G, seed, min, beta);
        else TORCH_CHECK(false, "Unsupported SM for bit quant");
    }

    // -------------------------------------------------------------
    // 2-bit dispatch
    // -------------------------------------------------------------
    // if constexpr (Bits == 2) {
    //     if (prop.major == 8 && prop.minor == 0)      launch_pack_2bit_sm80 (input, out, N, G, seed);
    //     else if (prop.major == 8 && prop.minor == 9) launch_pack_2bit_sm89 (input, out, N, G, seed);
    //     else if (prop.major == 9 && prop.minor == 0) launch_pack_2bit_sm90 (input, out, N, G, seed);
    //     else if (prop.major == 9 && prop.minor == 1) launch_pack_2bit_sm90a(input, out, N, G, seed);
    //     else TORCH_CHECK(false, "Unsupported SM for 2-bit quant");
    // }

    // -------------------------------------------------------------
    // 4-bit dispatch
    // -------------------------------------------------------------
    // if constexpr (Bits == 4) {
    //     if (prop.major == 8 && prop.minor == 0)      launch_pack_4bit_sm80 (input, out, N, G, seed);
    //     else if (prop.major == 8 && prop.minor == 9) launch_pack_4bit_sm89 (input, out, N, G, seed);
    //     else if (prop.major == 9 && prop.minor == 0) launch_pack_4bit_sm90 (input, out, N, G, seed);
    //     else if (prop.major == 9 && prop.minor == 1) launch_pack_4bit_sm90a(input, out, N, G, seed);
    //     else TORCH_CHECK(false, "Unsupported SM for 4-bit quant");
    // }

    // -------------------------------------------------------------
    // 8-bit dispatch
    // // -------------------------------------------------------------
    // if constexpr (Bits == 8) {
    //     if (prop.major == 8 && prop.minor == 0)      launch_pack_8bit_sm80 (input, out, N, G, seed);
    //     else if (prop.major == 8 && prop.minor == 9) launch_pack_8bit_sm89 (input, out, N, G, seed);
    //     else if (prop.major == 9 && prop.minor == 0) launch_pack_8bit_sm90 (input, out, N, G, seed);
    //     else if (prop.major == 9 && prop.minor == 1) launch_pack_8bit_sm90a(input, out, N, G, seed);
    //     else TORCH_CHECK(false, "Unsupported SM for 8-bit quant");
    // }

    // return {out, scaler.view({N, G}), min.view({N, G})};
}

// =======================================================================================
// Python API
// =======================================================================================
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
void pack1(torch::Tensor x, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, float beta) {
    quant_pack_bits<1>(x, out, scaler, min, beta);
}

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
void pack2(torch::Tensor x, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, float beta) {
    quant_pack_bits<2>(x, out, scaler, min, beta);

}

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
void pack3(torch::Tensor x, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, float beta) {
    quant_pack_bits<3>(x, out, scaler, min, beta);

}

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
void pack4(torch::Tensor x, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, float beta) {
    quant_pack_bits<4>(x, out, scaler, min, beta);
}

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
void pack8(torch::Tensor x, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, float beta) {
    quant_pack_bits<8>(x, out, scaler, min, beta);
}

