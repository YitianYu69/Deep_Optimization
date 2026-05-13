#include <torch/extension.h>
#include "quant_pack_dispatch.h"
#include "dequant_unpack_dispatch.h"
#include "copy.h"

using namespace at;  // makes Tensor visible

// Forward declarations of pack1/pack2/pack4/pack8 from quant_pack_dispatch.cpp
// std::tuple<Tensor, Tensor, Tensor> 
void pack1(Tensor x, Tensor out, Tensor scaler, torch::Tensor min, float beta);
void pack2(Tensor x, Tensor out, Tensor scaler, torch::Tensor min, float beta);
void pack3(Tensor x, Tensor out, Tensor scaler, torch::Tensor min, float beta);
void pack4(Tensor x, Tensor out, Tensor scaler, torch::Tensor min, float beta);
void pack8(Tensor x, Tensor out, Tensor scaler, torch::Tensor min, float beta);

void unpack1(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize);
void unpack2(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize);
void unpack3(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize);
void unpack4(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize);
void unpack8(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize);

void copy_launch(torch::Tensor input, torch::Tensor fake_unpack, int N, int G);

std::pair<Tensor, Tensor> pack_mixed_precision_cuda(
    Tensor data,
    Tensor min,
    Tensor max,
    int bits,
    bool stochastic);

Tensor unpack_mixed_precision_cuda(
    Tensor data,
    int bits,
    Tensor scale,
    Tensor min,
    int64_t N,
    int64_t num_groups,
    int64_t group_size);

Tensor calc_precision(Tensor b, Tensor C, Tensor w, double target);

std::tuple<Tensor, Tensor> min_max_kernel(Tensor data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calc_precision", &calc_precision);
    m.def("pack_mixed_precision_cuda", &pack_mixed_precision_cuda);
    m.def("unpack_mixed_precision_cuda", &unpack_mixed_precision_cuda);
    m.def("min_max_kernel", &min_max_kernel);
    
    m.def("pack1", &pack1);
    m.def("pack2", &pack2);
    m.def("pack3", &pack3);
    m.def("pack4", &pack4);
    m.def("pack8", &pack8);

    m.def("unpack1", &unpack1);
    m.def("unpack2", &unpack2);
    m.def("unpack3", &unpack3);
    m.def("unpack4", &unpack4);
    m.def("unpack8", &unpack8);

    m.def("copy_launch", &copy_launch);
}
