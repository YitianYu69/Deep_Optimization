#pragma once
#include <torch/extension.h>


template<int bits>
void launch_unpack_sm90(torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor ema_min, int N, int G, int batchSize);