#pragma once
#include <torch/extension.h>

// ================================
// 1-bit
// ================================
void launch_pack_1bit_sm80 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_1bit_sm89 (torch::Tensor in, torch::Tensor out, torch::Tensor scaler, torch::Tensor min, int N, int G, uint32_t seed);

template<int Bits>
void launch_pack_1bit_sm90 (torch::Tensor in, torch::Tensor out, torch::Tensor scaler, int N, int G, uint32_t seed, torch::Tensor min, float beta);
void launch_pack_1bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);

// ================================
// 2-bit
// ================================
void launch_pack_2bit_sm80 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_2bit_sm89 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_2bit_sm90 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_2bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);

// ================================
// 4-bit
// ================================
void launch_pack_4bit_sm80 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_4bit_sm89 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_4bit_sm90 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_4bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);

// ================================
// 8-bit
// ================================
void launch_pack_8bit_sm80 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_8bit_sm89 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_8bit_sm90 (torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);
void launch_pack_8bit_sm90a(torch::Tensor in, torch::Tensor out, int N, int G, uint32_t seed);

