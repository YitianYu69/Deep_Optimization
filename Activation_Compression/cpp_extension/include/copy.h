#pragma once
#include <torch/extension.h>


void copy_launch(torch::Tensor input, torch::Tensor fake_unpack, int N, int G);