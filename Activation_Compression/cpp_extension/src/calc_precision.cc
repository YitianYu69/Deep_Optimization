#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>


// Greedy
torch::Tensor calc_precision(torch::Tensor b, torch::Tensor C, torch::Tensor w, double target) {
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU Tensor!");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous!");
    TORCH_CHECK(C.device().is_cpu(), "C must be a CPU Tensor!");
    TORCH_CHECK(C.is_contiguous(), 'C must be contiguous()');
    TORCH_CHECK(w.device().is_cpu(), 'w must be CPU Tensor!');
    TORCH_CHECK(w.is_contiguous(), 'w must be contiguous!');

    std::priority_queue<std::pair<float, int64_t>> q;

    auto *b_data = b.data_ptr<int>();
    auto *C_data = C.data_ptr<int>();
    auto *w_data = w.data_ptr<int>();

    auto get_obj = [&](float C, int b) {
        int coeff_1 = ((1 << b) - 1) * ((1 << b) - 1);
        int coeff_2 = ((1 << (b - 1)) - 1) * ((1 << (b - 1)) - 1);
        return C * (1.0 / coeff_1 - 1.0 / coeff_2);  // It should be negative.
    };

    int64_t N = b.size(0);
    double b_sum = 0;
    for (int64_t i = 0; i < N; i++) {
        auto delta = get_obj(C_data[i], b_data[i] / w_data[i]);
        q.push(std::pair(delta, i));
        b_sum += b_data[i] * w_data[i];
    }



    while (b_sum > target) {
        assert(!q.empty());
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum -= w_data[i];
        if (b_data[i] > 1) {
            auto delta = get_obj(C_data[i], b_data[i] / w_data[i]);
            q.push(std::make_pair(delta, i));
        }
    }
    
    return b;
}

