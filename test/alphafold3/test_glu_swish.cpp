#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstring>
#include <random>

#include "helper.h"
#include "kutacc.h"
#include "operator_norm.h"

void test_glu_swish(std::vector<int64_t> &cas, int num_channels, int num_intermediate)
{

    std::mt19937 rnd(time(0));
    int batch = cas[0];
    int bucket1 = cas[1];
    int bucket2 = bucket1;
    int T = 10;

    std::cout << "test glu_swish of " << batch << " batch, " << bucket1 << " buckets" << std::endl;

    int x1_size = bucket1*bucket2*num_channels;
    int x2_size = batch*bucket1*num_channels;
    int w_size = num_channels * 2 * num_intermediate;
    int output_size = bucket1*bucket2*num_intermediate;

    std::unique_ptr<bfloat16_t[]> x1(new bfloat16_t[x1_size]);
    std::unique_ptr<float[]> x2(new float[x2_size]);
    std::unique_ptr<float[]> w(new float[w_size]);
    std::unique_ptr<bfloat16_t[]> output1(new bfloat16_t[output_size]);
    std::unique_ptr<bfloat16_t[]> output2(new bfloat16_t[output_size]);

    for(int i = 0; i < x1_size; ++i){
        x1[i] = rnd() * 1.0 / rnd.max();
    }
    for(int i = 0; i < x2_size; ++i){
        x2[i] = rnd() * 1.0 / rnd.max();
    }
    for(int i = 0; i < w_size; ++i){
        w[i] = rnd() * 1.0 / rnd.max();
    }

    for(int t = 0; t < 10; ++t)
    {
        bfloat16_t* w1 = (bfloat16_t*)w.get();
        const int aligned_elements = num_channels*num_intermediate;
        bfloat16_t *w2 = w1+aligned_elements;
        kutacc::glu_swish(x1.get(), output1.get(), bucket1, bucket2, num_channels, num_intermediate, w1, w2, false);
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t < T; ++t)
    {
        bfloat16_t* w1 = (bfloat16_t*)w.get();
        const int aligned_elements = num_channels*num_intermediate;
        bfloat16_t *w2 = w1+aligned_elements;
        kutacc::glu_swish(x1.get(), output1.get(), bucket1, bucket2, num_channels, num_intermediate, w1, w2, false);
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store1 = end_o_st - start_o_st;
    std::cout << "glu_swish_c34_bf16_bf16 time = " << elapse_o_store1.count()/T << std::endl;

    for(int t = 0; t < 10; ++t)
    {
        const int vscalew = svcntw();
        float* w1 = w.get();
        float *w2 = w1+num_channels*num_intermediate+(vscalew<<1);;
        kutacc::glu_swish(x2.get(), output2.get(), batch, bucket1, num_channels, num_intermediate, w1, w2, true);
    }
    start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t < T; ++t)
    {
        const int vscalew = svcntw();
        float* w1 = w.get();
        float *w2 = w1+num_channels*num_intermediate+(vscalew<<1);;
        kutacc::glu_swish(x2.get(), output2.get(), batch, bucket1, num_channels, num_intermediate, w1, w2, true);
    }
    end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store2 = end_o_st - start_o_st;
    std::cout << "glu_swish_cbatch_bf16 time = " << elapse_o_store2.count()/T << std::endl;

}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args_skip_n(cases, 2, argc, argv, 2);
        int num_channels = std::stoi(argv[1]);
        int num_intermediate = std::stoi(argv[2]);
        for (auto cas : cases) {
            test_glu_swish(cas, num_channels, num_intermediate);
        }
    });
}
