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

void test_glu_permute(std::vector<int64_t> &cas, int num_channels, int num_intermediate)
{

    std::mt19937 rnd(time(0));
    int batch = cas[0];
    int bucket = cas[1];
    int T = 50;

    std::cout << "test glu_permute of " << batch << " batch, " << bucket << " buckets" << std::endl;

    int x_size = batch*bucket*num_channels;
    int w_size = num_channels * 2 * num_intermediate;
    int mask_size = batch*bucket;
    int output_size = batch*bucket*num_intermediate;

    std::unique_ptr<bfloat16_t[]> x(new bfloat16_t[x_size]);
    std::unique_ptr<bfloat16_t[]> w(new bfloat16_t[w_size]);
    std::unique_ptr<bfloat16_t[]> packed_x(new bfloat16_t[x_size]);
    std::unique_ptr<bfloat16_t[]> packed_w(new bfloat16_t[w_size]);
    std::unique_ptr<bfloat16_t[]> output_kutacc_bf16(new bfloat16_t[output_size]);

    std::unique_ptr<float[]> mask(new float[mask_size]);


    for(int i = 0; i < x_size; ++i){
        x[i] = rnd() * 1.0 / rnd.max();
    }
    
    for(int i = 0; i < w_size; ++i){
        w[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < mask_size; ++i){
        mask[i] = rnd() * 1.0 / rnd.max();
    }

    kutacc::glu_pack_weight_bf16(w.get(), packed_w.get(), num_channels, num_intermediate*2);
    kutacc::glu_pack_left_bf16(x.get(), packed_x.get(), batch*bucket, num_channels);
    kutacc::glu_mask_permute(packed_x.get(), packed_w.get(), output_kutacc_bf16.get(), mask.get(), batch, bucket, num_channels, num_intermediate);

    for(int t = 0; t<20; ++t)
    {
        kutacc::glu_pack_left_bf16(x.get(), packed_x.get(), batch*bucket, num_channels);
        kutacc::glu_mask_permute(packed_x.get(), packed_w.get(), output_kutacc_bf16.get(), mask.get(), batch, bucket, num_channels, num_intermediate);
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<T; ++t)
    {
        kutacc::glu_pack_left_bf16(x.get(), packed_x.get(), batch*bucket, num_channels);
        kutacc::glu_mask_permute(packed_x.get(), packed_w.get(), output_kutacc_bf16.get(), mask.get(), batch, bucket, num_channels, num_intermediate);
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    std::cout << "glu_permute time = " << elapse_o_store.count()/T << std::endl;

}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args_skip_n(cases, 2, argc, argv, 2);
        int num_channels = std::stoi(argv[1]);
        int num_intermediate = std::stoi(argv[2]);
        for (auto cas : cases) {
            test_glu_permute(cas, num_channels, num_intermediate);
        }
    });
}
