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

void test_einsum(std::vector<int64_t> &cas)
{

    std::mt19937 rnd(time(0));

    int batch = 128;
    int bucket = cas[0];
    int size = bucket / 8;
    int T = 50;

    std::cout << "test einsum of " << bucket << " buckets" << std::endl;

    int a_size = batch*bucket*bucket;
    int b_size = batch*bucket*size;


    std::unique_ptr<float[]> a_fp32(new float[a_size]);
    std::unique_ptr<float[]> b_fp32(new float[b_size]);
    std::unique_ptr<float[]> output_norm(new float[b_size]);
    std::unique_ptr<float[]> output_kutacc_fp32(new float[b_size]);


    std::unique_ptr<bfloat16_t[]> a_bf16(new bfloat16_t[a_size]);
    std::unique_ptr<bfloat16_t[]> b_bf16(new bfloat16_t[b_size]);
    std::unique_ptr<bfloat16_t[]> output_kutacc_bf16(new bfloat16_t[b_size]);

    for(int i = 0; i < a_size; ++i){
        a_fp32[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < b_size; ++i){
        b_fp32[i] = rnd() * 1.0 / rnd.max();
    }

    cvt_fp32_bf16(a_fp32.get(), a_bf16.get(), a_size);
    cvt_fp32_bf16(b_fp32.get(), b_bf16.get(), b_size);

    // test einsum("...cik,...cjk->cij")
    einsum_cik_cjk_to_cij(a_fp32.get(), b_fp32.get(), output_norm.get(), 1, batch, bucket, size, bucket);
    kutacc::einsum_eq1(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), b_size);
    for(int i = 0; i < b_size; i++){
        float out = output_kutacc_fp32[i];
        float expect = output_norm[i];
        float max_diff = std::max(expect*0.1, 0.1);
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "einsum(...cik,...cjk->cij)" << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }

    // test einsum("...ckj,...cki->cij")
    einsum_ckj_cki_to_cij(a_fp32.get(), b_fp32.get(), output_norm.get(), 1, batch, bucket, size, bucket);
    kutacc::einsum_eq2(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), b_size);
    for(int i = 0; i < b_size; i++){
        float out = output_kutacc_fp32[i];
        float expect = output_norm[i];
        float max_diff = std::max(expect*0.1, 0.1);
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "einsum(...ckj,...cki->cij)" << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }


    for(int t = 0; t<20; ++t)
    {
        kutacc::einsum_eq1(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<T; ++t)
    {
        kutacc::einsum_eq1(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    std::cout << "einsum_eq1 time = " << elapse_o_store.count()/T << std::endl;

    for(int t = 0; t<20; ++t)
    {
        kutacc::einsum_eq2(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    }
    start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<T; ++t)
    {
        kutacc::einsum_eq2(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    }
    end_o_st = std::chrono::steady_clock::now();
    elapse_o_store = end_o_st - start_o_st;
    std::cout << "einsum_eq2 time = " << elapse_o_store.count()/T << std::endl;

}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args(cases, 1, argc, argv);
        for (auto cas : cases) {
            test_einsum(cas);
        }
    });
}
