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

    const uint32_t seed_a = rnd();
    const uint32_t seed_b = rnd();
    kutacc::parallel_for(0, a_size, 1, [&](int64_t start, int64_t end) {
        std::mt19937 local_rnd(seed_a + kutacc::get_thread_id());
        for (int64_t i = start; i < end; ++i) a_fp32[i] = local_rnd() * 1.0 / local_rnd.max();
    });
    kutacc::parallel_for(0, b_size, 1, [&](int64_t start, int64_t end) {
        std::mt19937 local_rnd(seed_b + kutacc::get_thread_id());
        for (int64_t i = start; i < end; ++i) b_fp32[i] = local_rnd() * 1.0 / local_rnd.max();
    });

    cvt_fp32_bf16(a_fp32.get(), a_bf16.get(), a_size);
    cvt_fp32_bf16(b_fp32.get(), b_bf16.get(), b_size);

    const int64_t output_size = b_size;
    const int64_t check_count = bucket <= 128 ? output_size : std::min<int64_t>(4096, output_size);
    std::vector<int64_t> check_indices(check_count);
    for (int64_t s = 0; s < check_count; ++s) {
        check_indices[s] = s * output_size / check_count;
    }

    // test einsum("...cik,...cjk->cij")
    if (bucket <= 128) {
        einsum_cik_cjk_to_cij(a_fp32.get(), b_fp32.get(), output_norm.get(), 1, batch, bucket, size, bucket);
    } else {
        kutacc::parallel_for(0, check_count, 1, [&](int64_t start, int64_t end) {
            for (int64_t s = start; s < end; ++s) {
                const int64_t index = check_indices[s];
                const int c = index / (bucket * size);
                const int rem = index % (bucket * size);
                const int i = rem / size;
                const int j = rem % size;
                float sum = 0.0f;
                for (int k = 0; k < bucket; ++k) {
                    sum += a_fp32[(static_cast<int64_t>(c) * bucket + i) * bucket + k] *
                           b_fp32[(static_cast<int64_t>(c) * size + j) * bucket + k];
                }
                output_norm[index] = sum;
            }
        });
    }
    auto start_o_st = std::chrono::steady_clock::now();
    kutacc::einsum_eq1(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), b_size);
    for(int64_t s = 0; s < check_count; ++s){
        const int64_t i = check_indices[s];
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
    if (bucket <= 128) {
        einsum_ckj_cki_to_cij(a_fp32.get(), b_fp32.get(), output_norm.get(), 1, batch, bucket, size, bucket);
    } else {
        kutacc::parallel_for(0, check_count, 1, [&](int64_t start, int64_t end) {
            for (int64_t s = start; s < end; ++s) {
                const int64_t index = check_indices[s];
                const int c = index / (size * bucket);
                const int rem = index % (size * bucket);
                const int i = rem / bucket;
                const int j = rem % bucket;
                float sum = 0.0f;
                for (int k = 0; k < bucket; ++k) {
                    sum += a_fp32[(static_cast<int64_t>(c) * bucket + k) * bucket + j] *
                           b_fp32[(static_cast<int64_t>(c) * bucket + k) * size + i];
                }
                output_norm[index] = sum;
            }
        });
    }
    std::cout << "einsum_eq1 time = " << elapse_o_store.count() << std::endl;

    start_o_st = std::chrono::steady_clock::now();
    kutacc::einsum_eq2(a_bf16.get(), b_bf16.get(), output_kutacc_bf16.get(), batch, size, bucket);
    end_o_st = std::chrono::steady_clock::now();
    elapse_o_store = end_o_st - start_o_st;
    cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), b_size);
    for(int64_t s = 0; s < check_count; ++s){
        const int64_t i = check_indices[s];
        float out = output_kutacc_fp32[i];
        float expect = output_norm[i];
        float max_diff = std::max(expect*0.1, 0.1);
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "einsum(...ckj,...cki->cij)" << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }
    std::cout << "einsum_eq2 time = " << elapse_o_store.count() << std::endl;

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
