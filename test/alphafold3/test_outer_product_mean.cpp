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

void test_outer_product_mean(std::vector<int64_t> &cas)
{

    std::mt19937 rnd(time(0));
    int T = cas[0];
    int B = 1024;
    int D = 64;
    int C = 32;
    int F = 128;

    std::cout << "test outer_product_mean of " << T << " buckets" << std::endl;

    int sz_perm = B * C * T;
    int sz_ein1 = T * C * C * T;
    int sz_temp = T * T * F;
    int ws_total = sz_perm + sz_ein1 + sz_temp;

    std::unique_ptr<float[]> act(new float[B*T*D]);
    std::unique_ptr<float[]> act_layernorm(new float[B*T*D]);
    std::unique_ptr<float[]> act_permute(new float[B*T*D]);
    std::unique_ptr<float[]> mask(new float[B * T]);
    std::unique_ptr<float[]> mask_permute(new float[B * T]);
    std::unique_ptr<float[]> ln_w(new float[D]);
    std::unique_ptr<float[]> ln_b(new float[D]);
    std::unique_ptr<float[]> leftW(new float[D*C]);
    std::unique_ptr<float[]> rightW(new float[D*C]);
    std::unique_ptr<float[]> leftW_permute(new float[D*C]);
    std::unique_ptr<float[]> rightW_permute(new float[D*C]);
    std::unique_ptr<float[]> outW(new float[C*C*F]);
    std::unique_ptr<float[]> outW_permute(new float[C*C*F]);
    std::unique_ptr<float[]> outB(new float[F]);
    std::unique_ptr<float[]> output(new float[T*T*F]);
    std::unique_ptr<float[]> output_kutacc_fp32(new float[T*T*F]);
    std::unique_ptr<float[]> norm_kutacc_fp32(new float[T*T]);
    std::unique_ptr<float[]> norm_buf(new float[T * T]);
    std::unique_ptr<float[]> workspace(new float[T <= 64 ? ws_total : 1]);


    std::unique_ptr<bfloat16_t[]> act_bf16(new bfloat16_t[B*T*D]);
    std::unique_ptr<bfloat16_t[]> mask_bf16(new bfloat16_t[B * T]);
    std::unique_ptr<bfloat16_t[]> ln_w_bf16(new bfloat16_t[D]);
    std::unique_ptr<bfloat16_t[]> ln_b_bf16(new bfloat16_t[D]);
    std::unique_ptr<bfloat16_t[]> leftW_bf16(new bfloat16_t[D*C]);
    std::unique_ptr<bfloat16_t[]> rightW_bf16(new bfloat16_t[D*C]);
    std::unique_ptr<bfloat16_t[]> outW_bf16(new bfloat16_t[C*C*F]);
    std::unique_ptr<bfloat16_t[]> output_kutacc_bf16(new bfloat16_t[T*T*F]);

    std::unique_ptr<bfloat16_t[]> left_proj(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> left_proj_(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> right_proj(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> right_proj_(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> right_proj_new(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> norm(new bfloat16_t[T*T]);


    const uint32_t seed_act = rnd();
    const uint32_t seed_mask = rnd();
    kutacc::parallel_for(0, static_cast<int64_t>(B) * T * D, 1,
        [&](int64_t start, int64_t end) {
        std::mt19937 local_rnd(seed_act + kutacc::get_thread_id());
        for (int64_t i = start; i < end; ++i) act[i] = local_rnd() * 1.0 / local_rnd.max();
    });
    kutacc::parallel_for(0, static_cast<int64_t>(B) * T, 1,
        [&](int64_t start, int64_t end) {
        std::mt19937 local_rnd(seed_mask + kutacc::get_thread_id());
        for (int64_t i = start; i < end; ++i) mask[i] = local_rnd() * 1.0 / local_rnd.max();
    });
    for(int i = 0; i < D; ++i){
        ln_w[i] = rnd() * 1.0 / rnd.max();
        ln_b[i] = rnd() * 1.0 / rnd.max();
    }
    for(int i = 0; i < D*C; ++i){
        leftW[i] = rnd() * 1.0 / rnd.max();
        rightW[i] = rnd() * 1.0 / rnd.max();
    }
    for(int i = 0; i < C*C*F; ++i){
        outW[i] = rnd() * 1.0 / rnd.max();
    }
    for(int i = 0; i < F; ++i){
        outB[i] = rnd() * 1.0 / rnd.max();
    }

    cvt_fp32_bf16(ln_w.get(), ln_w_bf16.get(), D);
    cvt_fp32_bf16(ln_b.get(), ln_b_bf16.get(), D);
    transpose_BNHD_to_BHND_sve(mask.get(),mask_permute.get(),1,B,T,1);
    transpose_BNHD_to_BHND_sve(leftW.get(),leftW_permute.get(),1,D,C,1);
    transpose_BNHD_to_BHND_sve(rightW.get(),rightW_permute.get(),1,D,C,1);
    transpose_DAB_to_BDA_sve(outW.get(),outW_permute.get(),C,C,F);
    cvt_fp32_bf16(mask_permute.get(), mask_bf16.get(), B*T);
    cvt_fp32_bf16(leftW_permute.get(), leftW_bf16.get(), D*C);
    cvt_fp32_bf16(rightW_permute.get(), rightW_bf16.get(), D*C);
    cvt_fp32_bf16(outW_permute.get(), outW_bf16.get(), C*C*F);

    layer_norm_chunk(act.get(), act_layernorm.get(), ln_w.get(), ln_b.get(), B, T, D);
    // The extension applies layernorm(act.transpose(0, 1)); the KuTACC kernel
    // therefore consumes [T, B, D], not the original [B, T, D] layout.
    transpose_BNHD_to_BHND_sve(act_layernorm.get(), act_permute.get(), 1, B, T, D);
    cvt_fp32_bf16(act_permute.get(), act_bf16.get(), B*T*D);

    if (T <= 64) {
        outer_product_mean_forward(
            act.get(), mask.get(), ln_w.get(), ln_b.get(), leftW.get(), rightW.get(), outW.get(), outB.get(),
            output.get(), norm_buf.get(), workspace.get(), B, T, D, C, F);
    }
    int64_t left_block_size, right_block_size;

    kutacc::default_block_size(B, T, left_block_size, right_block_size);
    auto start_o_st = std::chrono::steady_clock::now();
    kutacc::outer_product_mean(act_bf16.get(), leftW_bf16.get(), rightW_bf16.get(), left_proj.get(),\
    right_proj.get(), left_proj_.get(), right_proj_.get(), right_proj_new.get(), outW_bf16.get(),\
    outB.get(), output_kutacc_bf16.get(), mask_bf16.get(), norm.get(), C, D, F, T, B, T, 0,\
    B, left_block_size, right_block_size);
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    if (T <= 64) {
        cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), T*T*F);
    }
    cvt_bf16_fp32(norm.get(), norm_kutacc_fp32.get(), T*T);

    std::vector<int64_t> check_indices;
    if (T <= 64) {
        check_indices.resize(static_cast<int64_t>(T) * T * F);
        for (int64_t i = 0; i < static_cast<int64_t>(T) * T * F; ++i) check_indices[i] = i;
    } else {
        constexpr int sample_pairs = 16;
        check_indices.resize(sample_pairs * F);
        std::unique_ptr<float[]> left_act(new float[static_cast<int64_t>(B) * T * C]);
        std::unique_ptr<float[]> right_act(new float[static_cast<int64_t>(B) * T * C]);
        linear_proj(act_layernorm.get(), leftW.get(), left_act.get(), B, T, D, C);
        linear_proj(act_layernorm.get(), rightW.get(), right_act.get(), B, T, D, C);
        kutacc::parallel_for(0, sample_pairs, 1, [&](int64_t start, int64_t end) {
            for (int64_t s = start; s < end; ++s) {
                const int64_t pair = s * static_cast<int64_t>(T) * T / sample_pairs;
                const int t1 = pair / T;
                const int t2 = pair % T;
                std::vector<float> outer(static_cast<size_t>(C) * C, 0.0f);
                float norm_value = 0.0f;
                for (int b = 0; b < B; ++b) {
                    const float m1 = mask[static_cast<int64_t>(b) * T + t1];
                    const float m2 = mask[static_cast<int64_t>(b) * T + t2];
                    norm_value += m1 * m2;
                    for (int c1 = 0; c1 < C; ++c1) {
                        const float lv = left_act[(static_cast<int64_t>(b) * T + t1) * C + c1] * m1;
                        for (int c2 = 0; c2 < C; ++c2) {
                            outer[c1 * C + c2] += lv * right_act[(static_cast<int64_t>(b) * T + t2) * C + c2] * m2;
                        }
                    }
                }
                for (int f = 0; f < F; ++f) {
                    float value = outB[f];
                    for (int cc = 0; cc < C * C; ++cc) value += outer[cc] * outW[static_cast<int64_t>(cc) * F + f];
                    const int64_t index = pair * F + f;
                    output[index] = value / (DIV_EPS + norm_value);
                    norm_buf[pair] = norm_value;
                    check_indices[s * F + f] = index;
                }
            }
        });
        kutacc::parallel_for(0, static_cast<int64_t>(check_indices.size()), 1,
            [&](int64_t start, int64_t end) {
            for (int64_t s = start; s < end; ++s) {
                const int64_t index = check_indices[s];
                uint16_t bf16_value;
                std::memcpy(&bf16_value, output_kutacc_bf16.get() + index, sizeof(bf16_value));
                const uint32_t fp32_bits = static_cast<uint32_t>(bf16_value) << 16;
                std::memcpy(output_kutacc_fp32.get() + index, &fp32_bits, sizeof(fp32_bits));
            }
        });
    }

    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int64_t max_diff_index = 0;
    for (const int64_t i : check_indices) {
        const float abs_diff = std::abs(output_kutacc_fp32[i] - output[i]);
        const float rel_diff = abs_diff / std::max(std::abs(output[i]), 1.0f);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_diff_index = i;
        }
        max_rel_diff = std::max(max_rel_diff, rel_diff);
    }
    std::cout << "outer_product_mean max_abs_diff = " << max_abs_diff
              << ", max_rel_diff = " << max_rel_diff
              << ", expected = " << output[max_diff_index]
              << ", actual = " << output_kutacc_fp32[max_diff_index]
              << ", index = " << max_diff_index
              << ", expected_norm = " << norm_buf[max_diff_index / F]
              << ", actual_norm = " << norm_kutacc_fp32[max_diff_index / F]
              << std::endl;
    if (max_rel_diff > 0.1f) {
        std::cerr << "outer_product_mean correctness check failed" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "outer_product_mean time = " << elapse_o_store.count() << std::endl;

}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args(cases, 1, argc, argv);
        for (auto cas : cases) {
            test_outer_product_mean(cas);
        }
    });
}
