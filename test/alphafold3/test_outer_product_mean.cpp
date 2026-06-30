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
    int times = 20;
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
    std::unique_ptr<float[]> norm_buf(new float[B * B]);
    std::unique_ptr<float[]> workspace(new float[ws_total]);


    std::unique_ptr<bfloat16_t[]> act_bf16(new bfloat16_t[B*T*D]);
    std::unique_ptr<bfloat16_t[]> mask_bf16(new bfloat16_t[B * T]);
    std::unique_ptr<bfloat16_t[]> ln_w_bf16(new bfloat16_t[D]);
    std::unique_ptr<bfloat16_t[]> ln_b_bf16(new bfloat16_t[D]);
    std::unique_ptr<bfloat16_t[]> leftW_bf16(new bfloat16_t[D*C]);
    std::unique_ptr<bfloat16_t[]> rightW_bf16(new bfloat16_t[D*C]);
    std::unique_ptr<bfloat16_t[]> outW_bf16(new bfloat16_t[C*C*F]);
    std::unique_ptr<bfloat16_t[]> outB_bf16(new bfloat16_t[F]);
    std::unique_ptr<bfloat16_t[]> output_kutacc_bf16(new bfloat16_t[T*T*F]);

    std::unique_ptr<bfloat16_t[]> left_proj(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> left_proj_(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> right_proj(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> right_proj_(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> right_proj_new(new bfloat16_t[B*T*C]);
    std::unique_ptr<bfloat16_t[]> norm(new bfloat16_t[T*T]);


    for(int i = 0; i < B*T*D; ++i){
        act[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < B*T; ++i){
        mask[i] = rnd() * 1.0 / rnd.max();
    }
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
    cvt_fp32_bf16(outB.get(), outB_bf16.get(), F);
    transpose_BNHD_to_BHND_sve(mask.get(),mask_permute.get(),1,B,T,1);
    transpose_BNHD_to_BHND_sve(leftW.get(),leftW_permute.get(),1,D,C,1);
    transpose_BNHD_to_BHND_sve(rightW.get(),rightW_permute.get(),1,D,C,1);
    transpose_DAB_to_BDA_sve(outW.get(),outW_permute.get(),C,C,F);
    cvt_fp32_bf16(mask_permute.get(), mask_bf16.get(), B*T);
    cvt_fp32_bf16(leftW_permute.get(), leftW_bf16.get(), D*C);
    cvt_fp32_bf16(rightW_permute.get(), rightW_bf16.get(), D*C);
    cvt_fp32_bf16(outW_permute.get(), outW_bf16.get(), C*C*F);

    // transpose_BNHD_to_BHND_sve(act.get(),act_permute.get(),1,B,T,D);
    layer_norm_chunk(act.get(), act_layernorm.get(), ln_w.get(), ln_b.get(), B, T, D);
    cvt_fp32_bf16(act_layernorm.get(), act_bf16.get(), B*T*D);

    outer_product_mean_forward(
        act.get(), mask.get(), ln_w.get(), ln_b.get(), leftW.get(), rightW.get(), outW.get(), outB.get(),
        output.get(), norm_buf.get(), workspace.get(),
        B, T, D, C, F
    );
    int64_t left_block_size, right_block_size;

    kutacc::default_block_size(B, T, left_block_size, right_block_size);
    kutacc::outer_product_mean(act_bf16.get(), leftW_bf16.get(), rightW_bf16.get(), left_proj.get(),\
    right_proj.get(), left_proj_.get(), right_proj_.get(), right_proj_new.get(), outW_bf16.get(),\
    outB_bf16.get(), output_kutacc_bf16.get(), mask_bf16.get(), norm.get(), C, D, F, T, B, T, 0,\
    B, left_block_size, right_block_size);
    cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), T*T*F);

    for(int t = 0; t<10; ++t)
    {
        kutacc::outer_product_mean(act_bf16.get(), leftW_bf16.get(), rightW_bf16.get(), left_proj.get(),\
        right_proj.get(), left_proj_.get(), right_proj_.get(), right_proj_new.get(), outW_bf16.get(),\
        outB_bf16.get(), output_kutacc_bf16.get(), mask_bf16.get(), norm.get(), C, D, F, T, B, T, 0,\
        B, left_block_size, right_block_size);
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<times; ++t)
    {
        kutacc::outer_product_mean(act_bf16.get(), leftW_bf16.get(), rightW_bf16.get(), left_proj.get(),\
        right_proj.get(), left_proj_.get(), right_proj_.get(), right_proj_new.get(), outW_bf16.get(),\
        outB_bf16.get(), output_kutacc_bf16.get(), mask_bf16.get(), norm.get(), C, D, F, T, B, T, 0,\
        B, left_block_size, right_block_size);
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    std::cout << "outer_product_mean time = " << elapse_o_store.count()/times << std::endl;

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
