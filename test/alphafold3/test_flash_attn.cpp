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

#define block_n 32
#define block_q 128
#define block_kv 64
#define block_kv_grid 128
#define data_min -1e9
#define data_min_grid -3.3895313892515355e+38


void test_flash_attn(std::vector<int64_t> &cas, int Nq, int Nkv, int H, int D)
{

    std::mt19937 rnd(time(0));

    int N_s = cas[0];
    int B = cas[1];
    int T = 50;

    std::cout << "test flash_attn of " << N_s << " samples, " << B << "bucket" << std::endl;

    int q_size = N_s*B*Nq*H*D;
    int kv_size = N_s*B*Nkv*H*D;
    int bias_size = B*Nq*Nkv;
    int pair_logits_size = B*H*Nq*Nkv;
    int logits_size = N_s*B*H*Nq*Nkv;
    int mask_size = B * Nq;

    float scale = (1/std::sqrt((float)D));

    std::unique_ptr<float[]> q(new float[q_size]);
    std::unique_ptr<float[]> k(new float[kv_size]);
    std::unique_ptr<float[]> v(new float[kv_size]);
    std::unique_ptr<float[]> pair_logits(new float[pair_logits_size]);
    std::unique_ptr<float[]> bias(new float[bias_size]);
    std::unique_ptr<float[]> mask(new float[mask_size]);

    std::unique_ptr<float[]> logits(new float[logits_size]);
    std::unique_ptr<float[]> weights(new float[logits_size]);
    std::unique_ptr<float[]> output_norm(new float[q_size]);

    std::unique_ptr<float[]> q_transpose(new float[q_size]);
    std::unique_ptr<float[]> k_transpose(new float[kv_size]);
    std::unique_ptr<float[]> v_transpose(new float[kv_size]);
    std::unique_ptr<float[]> bias_r(new float[bias_size]);
    std::unique_ptr<float[]> pair_logits_r(new float[pair_logits_size]);
    std::unique_ptr<float[]> output_kutacc_fp32(new float[q_size]);
    std::unique_ptr<float[]> output_kutacc(new float[q_size]);

    std::unique_ptr<bfloat16_t[]> q_bf16(new bfloat16_t[q_size]);
    std::unique_ptr<bfloat16_t[]> k_bf16(new bfloat16_t[kv_size]);
    std::unique_ptr<bfloat16_t[]> v_bf16(new bfloat16_t[kv_size]);
    std::unique_ptr<bfloat16_t[]> bias_bf16(new bfloat16_t[bias_size]);
    std::unique_ptr<bfloat16_t[]> pair_logits_bf16(new bfloat16_t[pair_logits_size]);
    std::unique_ptr<bfloat16_t[]> output_kutacc_bf16(new bfloat16_t[q_size]);

    for(int i = 0; i < q_size; ++i){
        q[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < kv_size; ++i){
        k[i] = rnd() * 1.0 / rnd.max();
        v[i] = rnd() * 1.0 / rnd.max();
    }
    
    for(int i = 0; i < pair_logits_size; ++i){
        pair_logits[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < mask_size; ++i){
        mask[i] = (rnd() * 1.0 / rnd.max()) < 0.87 ? 1.0 : 0.0;
    }

    for(int i = 0; i < bias_size; ++i){
        bias[i] = (rnd() * 1.0 / rnd.max()) < 0.87 ? 0 : -1e9;
        bias_bf16[i] = rnd() * 1.0 / rnd.max();
    }

    compute_logits(q.get(), k.get(), pair_logits.get(), bias.get(), logits.get(), N_s, B, H, Nq, Nkv, D, scale);
    softmax_last_dim(logits.get(), weights.get(), N_s*B, H, Nq, Nkv);
    weighted_sum_v(weights.get(), v.get(), output_norm.get(), N_s, B, H, Nq, Nkv, D);

    transpose_BNHD_to_BHND_sve(q.get(), q_transpose.get(), N_s*B, Nq, H, D);
    transpose_BNHD_to_BHND_sve(k.get(), k_transpose.get(), N_s*B, Nkv, H, D);
    transpose_BNHD_to_BHND_sve(v.get(), v_transpose.get(), N_s*B, Nkv, H, D);

    cvt_fp32_bf16(q_transpose.get(), q_bf16.get(), q_size);
    cvt_fp32_bf16(k_transpose.get(), k_bf16.get(), kv_size);
    cvt_fp32_bf16(v_transpose.get(), v_bf16.get(), kv_size);
    cvt_fp32_bf16(pair_logits.get(), pair_logits_bf16.get(), pair_logits_size);

    if(D == 32){
        int nblock_q = (Nq%block_q)?(Nq/block_q+1):(Nq/block_q);
        int nblock_kv = (Nkv%block_kv)?(Nkv/block_kv+1):(Nkv/block_kv);
        kutacc::cross_pack_pair_logits(pair_logits_bf16.get(), pair_logits_r.get(), nblock_q, nblock_kv, B, H, Nq, Nkv);
        kutacc::cross_pack_pair_logits(bias.get(), bias_r.get(), nblock_q, nblock_kv, B, 1, Nq, Nkv);
        kutacc::cross_flash_attn_mask_bias_bf16_hd32(N_s*B, B, H, Nq, Nkv, D, 
            q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), 
            bias_r.get(), pair_logits_r.get(), scale, data_min);
    }else if(D == 16){
        int N = Nq;
        int nblock_q = (N%block_q)?(N/block_q+1):(N/block_q);
        int nblock_kv = (N%block_kv_grid)?(N/block_kv_grid+1):(N/block_kv_grid);
        kutacc::grid_pack_pair_logits(bias_bf16.get(), bias_r.get(), nblock_q, nblock_kv, H, N, N);
        kutacc::grid_flash_attn_mask_bias_bf16_hd16(B, H, N, D, q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), mask.get(), bias_r.get(), scale, data_min_grid);
    }else if(D == 48){
        q_size = Nq*H*D;
        int nblock_q = (Nq%block_q)?(Nq/block_q+1):(Nq/block_q);
        int nblock_kv = (Nkv%block_kv)?(Nkv/block_kv+1):(Nkv/block_kv);
        kutacc::self_pack_pair_logits(pair_logits_bf16.get(), pair_logits_r.get(), nblock_q, nblock_kv, H, Nq, Nkv);
        kutacc::self_flash_attn_mask_bias_bf16_hd48(1, H, Nq, Nkv, D, q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), bias.get(), pair_logits_r.get(), scale, data_min);
    }

    cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), q_size);
    transpose_BNHD_to_BHND_sve(output_kutacc_fp32.get(), output_kutacc.get(), N_s*B, Nq, H, D);

    for(int i = 0; i < q_size; i++){
        float out = output_kutacc[i];
        float expect = output_norm[i];
        if (std::abs(out - expect) > 0.3) {
            std::cerr << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }

    for(int t = 0; t<20; ++t)
    {
        if(D == 32){
            int nblock_q = (Nq%block_q)?(Nq/block_q+1):(Nq/block_q);
            int nblock_kv = (Nkv%block_kv)?(Nkv/block_kv+1):(Nkv/block_kv);
            kutacc::cross_pack_pair_logits(pair_logits_bf16.get(), pair_logits_r.get(), nblock_q, nblock_kv, B, H, Nq, Nkv);
            kutacc::cross_pack_pair_logits(bias.get(), bias_r.get(), nblock_q, nblock_kv, B, 1, Nq, Nkv);
            kutacc::cross_flash_attn_mask_bias_bf16_hd32(N_s*B, B, H, Nq, Nkv, D, 
                q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), 
                bias_r.get(), pair_logits_r.get(), scale, data_min);
        }else if(D == 16){
            int N = Nq;
            int nblock_q = (N%block_q)?(N/block_q+1):(N/block_q);
            int nblock_kv = (N%block_kv_grid)?(N/block_kv_grid+1):(N/block_kv_grid);
            kutacc::grid_pack_pair_logits(bias_bf16.get(), bias_r.get(), nblock_q, nblock_kv, H, N, N);
            kutacc::grid_flash_attn_mask_bias_bf16_hd16(B, H, N, D, q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), mask.get(), bias_r.get(), scale, data_min_grid);
        }else if(D == 48){
            q_size = Nq*H*D;
            int nblock_q = (Nq%block_q)?(Nq/block_q+1):(Nq/block_q);
            int nblock_kv = (Nkv%block_kv)?(Nkv/block_kv+1):(Nkv/block_kv);
            kutacc::self_pack_pair_logits(pair_logits_bf16.get(), pair_logits_r.get(), nblock_q, nblock_kv, H, Nq, Nkv);
            kutacc::self_flash_attn_mask_bias_bf16_hd48(1, H, Nq, Nkv, D, q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), bias.get(), pair_logits_r.get(), scale, data_min);
        }
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<50; ++t)
    {
        if(D == 32){
            int nblock_q = (Nq%block_q)?(Nq/block_q+1):(Nq/block_q);
            int nblock_kv = (Nkv%block_kv)?(Nkv/block_kv+1):(Nkv/block_kv);
            kutacc::cross_pack_pair_logits(pair_logits_bf16.get(), pair_logits_r.get(), nblock_q, nblock_kv, B, H, Nq, Nkv);
            kutacc::cross_pack_pair_logits(bias.get(), bias_r.get(), nblock_q, nblock_kv, B, 1, Nq, Nkv);
            kutacc::cross_flash_attn_mask_bias_bf16_hd32(N_s*B, B, H, Nq, Nkv, D, 
                q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), 
                bias_r.get(), pair_logits_r.get(), scale, data_min);
        }else if(D == 16){
            int N = Nq;
            int nblock_q = (N%block_q)?(N/block_q+1):(N/block_q);
            int nblock_kv = (N%block_kv_grid)?(N/block_kv_grid+1):(N/block_kv_grid);
            kutacc::grid_pack_pair_logits(bias_bf16.get(), bias_r.get(), nblock_q, nblock_kv, H, N, N);
            kutacc::grid_flash_attn_mask_bias_bf16_hd16(B, H, N, D, q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), mask.get(), bias_r.get(), scale, data_min_grid);
        }else if(D == 48){
            q_size = Nq*H*D;
            int nblock_q = (Nq%block_q)?(Nq/block_q+1):(Nq/block_q);
            int nblock_kv = (Nkv%block_kv)?(Nkv/block_kv+1):(Nkv/block_kv);
            kutacc::self_pack_pair_logits(pair_logits_bf16.get(), pair_logits_r.get(), nblock_q, nblock_kv, H, Nq, Nkv);
            kutacc::self_flash_attn_mask_bias_bf16_hd48(1, H, Nq, Nkv, D, q_bf16.get(), k_bf16.get(), v_bf16.get(), output_kutacc_bf16.get(), bias.get(), pair_logits_r.get(), scale, data_min);
        }
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    std::cout << "flash_attn time = " << elapse_o_store.count()/T << std::endl;

}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args_skip_n(cases, 2, argc, argv, 4);
        int Nq = std::stoi(argv[1]);
        int Nkv = std::stoi(argv[2]);
        int H = std::stoi(argv[3]);
        int D = std::stoi(argv[4]);

        for (auto cas : cases) {
            test_flash_attn(cas, Nq, Nkv, H, D);
        }
    });
}
