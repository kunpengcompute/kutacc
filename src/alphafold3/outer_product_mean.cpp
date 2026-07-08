#include "linear.h"
#include "utils/memory.h"
#include "utils/bf16.h"
#include "utils/collapse.h"
#include "wrapper/wrapper.h"
#include <iostream>
namespace kutacc{

void default_block_size(int64_t n_seq, int64_t n_res_gather, int64_t &left_block_size, int64_t &right_block_size)
{
    if (n_seq == 512) {
        if (n_res_gather < 200) {
            left_block_size = 4;
        } else if (n_res_gather < 400) {
            left_block_size = 8;
        } else {
            left_block_size = 16;
        }
        if (n_res_gather < 200) {
            right_block_size = 16;
        } else if (n_res_gather < 300) {
            right_block_size = 32;
        } else if (n_res_gather < 700) {
            right_block_size = 80;
        } else {
            right_block_size = 112;
        }
    } else {
        left_block_size = std::clamp(n_res_gather / 24, (int64_t)8, (int64_t)80);
        right_block_size = std::clamp(n_res_gather / 8, (int64_t)16, (int64_t)192);
    }
}

void outer_product_mean(bfloat16_t* input_act, bfloat16_t* left_proj_w, bfloat16_t* right_proj_w, bfloat16_t* left_proj, bfloat16_t* right_proj, bfloat16_t* left_proj_, bfloat16_t* right_proj_,\
    bfloat16_t* right_proj_new, bfloat16_t* output_w, bfloat16_t* output_b, bfloat16_t* out, bfloat16_t* mask, bfloat16_t* norm, int64_t c_i, int64_t c_m, int64_t c_z, int64_t n_res, int64_t n_seq,\
    int64_t n_res_gather, int64_t mask_bias, int64_t mask_stride0, int64_t left_block_size, int64_t right_block_size)
{   
    auto [tm, tn] = kutacc::compute_tm_tn(c_i, n_res * n_seq);
    kutacc::MatrixTilingBlock tiling(tm, tn, c_m);
    std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[n_res * n_seq * c_m]);
    kutacc::bf16_gemm_pack(n_res * n_seq, c_m, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)input_act, pack_b.get());
    {
        std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[c_i * c_m]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[c_i * n_res * n_seq]);
        kutacc::bf16_gemm_pack(c_i, c_m, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)left_proj_w, pack_a.get());
        kutacc::bf16_packed_gemm(c_i, n_res * n_seq, c_m, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)left_proj,
            tmpc.get());
    }
    {
        std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[c_i * c_m]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[c_i * n_res * n_seq]);
        kutacc::bf16_gemm_pack(c_i, c_m, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)right_proj_w, pack_a.get());
        kutacc::bf16_packed_gemm(c_i, n_res * n_seq, c_m, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)right_proj,
            tmpc.get());
    }
    {
        kutacc::parallel_for(0, c_i * n_res, 1, [&](int64_t start, int64_t end) {
            int64_t ci, ri;
            kutacc::data_index_init(start, ci, c_i, ri, n_res);
            for (int64_t unused = start; unused < end; unused ++) {
                auto left_proj_data = left_proj + ci * n_res * n_seq + ri * n_seq;
                auto right_proj_data = right_proj + ci * n_res * n_seq + ri * n_seq;
                auto mask_data = mask + mask_bias + ri * mask_stride0;
                auto left_proj_data_ = left_proj_ + ri * c_i * n_seq + ci * n_seq;
                auto right_proj_data_ = right_proj_ + ri * c_i * n_seq + ci * n_seq;
                int64_t vl = svcntw();
                for (int64_t i = 0; i < n_seq; i += vl) {
                    svbool_t pg = svwhilelt_b32(i, n_seq);
                    auto left_values = kutacc::svld1<float, bfloat16_t>(pg, &left_proj_data[i]);
                    auto right_values = kutacc::svld1<float, bfloat16_t>(pg, &right_proj_data[i]);
                    auto mask_values = kutacc::svld1<float, bfloat16_t>(pg, &mask_data[i]);
                    left_values = svmul_x(pg, left_values, mask_values);
                    right_values = svmul_x(pg, right_values, mask_values);
                    kutacc::svst1<bfloat16_t, float>(pg, &left_proj_data_[i], left_values);
                    kutacc::svst1<bfloat16_t, float>(pg, &right_proj_data_[i], right_values);
                }
                kutacc::data_index_step(ci, c_i, ri, n_res);
            }
        });
    }
    {
        auto [tm, tn] = kutacc::compute_tm_tn(n_res, n_res_gather);
        kutacc::MatrixTilingBlock tiling(tm, tn, n_seq);
        std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[n_res * n_seq]);
        std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[n_seq * n_res_gather]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[n_res * n_res_gather]);
        kutacc::bf16_gemm_pack(n_res, n_seq, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)(mask + mask_bias), pack_a.get());
        kutacc::bf16_gemm_pack(n_res_gather, n_seq, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)mask, pack_b.get());
        kutacc::bf16_packed_gemm(n_res, n_res_gather, n_seq, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)norm,
            tmpc.get());
    }
    if (n_res_gather > n_res) {
        right_proj_ = right_proj_new;
    }
    {
        int64_t left_nblocks = (n_res + left_block_size - 1) / left_block_size;
        int64_t right_nblocks = (n_res_gather + right_block_size - 1) / right_block_size;
        // for (int64_t left_block_i = 0; left_block_i < left_nblocks; ++left_block_i) {
        //     for (int64_t right_block_i = 0; right_block_i < right_nblocks; ++right_block_i) {
        kutacc::parallel_for(0, left_nblocks * right_nblocks, 1, [&](int64_t start, int64_t end) {
            int64_t left_block_i, right_block_i;
            kutacc::data_index_init(start, left_block_i, left_nblocks, right_block_i, right_nblocks);
            auto chunk_buf = kutacc::alloc<bfloat16_t>(left_block_size * c_i * right_block_size * c_i);
            auto chunk_buf_ = kutacc::alloc<bfloat16_t>(left_block_size * right_block_size * c_i * c_i);
            auto out_buf = kutacc::alloc<bfloat16_t>(left_block_size * right_block_size * c_z);
            for (int64_t unused = start; unused < end; unused ++) {
                int64_t left_start = left_block_i * left_block_size;
                int64_t left_end = std::min(left_start + left_block_size, n_res);
                int64_t right_start = right_block_i * right_block_size;
                int64_t right_end = std::min(right_start + right_block_size, n_res_gather);

                int64_t m = (left_end - left_start) * c_i;
                int64_t n = (right_end - right_start) * c_i;
                auto [tm, tn] = kutacc::compute_tm_tn(m, n);
                kutacc::MatrixTilingBlock tiling(m, n, n_seq);
                kutacc::MatrixTilingBlock tiling_shard(ROW_BLOCK_SIZE, COL_BLOCK_SIZE, n_seq);
                std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[m * n_seq]);
                std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[n_seq * n]);
                std::unique_ptr<__bf16[]> tmpc(new __bf16[m * n]);
                kutacc::bf16_gemm_pack_singlethread(m, n_seq, std::get<0>(tiling), std::get<2>(tiling),
                                    (__bf16 *)(left_proj_ + left_start * c_i * n_seq), pack_a.get());
                kutacc::bf16_gemm_pack_singlethread(n, n_seq, std::get<1>(tiling), std::get<2>(tiling),
                                    (__bf16 *)(right_proj_ + right_start * c_i * n_seq), pack_b.get());
                auto chunk_buf = std::unique_ptr<bfloat16_t[]>(new bfloat16_t[m * n]);
                kutacc::bf16_packed_gemm_singlethread(m, n, n_seq, tiling_shard,
                                        pack_a.get(), pack_b.get(),
                                        (__bf16 *)chunk_buf.get(), tmpc.get(), nullptr, false);

                for (int64_t left_ri = left_start; left_ri < left_end; left_ri ++) {
                    for (int64_t right_ri = right_start; right_ri < right_end; right_ri ++) {
                        for (int64_t left_ci = 0; left_ci < c_i; left_ci ++) {
                            std::memcpy(chunk_buf_.get() +
                                            (left_ri - left_start) * (right_end - right_start) * c_i * c_i +
                                            (right_ri - right_start) * c_i * c_i + left_ci * c_i,
                                chunk_buf.get() + (left_ri - left_start) * (right_end - right_start) * c_i * c_i +
                                    left_ci * (right_end - right_start) * c_i + (right_ri - right_start) * c_i,
                                c_i * sizeof(bfloat16_t));
                        }
                    }
                }
                int64_t m2 = (left_end - left_start) * (right_end - right_start);
                int64_t n2 = c_z;
                // auto [tm2, tn2] = kutacc::compute_tm_tn(m2, n2);
                int64_t k2 = c_i * c_i;
                kutacc::MatrixTilingBlock tiling2(m2, n2, k2);
                std::unique_ptr<bfloat16_t[]> pack_a2(new bfloat16_t[m2 * k2]);
                std::unique_ptr<bfloat16_t[]> pack_b2(new bfloat16_t[k2 * n2]);
                std::unique_ptr<__bf16[]> tmpc2(new __bf16[m2 * n2]);
                kutacc::bf16_gemm_pack_singlethread(m2, k2, std::get<0>(tiling2), std::get<2>(tiling2),
                                    (__bf16 *)chunk_buf_.get(), pack_a2.get());
                kutacc::bf16_gemm_pack_singlethread(n2, k2, std::get<1>(tiling2), std::get<2>(tiling2),
                                    (__bf16 *)output_w, pack_b2.get());
                // auto out_buf = std::unique_ptr<bfloat16_t[]>(new bfloat16_t[m2 * n2]);
                kutacc::bf16_packed_gemm_singlethread(m2, n2, k2, tiling2,
                                        pack_a2.get(), pack_b2.get(),
                                        (__bf16 *)out_buf.get(), tmpc2.get(),
                                        (float *)output_b, true);

                int64_t vl = svcntw();
                for (int64_t left_ri = left_start; left_ri < left_end; left_ri ++) {
                    for (int64_t right_ri = right_start; right_ri < right_end; right_ri ++) {
                        auto out_buf_data = out_buf.get() + (left_ri - left_start) * (right_end - right_start) * c_z +
                                            (right_ri - right_start) * c_z;
                        auto out_data =
                            out + left_ri * n_res_gather * c_z + right_ri * c_z;
                        float norm_value =
                            kutacc::to_float((norm)[left_ri * n_res_gather + right_ri]);
                        norm_value = 1 / (norm_value + 1e-3f);
                        for (int64_t i = 0; i < c_z; i += vl) {
                            svbool_t pg = svwhilelt_b32(i, c_z);
                            auto values = kutacc::svld1<float, bfloat16_t>(pg, &out_buf_data[i]);
                            values = svmul_x(pg, values, norm_value);
                            kutacc::svst1<bfloat16_t, float>(pg, &out_data[i], values);
                        }
                    }
                }
                kutacc::data_index_step(left_block_i, left_nblocks, right_block_i, right_nblocks);
            }
        });
    }
}
} //namespace kutacc
