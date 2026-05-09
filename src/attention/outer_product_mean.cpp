/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuTACC is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <cstring>
#include "linear/mm.h"
#include "wrapper/wrapper.h"
#include "utils/parallel.h"
#include "utils/collapse.h"
#include "utils/check.h"
#include "utils/memory.h"

namespace kutacc {
/*
* [OUT] left_proj, right_proj, left_proj_, right_proj_, mask
* [IN] left_proj_w, left_proj_b, right_proj_w, right_proj_b
* [IN]c_i, c_m, n_res, n_res_gather, n_seq, mask_bias
*/

void outer_product_mean_calc_left_and_right_mul_kernel(Tensor &left_proj, Tensor &right_proj, Tensor &left_proj_,
                                                       Tensor &right_proj_, Tensor &input_act, Tensor &mask,
                                                       Tensor &norm, const Tensor &left_proj_w,
                                                       const Tensor &left_proj_b, const Tensor &right_proj_w,
                                                       const Tensor &right_proj_b, int64_t c_i, int64_t c_m,
                                                       int64_t n_res, int64_t n_res_gather, int64_t n_seq,
                                                       int64_t mask_bias)
{
    KUTACC_CHECK(c_i > 0 && c_m > 0 && n_res > 0 && n_res_gather > 0 && n_seq > 0 && mask_bias >= 0 &&
                     c_i <= INT64_MAX / n_res && c_m <= INT32_MAX && n_res <= INT32_MAX / n_seq &&
                     n_res_gather <= INT32_MAX && n_seq <= INT32_MAX,
                 "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    // 矩阵乘对结果转置
    // input_act [n_seq, n_res, n_cm] -> left_proj [c_i, n_res, n_seq]
    addmm(to_bf16(1), Tensor((__bf16 *)left_proj_w.data_ptr(), {c_i, c_m}, {c_m, 1}, 2, kBF16),
          Tensor((__bf16 *)input_act.data_ptr(), {c_m, n_res * n_seq}, {1, c_m}, 2, kBF16), to_bf16(0),
          Tensor((__bf16 *)left_proj.data_ptr(), {c_i, n_res * n_seq}, {n_res * n_seq, 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = true, .prepack_b = false, .col_bias = true, .bias = left_proj_b.data_ptr()});

    // 矩阵乘对结果转置
    // input_act [n_seq, n_res, n_cm] -> right_proj [c_i, n_res, n_seq]
    addmm(to_bf16(1), Tensor((__bf16 *)right_proj_w.data_ptr(), {c_i, c_m}, {c_m, 1}, 2, kBF16),
          Tensor((__bf16 *)input_act.data_ptr(), {c_m, n_res * n_seq}, {1, c_m}, 2, kBF16), to_bf16(0),
          Tensor((__bf16 *)right_proj.data_ptr(), {c_i, n_res * n_seq}, {n_res * n_seq, 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = true, .prepack_b = false, .col_bias = true, .bias = right_proj_b.data_ptr()});

    // left_proj_ = (left_proj + left_proj_b) * mask
    // right_proj_ = (right_proj + right_proj_b) * mask
    // left_proj / right_proj [c_i, n_res, n_seq] -> left_proj_ / right_proj_ [n_res, c_i, n_seq]
    parallel_for(0, c_i * n_res, 1, [&](int64_t start, int64_t end) {
        int64_t ci, ri;
        data_index_init(start, ci, c_i, ri, n_res);
        for (int64_t _i = start; _i < end; _i++) {
            auto left_proj_data =
                (__bf16 *)left_proj.data_ptr() + ci * left_proj.strides()[0] + ri * left_proj.strides()[1];
            auto right_proj_data =
                (__bf16 *)right_proj.data_ptr() + ci * right_proj.strides()[0] + ri * right_proj.strides()[1];
            auto mask_data = (__bf16 *)mask.data_ptr() + mask_bias + ri * mask.strides()[0];
            auto left_proj_data_ =
                (__bf16 *)left_proj_.data_ptr() + ri * left_proj_.strides()[0] + ci * left_proj_.strides()[1];
            auto right_proj_data_ =
                (__bf16 *)right_proj_.data_ptr() + ri * right_proj_.strides()[0] + ci * right_proj_.strides()[1];
            int64_t vl = (int64_t)svcntw();
            for (int64_t i = 0; i < n_seq; i += vl) {
                svbool_t pg = svwhilelt_b32(i, n_seq);
                auto left_values = svld1<float, __bf16>(pg, &left_proj_data[i]);
                auto right_values = svld1<float, __bf16>(pg, &right_proj_data[i]);
                auto mask_values = svld1<float, __bf16>(pg, &mask_data[i]);
                left_values = svmul_x(pg, left_values, mask_values);
                right_values = svmul_x(pg, right_values, mask_values);
                svst1<__bf16, float>(pg, &left_proj_data_[i], left_values);
                svst1<__bf16, float>(pg, &right_proj_data_[i], right_values);
            }
            data_index_step(ci, c_i, ri, n_res);
        }
    });

    // norm = mask @ mask.transpose(0, 1)
    addmm(to_bf16(1), Tensor((__bf16 *)mask.data_ptr() + mask_bias, {n_res, n_seq}, {n_seq, 1}, 2, kBF16),
          Tensor((__bf16 *)mask.data_ptr(), {n_seq, n_res_gather}, {1, n_seq}, 2, kBF16), to_bf16(0),
          Tensor((__bf16 *)norm.data_ptr(), {n_res, n_res_gather}, {n_res_gather, 1}, 2, kBF16));
}

/*
 * [out] output_b, output_w out
 * [IN] left_proj_, right_proj_, norm, left_block_size, right_block_size,
 * [IN] c_i, c_z, n_res, n_res_gather, n_seq
*/
void outer_product_mean_chunk_kernel(const Tensor &output_b, const Tensor &output_w, Tensor &out, Tensor &left_proj_,
                                     Tensor &right_proj_, Tensor &norm, int64_t left_block_size,
                                     int64_t right_block_size, int64_t c_i, int64_t c_z, int64_t n_res,
                                     int64_t n_res_gather, int64_t n_seq)
{
    KUTACC_CHECK(
        c_i > 0 && c_z > 0 && n_res > 0 && n_res_gather > 0 && n_seq > 0 &&
            left_block_size <= INT64_MAX / right_block_size / c_i / c_i && left_block_size <= INT64_MAX - n_res + 1 &&
            right_block_size <= INT64_MAX / left_block_size / c_z && right_block_size <= INT64_MAX - n_res_gather + 1 &&
            c_i <= INT32_MAX / c_i && c_z <= INT32_MAX && n_seq <= INT32_MAX,
        "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    int64_t left_nblocks = (n_res + left_block_size - 1) / left_block_size;
    int64_t right_nblocks = (n_res_gather + right_block_size - 1) / right_block_size;
    parallel_for(0, left_nblocks * right_nblocks, 1, [&](int64_t start, int64_t end) {
        int64_t left_block_i, right_block_i;
        data_index_init(start, left_block_i, left_nblocks, right_block_i, right_nblocks);
        auto chunk_buf = alloc<__bf16>(left_block_size * c_i * right_block_size * c_i);
        auto chunk_buf_ = alloc<__bf16>(left_block_size * right_block_size * c_i * c_i);
        auto out_buf = alloc<__bf16>(left_block_size * right_block_size * c_z);
        KUTACC_CHECK(chunk_buf != nullptr && chunk_buf_ != nullptr && out_buf != nullptr,
                     "outer_product_mean inner alloc memory failed!");
        if (kutacc::kutacc_check_err_set == true) {
            return;
        }
        for (int64_t _i = start; _i < end; _i++) {
            int64_t left_start = left_block_i * left_block_size;
            int64_t left_end = std::min(left_start + left_block_size, n_res);
            int64_t right_start = right_block_i * right_block_size;
            int64_t right_end = std::min(right_start + right_block_size, n_res_gather);
            addmm(to_bf16(1),
                  Tensor((__bf16 *)left_proj_.data_ptr() + left_start * left_proj_.strides()[0],
                         {(left_end - left_start) * c_i, n_seq}, {n_seq, 1}, 2, kBF16),
                  Tensor((__bf16 *)right_proj_.data_ptr() + right_start * right_proj_.strides()[0],
                         {n_seq, (right_end - right_start) * c_i}, {1, n_seq}, 2, kBF16),
                  to_bf16(0),
                  Tensor(chunk_buf.get(), {(left_end - left_start) * c_i, (right_end - right_start) * c_i},
                         {(right_end - right_start) * c_i, 1}, 2, kBF16),
                  BlasExtendParams{.num_threads = 1});
            for (int64_t left_ri = left_start; left_ri < left_end; left_ri++) {
                for (int64_t right_ri = right_start; right_ri < right_end; right_ri++) {
                    for (int64_t left_ci = 0; left_ci < c_i; left_ci++) {
                        std::memcpy(chunk_buf_.get() + (left_ri - left_start) * (right_end - right_start) * c_i * c_i +
                                        (right_ri - right_start) * c_i * c_i + left_ci * c_i,
                                    chunk_buf.get() + (left_ri - left_start) * (right_end - right_start) * c_i * c_i +
                                        left_ci * (right_end - right_start) * c_i + (right_ri - right_start) * c_i,
                                    size_t(c_i) * sizeof(__bf16));
                    }
                }
            }
            addmm(to_bf16(1),
                  Tensor(chunk_buf_.get(), {(left_end - left_start) * (right_end - right_start), c_i * c_i},
                         {c_i * c_i, 1}, 2, kBF16),
                  Tensor((__bf16 *)output_w.data_ptr(), {c_i * c_i, c_z}, {1, c_i * c_i}, 2, kBF16), to_bf16(0),
                  Tensor(out_buf.get(), {(left_end - left_start) * (right_end - right_start), c_z}, {c_z, 1}, 2, kBF16),
                  BlasExtendParams{.num_threads = 1,
                                   .prepack_a = false,
                                   .prepack_b = true,
                                   .row_bias = true,
                                   .bias = output_b.data_ptr()});
            int64_t vl = (int64_t)svcntw();
            for (int64_t left_ri = left_start; left_ri < left_end; left_ri++) {
                for (int64_t right_ri = right_start; right_ri < right_end; right_ri++) {
                    auto out_buf_data = out_buf.get() + (left_ri - left_start) * (right_end - right_start) * c_z +
                                        (right_ri - right_start) * c_z;
                    auto out_data = (__bf16 *)out.data_ptr() + left_ri * out.strides()[0] + right_ri * out.strides()[1];
                    float norm_value = to_float(((__bf16 *)norm.data_ptr())[left_ri * n_res_gather + right_ri]);
                    norm_value = 1 / (norm_value + 1e-3f);
                    for (int64_t i = 0; i < c_z; i += vl) {
                        svbool_t pg = svwhilelt_b32(i, c_z);
                        auto values = svld1<float, __bf16>(pg, &out_buf_data[i]);
                        values = svmul_x(pg, values, norm_value);
                        svst1<__bf16, float>(pg, &out_data[i], values);
                    }
                }
            }
            data_index_step(left_block_i, left_nblocks, right_block_i, right_nblocks);
        }
    });
}
} // namespace kutacc

kutacc_export void kutacc_af2_outer_product_mean_calc_left_and_right_mul(kutacc_af2_opm_act_inputs_t *opm_acts_ptr,
                                                                         kutacc_af2_opm_mask_inputs_t *opm_masks_ptr,
                                                                         kutacc_af2_opm_weights_t *opm_weights_ptr)
{
    KUTACC_CHECK(opm_acts_ptr != nullptr && opm_masks_ptr != nullptr && opm_weights_ptr != nullptr,
                 "kutacc_af2_outer_product_mean_calc_left_and_right_mul: input args nullptr error");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    kutacc_tensor_h input_act = opm_acts_ptr->input_act;
    kutacc_tensor_h left_proj = opm_acts_ptr->left_proj;
    kutacc_tensor_h right_proj = opm_acts_ptr->right_proj;
    kutacc_tensor_h left_proj_ = opm_acts_ptr->left_proj_;
    kutacc_tensor_h right_proj_ = opm_acts_ptr->right_proj_;
    int64_t n_seq = opm_acts_ptr->n_seq;
    int64_t n_res = opm_acts_ptr->n_res;

    kutacc_tensor_h mask = opm_masks_ptr->mask;
    kutacc_tensor_h norm = opm_masks_ptr->norm;
    int64_t n_res_gather = opm_masks_ptr->n_res_gather;
    int64_t mask_bias = opm_masks_ptr->mask_bias;

    kutacc_tensor_h left_proj_w = opm_weights_ptr->left_proj_w;
    kutacc_tensor_h right_proj_w = opm_weights_ptr->right_proj_w;
    kutacc_tensor_h left_proj_b = opm_weights_ptr->left_proj_b;
    kutacc_tensor_h right_proj_b = opm_weights_ptr->right_proj_b;
    int64_t c_i = opm_weights_ptr->c_i;
    int64_t c_m = opm_weights_ptr->c_m;

    outer_product_mean_calc_left_and_right_mul_kernel(
        *kutacc::convertKutaccTensor(left_proj), *kutacc::convertKutaccTensor(right_proj),
        *kutacc::convertKutaccTensor(left_proj_), *kutacc::convertKutaccTensor(right_proj_),
        *kutacc::convertKutaccTensor(input_act), *kutacc::convertKutaccTensor(mask), *kutacc::convertKutaccTensor(norm),
        *kutacc::convertKutaccTensor(left_proj_w), *kutacc::convertKutaccTensor(left_proj_b),
        *kutacc::convertKutaccTensor(right_proj_w), *kutacc::convertKutaccTensor(right_proj_b), c_i, c_m, n_res,
        n_res_gather, n_seq, mask_bias);
}

kutacc_export void kutacc_af2_outer_product_mean_chunk(kutacc_af2_opm_act_inputs_t *opm_acts_ptr,
                                                       kutacc_af2_opm_mask_inputs_t *opm_masks_ptr,
                                                       kutacc_af2_opm_weights_t *opm_weights_ptr, kutacc_tensor_h out,
                                                       int64_t left_block_size, int64_t right_block_size)
{
    KUTACC_CHECK(opm_acts_ptr != nullptr && opm_masks_ptr != nullptr && opm_weights_ptr != nullptr && out != nullptr,
                 "kutacc_af2_outer_product_mean_chunk: input args nullptr error");
    KUTACC_CHECK(
        left_block_size > 0 && right_block_size > 0,
        "kutacc_af2_outer_product_mean_chunk: input args int values error, values are less than or equal to zero\n");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    kutacc_tensor_h output_w = opm_weights_ptr->outer_w;
    kutacc_tensor_h output_b = opm_weights_ptr->outer_b;
    int64_t c_i = opm_weights_ptr->c_i;
    int64_t c_z = opm_weights_ptr->c_z;

    kutacc_tensor_h left_proj_ = opm_acts_ptr->left_proj_;
    kutacc_tensor_h right_proj_ = opm_acts_ptr->right_proj_;
    int64_t n_seq = opm_acts_ptr->n_seq;
    int64_t n_res = opm_acts_ptr->n_res;

    kutacc_tensor_h norm = opm_masks_ptr->norm;
    int64_t n_res_gather = opm_masks_ptr->n_res_gather;

    kutacc::outer_product_mean_chunk_kernel(
        *kutacc::convertKutaccTensor(output_b), *kutacc::convertKutaccTensor(output_w),
        *kutacc::convertKutaccTensor(out), *kutacc::convertKutaccTensor(left_proj_),
        *kutacc::convertKutaccTensor(right_proj_), *kutacc::convertKutaccTensor(norm), left_block_size,
        right_block_size, c_i, c_z, n_res, n_res_gather, n_seq);
}
