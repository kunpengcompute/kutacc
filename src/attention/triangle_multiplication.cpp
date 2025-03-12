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

#include "utils/parallel.h"
#include "utils/collapse.h"
#include "utils/check.h"
#include "kutacc.h"
#include "wrapper/wrapper.h"
#include "activation/sigmoid.h"
#include "linear/mm.h"

namespace kutacc {
/*
 * [out] proj_act
 */
void calc_proj_act(Tensor &proj_act, Tensor &gate, Tensor &input_act, Tensor &mask, 
    const Tensor &proj_w, const Tensor &proj_b, const Tensor &gate_w, const Tensor &gate_b, 
    int64_t n_res, int64_t n_res_gather, int64_t c_o, int64_t c_i, bool input_prepack)
{
    KUTACC_CHECK(n_res > 0 && n_res_gather > 0  && c_o > 0 && c_i > 0 &&
        n_res <= INT64_MAX / n_res_gather && n_res <= INT64_MAX / c_i && c_o <= INT32_MAX,
        "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    // 矩阵乘对结果转置
    addmm(to_bf16(1),
        Tensor((__bf16 *)proj_w.data_ptr(), {c_i, c_o}, {c_o, 1}, 2, kBF16),
        Tensor((__bf16 *)input_act.data_ptr(), {c_o, n_res * n_res_gather}, {1, c_o}, 2, kBF16),
        to_bf16(0),
        Tensor((__bf16 *)proj_act.data_ptr(), {c_i, n_res * n_res_gather}, {proj_act.strides()[0], 1}, 
            2, kBF16),
        BlasExtendParams{.prepack_a = true,
        .prepack_b = input_prepack,
        .col_bias = true,
        .bias = proj_b.data_ptr()});
    // 矩阵乘对结果转置
    addmm(to_bf16(1),
        Tensor((__bf16 *)gate_w.data_ptr(), {c_i, c_o}, {c_o, 1}, 2, kBF16),
        Tensor((__bf16 *)input_act.data_ptr(), {c_o, n_res * n_res_gather}, {1, c_o}, 2, kBF16),
        to_bf16(0),
        Tensor((__bf16 *)gate.data_ptr(), {c_i, n_res * n_res_gather}, {gate.strides()[0], 1},
            2, kBF16),
        BlasExtendParams{.prepack_a = true,
        .prepack_b = input_prepack,
        .col_bias = true,
        .bias = gate_b.data_ptr()});
    // proj_act = proj_act * mask * sigmoid(gate)
    parallel_for(0, c_i * n_res, 1, [&](int64_t start, int64_t end) {
        int64_t ci, bi;
        data_index_init(start, ci, c_i, bi, n_res);
        for ([[maybe_unused]] int64_t j = start; j < end; j++) {
            int64_t vl = (int64_t)svcntw();
            auto proj_act_data =
                (__bf16 *)proj_act.data_ptr() + ci * proj_act.strides()[0] + bi * proj_act.strides()[1];
            auto gate_data = 
                (__bf16 *)gate.data_ptr() + ci * gate.strides()[0] + bi * gate.strides()[1];
            auto mask_data = (__bf16 *)mask.data_ptr() + bi * mask.strides()[0];
            for (int64_t i = 0; i < n_res_gather; i += vl) {
                svbool_t pg = svwhilelt_b32(i, n_res_gather);
                auto values = svld1<float, __bf16>(pg, &proj_act_data[i]);
                values = svmul_x(pg, values, svld1<float, __bf16>(pg, &mask_data[i]));
                auto gate_values = svld1<float, __bf16>(pg, &gate_data[i]);
                gate_values = Sigmoid::call(pg, gate_values);
                values = svmul_x(pg, values, gate_values);
                svst1<__bf16, float>(pg, &proj_act_data[i], values);
            }
            data_index_step(ci, c_i, bi, n_res);
        }
    });
}

/*
 * [out] center_act
 */
void equation(Tensor &center_act, Tensor &left_proj_act, Tensor &right_proj_act,
    int64_t n_res_gather, bool is_incoming)
{
    parallel_for(0, left_proj_act.sizes()[0], 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            if (is_incoming == 0) {
                addmm(to_bf16(1),
                    Tensor(
                        (__bf16 *)left_proj_act.data_ptr() + i * left_proj_act.strides()[0],
                        {n_res_gather, n_res_gather}, {left_proj_act.strides()[1], 1}, 2, kBF16),
                    Tensor(
                        (__bf16 *)right_proj_act.data_ptr() + i * right_proj_act.strides()[0],
                        {n_res_gather, n_res_gather}, {1, right_proj_act.strides()[1]}, 2, kBF16),
                    to_bf16(0),
                    Tensor(
                        (__bf16 *)center_act.data_ptr() + i * center_act.strides()[0],
                        {n_res_gather, n_res_gather}, {center_act.strides()[1], 1}, 2, kBF16),
                    BlasExtendParams{.num_threads = 1});
            } else {
                addmm(to_bf16(1),
                        Tensor(
                        (__bf16 *)right_proj_act.data_ptr() + i * right_proj_act.strides()[0],
                        {n_res_gather, n_res_gather}, {1, right_proj_act.strides()[1]}, 2, kBF16),
                    Tensor(
                        (__bf16 *)left_proj_act.data_ptr() + i * left_proj_act.strides()[0],
                        {n_res_gather, n_res_gather}, {left_proj_act.strides()[1], 1}, 2, kBF16),
                    to_bf16(0),
                    Tensor(
                        (__bf16 *)center_act.data_ptr() + i * center_act.strides()[0],
                        {n_res_gather, n_res_gather}, {center_act.strides()[1], 1}, 2, kBF16),
                    BlasExtendParams{.num_threads = 1});
            }
        }
    });
}

/*
 * [out] gate, out
 */
void gate_and_out_linear(Tensor &gate, Tensor &out, Tensor &input_act, Tensor &center_act, 
    const Tensor &gating_w, const Tensor &gating_b, const Tensor &output_proj_w, const Tensor &output_proj_b,
    int64_t n_res, int64_t n_res_gather, int64_t c_o, int64_t c_i, bool input_prepack)
{
    KUTACC_CHECK(n_res > 0 && n_res_gather > 0  && c_o > 0 && c_i > 0 &&
        n_res <= INT64_MAX / n_res_gather && c_o <= INT32_MAX && c_i <= INT32_MAX,
        "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    addmm(to_bf16(1),
        Tensor((__bf16 *)input_act.data_ptr(), {n_res * n_res_gather, c_o}, {c_o, 1}, 2, kBF16),
        Tensor((__bf16 *)gating_w.data_ptr(), {c_o, c_o}, {1, c_o}, 2, kBF16),
        to_bf16(0),
        Tensor((__bf16 *)gate.data_ptr(), {n_res * n_res_gather, c_o},
            {gate.strides()[1], 1}, 2, kBF16),
        BlasExtendParams{
            .prepack_a = input_prepack,
            .prepack_b = true,
            .row_bias = true,
            .bias = gating_b.data_ptr()});

    addmm(to_bf16(1),
        Tensor((__bf16 *)center_act.data_ptr(), {n_res * n_res_gather, c_i},
            {center_act.strides()[1], 1}, 2, kBF16),
        Tensor((__bf16 *)output_proj_w.data_ptr(), {c_i, c_o}, {1, c_i}, 2, kBF16),
        to_bf16(0),
        Tensor((__bf16 *)out.data_ptr(), {n_res * n_res_gather, c_o},
            {out.strides()[1], 1}, 2, kBF16),
        BlasExtendParams{.prepack_b = true, .row_bias = true, .bias = output_proj_b.data_ptr()});
}

/*
 * [out] out
 */
void last(Tensor &out, Tensor &gate, int64_t n_res, int64_t n_res_gather, int64_t c_o)
{
    parallel_for(0, n_res * n_res_gather, 1, [&](int64_t start, int64_t end) {
        int64_t bi, si;
        data_index_init(start, bi, n_res, si, n_res_gather);
        for ([[maybe_unused]] int64_t j = start; j < end; j++) {
            int64_t vl = (int64_t)svcntw();
            auto out_data = (__bf16 *)out.data_ptr() + bi * out.strides()[0] + si * out.strides()[1];
            auto gate_data = (__bf16 *)gate.data_ptr() + bi * gate.strides()[0] + si * gate.strides()[1];
            for (int64_t i = 0; i < c_o; i += vl) {
                svbool_t pg = svwhilelt_b32(i, c_o);
                auto values = svld1<float, __bf16>(pg, &out_data[i]);
                auto gate_values = svld1<float, __bf16>(pg, &gate_data[i]);
                gate_values = Sigmoid::call(pg, gate_values);
                values = svmul_x(pg, values, gate_values);
                svst1<__bf16, float>(pg, &out_data[i], values);
            }
            data_index_step(bi, n_res, si, n_res_gather);
        }
    }); 
}
}

kutacc_export void kutacc_af2_triangle_multiplication_calc_proj(kutacc_af2_tm_act_inputs_t *tm_acts_ptr, kutacc_tensor_h mask, kutacc_af2_tm_proj_weights_t *tm_weights_ptr, bool input_prepack)
{
    KUTACC_CHECK(tm_acts_ptr != nullptr && mask != nullptr && tm_weights_ptr != nullptr,
        "kutacc_af2_triangle_multiplication_calc_proj: input args nullptr error\n");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    kutacc_tensor_h proj_act = tm_acts_ptr->proj_act;
    kutacc_tensor_h input_act = tm_acts_ptr->input_act;
    kutacc_tensor_h gate = tm_acts_ptr->proj_act_gate;
    int64_t n_res = tm_acts_ptr->n_res;
    int64_t n_res_gather = tm_acts_ptr->n_res_gather;

    kutacc_tensor_h proj_w = tm_weights_ptr->proj_w;
    kutacc_tensor_h proj_b = tm_weights_ptr->proj_b;
    kutacc_tensor_h gate_w = tm_weights_ptr->gate_w;
    kutacc_tensor_h gate_b = tm_weights_ptr->gate_b;
    int64_t c_o = tm_weights_ptr->c_o;
    int64_t c_i = tm_weights_ptr->c_i;
    kutacc::calc_proj_act(*kutacc::convertKutaccTensor(proj_act), *kutacc::convertKutaccTensor(gate), *kutacc::convertKutaccTensor(input_act),
        *kutacc::convertKutaccTensor(mask), *kutacc::convertKutaccTensor(proj_w), *kutacc::convertKutaccTensor(proj_b),
        *kutacc::convertKutaccTensor(gate_w), *kutacc::convertKutaccTensor(gate_b), n_res, n_res_gather, c_o, c_i, input_prepack);
}

kutacc_export void kutacc_af2_triangle_multiplication_equation(kutacc_tensor_h center_act, kutacc_tensor_h left_proj_act, kutacc_tensor_h right_proj_act,
    int64_t n_res_gather, bool is_incoming)
{
    KUTACC_CHECK(center_act != nullptr && left_proj_act != nullptr && right_proj_act != nullptr, 
        "kutacc_af2_triangle_multiplication_equation: input args nullptr error\n");
    KUTACC_CHECK(n_res_gather > 0,
        "kutacc_af2_triangle_multiplication_equation: input args int values error, values are less than or equal to zero\n");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    kutacc::equation(*kutacc::convertKutaccTensor(center_act), *kutacc::convertKutaccTensor(left_proj_act), *kutacc::convertKutaccTensor(right_proj_act),
        n_res_gather, is_incoming);
}

kutacc_export void kutacc_af2_triangle_multiplication_gate_and_out_linear(kutacc_tensor_h gate, kutacc_tensor_h out, kutacc_af2_tm_act_inputs_t *tm_acts_ptr, kutacc_tensor_h center_act,
    kutacc_af2_tm_linear_weights_t *tm_weights_ptr, bool input_prepack)
{
    KUTACC_CHECK(gate != nullptr && out != nullptr && tm_acts_ptr != nullptr && center_act != nullptr && tm_weights_ptr != nullptr,
        "kutacc_af2_triangle_multiplication_gate_and_out_linear: input args nullptr error\n");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    kutacc_tensor_h input_act = tm_acts_ptr->input_act;
    int64_t n_res = tm_acts_ptr->n_res;
    int64_t n_res_gather = tm_acts_ptr->n_res_gather;

    kutacc_tensor_h gating_w = tm_weights_ptr->gating_w;
    kutacc_tensor_h gating_b = tm_weights_ptr->gating_b;
    kutacc_tensor_h output_proj_w = tm_weights_ptr->output_proj_w;
    kutacc_tensor_h output_proj_b = tm_weights_ptr->output_proj_b;
    int64_t c_o = tm_weights_ptr->c_o;
    int64_t c_i = tm_weights_ptr->c_i;

    kutacc::gate_and_out_linear(*kutacc::convertKutaccTensor(gate), *kutacc::convertKutaccTensor(out), *kutacc::convertKutaccTensor(input_act),
        *kutacc::convertKutaccTensor(center_act), *kutacc::convertKutaccTensor(gating_w), *kutacc::convertKutaccTensor(gating_b), 
        *kutacc::convertKutaccTensor(output_proj_w), *kutacc::convertKutaccTensor(output_proj_b), n_res, n_res_gather, c_o, c_i, input_prepack);
}

kutacc_export void kutacc_af2_triangle_multiplication_last(kutacc_tensor_h out, kutacc_tensor_h gate, int64_t n_res, int64_t n_res_gather, int64_t c_o)
{
    KUTACC_CHECK(out != nullptr && gate != nullptr, "kutacc_af2_triangle_multiplication_last: input args nullptr error\n");
    KUTACC_CHECK(n_res > 0 && n_res_gather > 0 && c_o > 0 && n_res <= INT64_MAX / n_res_gather,
        "kutacc_af2_triangle_multiplication_last: input args int values error, values are less than or equal to zero, or overflow\n");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    kutacc::last(*kutacc::convertKutaccTensor(out), *kutacc::convertKutaccTensor(gate), n_res, n_res_gather, c_o);
}