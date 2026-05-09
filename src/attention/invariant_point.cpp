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

#include "kutacc.h"
#include <cmath>
#include "tensor/tensor.h"
#include "linear/mm.h"
#include "wrapper/wrapper.h"
#include "utils/collapse.h"
#include "utils/parallel.h"
#include "utils/blas.h"
#include "utils/memory.h"
#include "softmax/softmax.h"
#include <arm_sve.h>
#include <arm_neon.h>
#include <arm_fp16.h>
#include "rigid.h"

namespace kutacc {
void kutacc_af2_invariant_point_kernel(Tensor q, Tensor k, Tensor v, Tensor q_pts, Tensor k_pts, Tensor v_pts, Tensor b,
                                       Tensor a, Tensor head_weights, Tensor weights_head_weights, Tensor o,
                                       Tensor o_pt, Tensor o_pt_norm, Tensor o_pair, Tensor z, Tensor rigid_rot_mats,
                                       Tensor rigid_trans, Tensor mask, Tensor linear_b_w, Tensor linear_b_b,
                                       int64_t n_res, int64_t c_z, int64_t c_hidden, int64_t no_heads,
                                       int64_t no_qk_points, int64_t no_v_points)
{
    KUTACC_CHECK(n_res > 0 && c_z > 0 && c_hidden > 0 && no_heads > 0 && no_qk_points > 0 && no_v_points > 0 &&
                     n_res <= INT64_MAX / no_heads,
                 "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    addmm(to_bf16(1), Tensor((__bf16 *)linear_b_w.data_ptr(), {no_heads, c_z}, {linear_b_w.strides()[0], 1}, 2, kBF16),
          Tensor((__bf16 *)z.data_ptr(), {c_z, n_res * n_res}, {1, z.strides()[1]}, 2, kBF16), to_bf16(0),
          Tensor((__bf16 *)b.data_ptr(), {no_heads, n_res * n_res}, {b.strides()[0], 1}, 2, kBF16),
          {.col_bias = true, .bias = linear_b_b.data_ptr()});

    for (int64_t hi = 0; hi < no_heads; hi++) {
        addmm(to_bf16(1),
              Tensor((__bf16 *)q.data_ptr() + hi * q.strides()[1], {n_res, c_hidden}, {q.strides()[0], 1}, 2, kBF16),
              Tensor((__bf16 *)k.data_ptr() + hi * k.strides()[1], {c_hidden, n_res}, {1, k.strides()[0]}, 2, kBF16),
              to_bf16(0),
              Tensor((__bf16 *)a.data_ptr() + hi * a.strides()[0], {n_res, n_res}, {a.strides()[1], 1}, 2, kBF16));
    }

    for (int64_t i = 0; i < no_heads; i++) {
        float x = ((float *)weights_head_weights.data_ptr())[i];
        x = std::log(std::exp(x) + 1);
        x *= std::sqrt(1.0f / (3 * ((float)no_qk_points * 9.0f / 2)));
        ((float *)head_weights.data_ptr())[i] = x;
    }
    auto pg_pt_att = svwhilelt_b32((int64_t)0, no_qk_points * 3);
    float scale_a = std::sqrt(1.0f / (3 * (float)c_hidden));
    float scale_b = std::sqrt(1.0f / 3);
    parallel_for(0, no_heads * n_res, 1, [&](int64_t start, int64_t end) {
        auto pt_att = kutacc::alloc<float>(n_res);
        auto softmax_buf = kutacc::alloc<float>(n_res);
        KUTACC_CHECK(pt_att != nullptr && softmax_buf != nullptr, "invariant_point inner alloc memory failed!");
        if (kutacc::kutacc_check_err_set == true) {
            return;
        }
        kutacc::collapse_for(start, end, no_heads, n_res, [&](int64_t hi, int64_t qi) {
            for (int64_t ki = 0; ki < n_res; ki++) {
                auto q_values = svld1<float, __bf16>(
                    pg_pt_att, (__bf16 *)q_pts.data_ptr() + qi * q_pts.strides()[0] + hi * q_pts.strides()[1]);
                auto k_values = svld1<float, __bf16>(
                    pg_pt_att, (__bf16 *)k_pts.data_ptr() + ki * k_pts.strides()[0] + hi * k_pts.strides()[1]);
                auto values = svsub_x(pg_pt_att, q_values, k_values);
                values = svmul_x(pg_pt_att, values, values);
                float sum = svaddv(pg_pt_att, values);
                sum *= ((float *)head_weights.data_ptr())[hi] * (-0.5f);
                pt_att[(unsigned long)ki] = sum;
            }
            int64_t vl = (int64_t)svcntw();
            svfloat32_t q_mask = svdup_f32(kutacc::to_float(((__bf16 *)mask.data_ptr())[qi]));
            svfloat32_t reduce_max = svdup_f32(-INFINITY);
            for (int64_t ki = 0; ki < n_res; ki += vl) {
                svbool_t pg = svwhilelt_b32(ki, n_res);
                auto values =
                    svld1<float, __bf16>(pg, (__bf16 *)a.data_ptr() + hi * a.strides()[0] + qi * a.strides()[1] + ki);
                {
                    auto b_values = svld1<float, __bf16>(
                        pg, (__bf16 *)b.data_ptr() + hi * b.strides()[0] + qi * b.strides()[1] + ki);
                    b_values = svmul_x(pg, b_values, scale_b);
                    values = svmla_x(pg, b_values, values, scale_a);
                }
                {
                    auto pt_att_values = svld1<float, float>(pg, pt_att.get() + ki);
                    values = svadd_x(pg, values, pt_att_values);
                }
                {
                    auto mask_values = svld1<float, __bf16>(pg, (__bf16 *)mask.data_ptr() + ki);
                    mask_values = svmad_x(pg, mask_values, q_mask, -1.f);
                    values = svmla_x(pg, values, mask_values, 1e5f);
                }
                reduce_max = svmax_m(pg, reduce_max, values);
                svst1<float, float>(pg, softmax_buf.get() + ki, values);
            }
            kutacc::softmax_with_max(softmax_buf.get(),
                                     (__bf16 *)a.data_ptr() + hi * a.strides()[0] + qi * a.strides()[1], n_res,
                                     svmaxv(svptrue_b32(), reduce_max));
        });
    });
    for (int64_t hi = 0; hi < no_heads; hi++) {
        addmm(to_bf16(1),
              Tensor((__bf16 *)a.data_ptr() + hi * a.strides()[0], {n_res, n_res}, {a.strides()[1], 1}, 2, kBF16),
              Tensor((__bf16 *)v.data_ptr() + hi * v.strides()[1], {n_res, c_hidden}, {v.strides()[0], 1}, 2, kBF16),
              to_bf16(0),
              Tensor((__bf16 *)o.data_ptr() + hi * o.strides()[1], {n_res, c_hidden}, {o.strides()[0], 1}, 2, kBF16));
    }
    parallel_for(0, n_res, 1, [&](int64_t start, int64_t end) {
        for (int64_t qi = start; qi < end; qi++) {
            for (int64_t hi = 0; hi < no_heads; hi++) {
                for (int64_t vpi = 0; vpi < no_v_points; vpi++) {
                    float o_pt_buf[3];
                    for (int64_t di = 0; di < 3; di++) {
                        auto a_data = (__bf16 *)a.data_ptr() + hi * a.strides()[0] + qi * a.strides()[1];
                        auto v_pts_data = (__bf16 *)v_pts.data_ptr() + hi * v_pts.strides()[0] +
                                          vpi * v_pts.strides()[1] + di * v_pts.strides()[2];
                        int64_t vl = (int64_t)svcntw();
                        svfloat32_t reduce_sum = svdup_f32(0);
                        for (int64_t ki = 0; ki < n_res; ki += vl) {
                            svbool_t pg = svwhilelt_b32(ki, n_res);
                            auto values = svld1<float, __bf16>(pg, a_data + ki);
                            auto v_pts_values = svld1<float, __bf16>(pg, v_pts_data + ki);
                            values = svmul_x(pg, values, v_pts_values);
                            reduce_sum = svadd_m(pg, reduce_sum, values);
                        }
                        float sum = svaddv(svptrue_b32(), reduce_sum);
                        o_pt_buf[di] = sum;
                    }
                    auto o_pt_data = (__bf16 *)o_pt.data_ptr() + qi * o_pt.strides()[0] + hi * o_pt.strides()[2] +
                                     vpi * o_pt.strides()[3];
                    auto o_pt_norm_data = (__bf16 *)o_pt_norm.data_ptr() + qi * o_pt_norm.strides()[0] +
                                          hi * o_pt_norm.strides()[1] + vpi;
                    rigid_rot_vec_mul_kernel(o_pt_buf, (float *)rigid_rot_mats.data_ptr() + qi * 9, o_pt_buf,
                                             (float *)rigid_trans.data_ptr() + qi * 3, true);
                    float sqrsum = 0;
                    for (int64_t i = 0; i < 3; i++) {
                        o_pt_data[i * o_pt.strides()[1]] = to_bf16(o_pt_buf[i]);
                        sqrsum += o_pt_buf[i] * o_pt_buf[i];
                    }
                    *o_pt_norm_data = to_bf16(std::sqrt(sqrsum) + 1e-8f);
                }
            }
        }
    });
    parallel_for(0, n_res, 1, [&](int64_t start, int64_t end) {
        for (int64_t ri = start; ri < end; ri++) {
            addmm(
                to_bf16(1),
                Tensor((__bf16 *)a.data_ptr() + ri * a.strides()[1], {no_heads, n_res}, {a.strides()[0], 1}, 2, kBF16),
                Tensor((__bf16 *)z.data_ptr() + ri * z.strides()[0], {n_res, c_z}, {z.strides()[1], 1}, 2, kBF16),
                to_bf16(0),
                Tensor((__bf16 *)o_pair.data_ptr() + ri * o_pair.strides()[0], {no_heads, c_z},
                       {o_pair.strides()[1], 1}, 2, kBF16),
                BlasExtendParams{.num_threads = 1});
        }
    });
    // out = linear(collect, linear_out_w, linenar_out_b);
}

} // namespace kutacc

kutacc_export void kutacc_af2_invariant_point(kutacc_af2_ipa_s_inputs_t *ipa_s_ptrs,
                                              kutacc_af2_ipa_o_inputs_t *ipa_o_ptrs, kutacc_tensor_h z,
                                              kutacc_tensor_h rigid_rot_mats, kutacc_tensor_h rigid_trans,
                                              kutacc_tensor_h mask, kutacc_af2_ipa_weights_t *ipa_weight_ptrs)
{
    KUTACC_CHECK(ipa_s_ptrs != nullptr && ipa_o_ptrs != nullptr && z != nullptr && rigid_rot_mats != nullptr &&
                     rigid_trans != nullptr && mask != nullptr && ipa_weight_ptrs != nullptr,
                 "kutacc_af2_invariant_point: input args nullptr error\n");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    kutacc_tensor_h q = ipa_s_ptrs->q;
    kutacc_tensor_h k = ipa_s_ptrs->k;
    kutacc_tensor_h v = ipa_s_ptrs->v;
    kutacc_tensor_h q_pts = ipa_s_ptrs->q_pts;
    kutacc_tensor_h k_pts = ipa_s_ptrs->k_pts;
    kutacc_tensor_h v_pts = ipa_s_ptrs->v_pts;
    kutacc_tensor_h a = ipa_s_ptrs->a;
    kutacc_tensor_h b = ipa_s_ptrs->b;
    int64_t n_res = ipa_s_ptrs->n_res;

    kutacc_tensor_h o = ipa_o_ptrs->o;
    kutacc_tensor_h o_pt = ipa_o_ptrs->o_pt;
    kutacc_tensor_h o_pt_norm = ipa_o_ptrs->o_pt_norm;
    kutacc_tensor_h o_pair = ipa_o_ptrs->o_pair;

    kutacc_tensor_h head_weights = ipa_weight_ptrs->head_weights;
    kutacc_tensor_h weights_head_weights = ipa_weight_ptrs->weights_head_weights;
    kutacc_tensor_h linear_b_w = ipa_weight_ptrs->linear_b_w;
    kutacc_tensor_h linear_b_b = ipa_weight_ptrs->linear_b_b;
    int64_t c_z = ipa_weight_ptrs->c_z;
    int64_t c_hidden = ipa_weight_ptrs->c_hidden;
    int64_t no_heads = ipa_weight_ptrs->no_heads;
    int64_t no_qk_points = ipa_weight_ptrs->no_qk_points;
    int64_t no_v_points = ipa_weight_ptrs->no_v_points;

    kutacc::kutacc_af2_invariant_point_kernel(
        *kutacc::convertKutaccTensor(q), *kutacc::convertKutaccTensor(k), *kutacc::convertKutaccTensor(v),
        *kutacc::convertKutaccTensor(q_pts), *kutacc::convertKutaccTensor(k_pts), *kutacc::convertKutaccTensor(v_pts),
        *kutacc::convertKutaccTensor(b), *kutacc::convertKutaccTensor(a), *kutacc::convertKutaccTensor(head_weights),
        *kutacc::convertKutaccTensor(weights_head_weights), *kutacc::convertKutaccTensor(o),
        *kutacc::convertKutaccTensor(o_pt), *kutacc::convertKutaccTensor(o_pt_norm),
        *kutacc::convertKutaccTensor(o_pair), *kutacc::convertKutaccTensor(z),
        *kutacc::convertKutaccTensor(rigid_rot_mats), *kutacc::convertKutaccTensor(rigid_trans),
        *kutacc::convertKutaccTensor(mask), *kutacc::convertKutaccTensor(linear_b_w),
        *kutacc::convertKutaccTensor(linear_b_b), n_res, c_z, c_hidden, no_heads, no_qk_points, no_v_points);
}
