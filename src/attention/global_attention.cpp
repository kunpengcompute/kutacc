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
#include <cstdint>
#include <cmath>
#include <arm_neon.h>
#include <arm_sve.h>
#include <arm_fp16.h>
#include "wrapper/wrapper.h"
#include "linear/mm.h"
#include "utils/collapse.h"
#include "utils/parallel.h"
#include "utils/memory.h"
#include "softmax/softmax.h"
#include "tensor/tensor.h"
#include "activation/sigmoid.h"
#include "utils/check.h"

namespace kutacc {
void global_attention_kernel(Tensor &q_avg, Tensor &q, Tensor &k, Tensor &v, int64_t batch, int64_t seq_len,
                             int64_t nchannels, int64_t nheads, int64_t head_size, Tensor &gate, Tensor &q_data,
                             Tensor &q_mask, const Tensor &query_w, const Tensor &key_w, const Tensor &value_w,
                             const Tensor &gating_w, const Tensor &gating_b, const Tensor &output_w,
                             const Tensor &output_b, Tensor &out)
{
    KUTACC_CHECK(batch > 0 && seq_len > 0 && nchannels > 0 && nheads > 0 && head_size > 0 &&
                     batch <= INT32_MAX / seq_len && seq_len <= INT32_MAX && nchannels <= INT32_MAX &&
                     head_size <= INT32_MAX && nheads <= INT64_MAX / seq_len && nheads <= INT64_MAX / head_size,
                 "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    Tensor input = q_data;
    parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        for (int64_t bi = start; bi < end; bi++) {
            int64_t vl = (int64_t)svcntw();
            svbool_t pg = svptrue_b32();
            svfloat32_t sum0 = svdup_f32(0);
            svfloat32_t sum1 = svdup_f32(0);
            svfloat32_t sum2 = svdup_f32(0);
            svfloat32_t sum3 = svdup_f32(0);
            float mask_sum = 0;
            __bf16 *data = (__bf16 *)q_data.data_ptr() + bi * seq_len * nchannels;
            for (int64_t si = 0; si < seq_len; si++) {
                float mask_value = kutacc::to_float(((__bf16 *)q_mask.data_ptr())[bi * seq_len + si]);
                mask_sum += mask_value;
                svfloat32_t mask_values = svdup_f32(mask_value);
                sum0 = svmla_x(pg, sum0, kutacc::svld1<float, __bf16>(pg, data), mask_values);
                sum1 = svmla_x(pg, sum1, kutacc::svld1<float, __bf16>(pg, data + vl), mask_values);
                sum2 = svmla_x(pg, sum2, kutacc::svld1<float, __bf16>(pg, data + vl * 2), mask_values);
                sum3 = svmla_x(pg, sum3, kutacc::svld1<float, __bf16>(pg, data + vl * 3), mask_values);
                data += nchannels;
            }
            svfloat32_t scale = svdup_f32(1 / (mask_sum + 1e-10f));
            sum0 = svmul_x(pg, sum0, scale);
            sum1 = svmul_x(pg, sum1, scale);
            sum2 = svmul_x(pg, sum2, scale);
            sum3 = svmul_x(pg, sum3, scale);
            __bf16 *q_avg_data = (__bf16 *)q_avg.data_ptr() + bi * nchannels;
            kutacc::svst1<__bf16, float>(pg, q_avg_data, sum0);
            kutacc::svst1<__bf16, float>(pg, q_avg_data + vl, sum1);
            kutacc::svst1<__bf16, float>(pg, q_avg_data + vl * 2, sum2);
            kutacc::svst1<__bf16, float>(pg, q_avg_data + vl * 3, sum3);
        }
    });
    addmm(kutacc::to_bf16(1), Tensor((__bf16 *)q_avg.data_ptr(), {batch, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)query_w.data_ptr(), {nchannels, nchannels}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)q.data_ptr(), {batch, nchannels}, {nchannels, 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = false, .prepack_b = true});
    addmm(kutacc::to_bf16(1),
          Tensor((__bf16 *)input.data_ptr(), {batch * seq_len, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)key_w.data_ptr(), {nchannels, head_size}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)k.data_ptr(), {batch * seq_len, head_size}, {head_size, 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = false, .prepack_b = true});
    addmm(kutacc::to_bf16(1), Tensor((__bf16 *)value_w.data_ptr(), {head_size, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)input.data_ptr(), {nchannels, batch * seq_len}, {1, nchannels}, 2, kBF16),
          kutacc::to_bf16(0),
          Tensor((__bf16 *)v.data_ptr(), {head_size, batch * seq_len}, {batch * seq_len, 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = true, .prepack_b = false});
    addmm(kutacc::to_bf16(1),
          Tensor((__bf16 *)input.data_ptr(), {batch * seq_len, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)gating_w.data_ptr(), {nchannels, nchannels}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)gate.data_ptr(), {batch * seq_len, nchannels}, {gate.strides()[1], 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = false, .prepack_b = true, .row_bias = true, .bias = gating_b.data_ptr()});
    float scale = (float)(1 / std::sqrt(head_size));
    parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        auto logits_buf = kutacc::alloc<__bf16>(nheads * seq_len);
        auto logits = Tensor(logits_buf.get(), {nheads, seq_len}, {seq_len, 1}, 2, kBF16);
        auto softmax_buf = kutacc::alloc<float>(seq_len);
        auto softmax_buf_data = softmax_buf.get();
        auto weighted_avg = kutacc::alloc<__bf16>(nheads * head_size);
        KUTACC_CHECK(logits_buf != nullptr && softmax_buf != nullptr && weighted_avg != nullptr,
                     "global_attention inner alloc memory failed!");
        if (kutacc::kutacc_check_err_set == true) {
            return;
        }
        for (int64_t bi = start; bi < end; bi++) {
            auto mask_data = (__bf16 *)q_mask.data_ptr() + bi * q_mask.strides()[0];
            addmm(kutacc::to_bf16(1),
                  Tensor((__bf16 *)q.data_ptr() + bi * q.strides()[0], {nheads, head_size}, {head_size, 1}, 2, kBF16),
                  Tensor((__bf16 *)k.data_ptr() + bi * k.strides()[0], {head_size, seq_len}, {1, head_size}, 2, kBF16),
                  kutacc::to_bf16(0), logits, BlasExtendParams{.num_threads = 1});
            for (int64_t hi = 0; hi < nheads; hi++) {
                auto data = (__bf16 *)logits.data_ptr() + hi * seq_len;
                auto vl = svcntw();
                svfloat32_t reduce = svdup_f32(-INFINITY);
                for (int64_t si = 0; si < seq_len; si += vl) {
                    svbool_t pg = svwhilelt_b32(si, seq_len);
                    svfloat32_t values = kutacc::svld1<float, __bf16>(pg, &data[si]);
                    auto bias_values =
                        svmla_x(pg, svdup_f32(-1e9f), kutacc::svld1<float, __bf16>(pg, &mask_data[si]), 1e9f);
                    values = svmul_x(pg, values, scale);
                    values = svadd_x(pg, values, bias_values);
                    reduce = svmax_m(pg, reduce, values);
                    svst1_f32(pg, &softmax_buf_data[si], values);
                }
                softmax_with_max(softmax_buf_data, data, seq_len, svmaxv(svptrue_b32(), reduce));
            }
            addmm(kutacc::to_bf16(1), logits,
                  Tensor((__bf16 *)v.data_ptr() + bi * v.strides()[1], {seq_len, head_size}, {1, v.strides()[0]}, 2,
                         kBF16),
                  kutacc::to_bf16(0), Tensor(weighted_avg.get(), {nheads, head_size}, {head_size, 1}, 2, kBF16),
                  BlasExtendParams{.num_threads = 1});
            if (nchannels == 64) {
                int64_t vl = (int64_t)svcntw();
                svbool_t pg = svptrue_b32();
                svfloat32_t w0 = kutacc::svld1<float, __bf16>(pg, weighted_avg.get());
                svfloat32_t w1 = kutacc::svld1<float, __bf16>(pg, weighted_avg.get() + vl);
                svfloat32_t w2 = kutacc::svld1<float, __bf16>(pg, weighted_avg.get() + vl * 2);
                svfloat32_t w3 = kutacc::svld1<float, __bf16>(pg, weighted_avg.get() + vl * 3);
                for (int64_t qi = 0; qi < seq_len; qi++) {
                    auto gate_data = (__bf16 *)gate.data_ptr() + bi * gate.strides()[0] + qi * gate.strides()[1];
                    svfloat32_t g0 = kutacc::svld1<float, __bf16>(pg, gate_data);
                    svfloat32_t g1 = kutacc::svld1<float, __bf16>(pg, gate_data + vl);
                    svfloat32_t g2 = kutacc::svld1<float, __bf16>(pg, gate_data + vl * 2);
                    svfloat32_t g3 = kutacc::svld1<float, __bf16>(pg, gate_data + vl * 3);
                    g0 = svmul_x(pg, kutacc::Sigmoid::call(pg, g0), w0);
                    g1 = svmul_x(pg, kutacc::Sigmoid::call(pg, g1), w1);
                    g2 = svmul_x(pg, kutacc::Sigmoid::call(pg, g2), w2);
                    g3 = svmul_x(pg, kutacc::Sigmoid::call(pg, g3), w3);
                    kutacc::svst1<__bf16, float>(pg, gate_data, g0);
                    kutacc::svst1<__bf16, float>(pg, gate_data + vl, g1);
                    kutacc::svst1<__bf16, float>(pg, gate_data + vl * 2, g2);
                    kutacc::svst1<__bf16, float>(pg, gate_data + vl * 3, g3);
                }
            }
        }
    });
    addmm(kutacc::to_bf16(1),
          Tensor((__bf16 *)gate.data_ptr(), {batch * seq_len, nchannels}, {gate.strides()[1], 1}, 2, kBF16),
          Tensor((__bf16 *)output_w.data_ptr(), {nchannels, nchannels}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)out.data_ptr(), {batch * seq_len, nchannels}, {out.strides()[1], 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = false, .prepack_b = true, .row_bias = true, .bias = output_b.data_ptr()});
}
} // namespace kutacc

kutacc_export void kutacc_af2_global_attention(kutacc_af2_attention_inputs_t *q_based_ptr, kutacc_tensor_h q_data,
                                               kutacc_tensor_h q_mask, kutacc_af2_attention_weights_t *weight_ptr,
                                               kutacc_tensor_h out)
{
    KUTACC_CHECK(q_based_ptr != nullptr && q_data != nullptr && q_mask != nullptr && weight_ptr != nullptr &&
                     weight_ptr != nullptr && out != nullptr && weight_ptr->nchannels == 64,
                 "kutacc_af2_gating_attention: input args nullptr error or invalid int value error");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    kutacc_tensor_h q = q_based_ptr->q;
    kutacc_tensor_h k = q_based_ptr->k;
    kutacc_tensor_h v = q_based_ptr->v;
    kutacc_tensor_h gate = q_based_ptr->gate;
    kutacc_tensor_h q_avg = q_based_ptr->avg;
    int64_t batch = q_based_ptr->batch;
    int64_t seq_len = q_based_ptr->seq_len;

    kutacc_tensor_h query_w = weight_ptr->query_w;
    kutacc_tensor_h key_w = weight_ptr->key_w;
    kutacc_tensor_h value_w = weight_ptr->value_w;
    kutacc_tensor_h gating_w = weight_ptr->gating_w;
    kutacc_tensor_h gating_b = weight_ptr->gating_b;
    kutacc_tensor_h output_w = weight_ptr->output_w;
    kutacc_tensor_h output_b = weight_ptr->output_b;
    int64_t head_size = weight_ptr->head_size;
    int64_t nheads = weight_ptr->nheads;
    int64_t nchannels = weight_ptr->nchannels;

    kutacc::global_attention_kernel(*kutacc::convertKutaccTensor(q_avg), *kutacc::convertKutaccTensor(q),
                                    *kutacc::convertKutaccTensor(k), *kutacc::convertKutaccTensor(v), batch, seq_len,
                                    nchannels, nheads, head_size, *kutacc::convertKutaccTensor(gate),
                                    *kutacc::convertKutaccTensor(q_data), *kutacc::convertKutaccTensor(q_mask),
                                    *kutacc::convertKutaccTensor(query_w), *kutacc::convertKutaccTensor(key_w),
                                    *kutacc::convertKutaccTensor(value_w), *kutacc::convertKutaccTensor(gating_w),
                                    *kutacc::convertKutaccTensor(gating_b), *kutacc::convertKutaccTensor(output_w),
                                    *kutacc::convertKutaccTensor(output_b), *kutacc::convertKutaccTensor(out));
}
