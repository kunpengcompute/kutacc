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
#include <vector>
#include "tensor/tensor.h"
#include "linear/mm.h"
#include "activation/sigmoid.h"
#include "softmax/softmax.h"
#include "math/fast_exp.h"
#include "wrapper/wrapper.h"
#include "utils/blas.h"
#include "utils/kutacc_malloc.h"
#include "utils/memory.h"
#include "utils/parallel.h"
#include "utils/collapse.h"

namespace kutacc {
void gating_attention_kernel(Tensor &input, Tensor &q, Tensor &k, Tensor &v, Tensor &gate, Tensor &weighted_avg,
                             int64_t batch, int64_t seq_len, Tensor &bias, Tensor &nonbatched_bias,
                             const Tensor &query_w, const Tensor &key_w, const Tensor &value_w, const Tensor &gating_w,
                             const Tensor &gating_b, const Tensor &output_w, const Tensor &output_b, Tensor &out,
                             int64_t block_size, int64_t head_size, int64_t nheads, int64_t nchannels)
{
    KUTACC_CHECK(batch > 0 && seq_len > 0 && block_size > 0 && head_size > 0 && nheads > 0 && nchannels > 0 &&
                     batch <= INT32_MAX / seq_len && seq_len <= INT32_MAX && nchannels <= INT32_MAX &&
                     head_size <= INT32_MAX && block_size <= INT64_MAX / seq_len,
                 "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    addmm(kutacc::to_bf16(1),
          Tensor((__bf16 *)input.data_ptr(), {batch * seq_len, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)query_w.data_ptr(), {nchannels, nchannels}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)q.data_ptr(), {batch * seq_len, nchannels}, {q.strides()[1], 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = true, .prepack_b = true});
    addmm(kutacc::to_bf16(1),
          Tensor((__bf16 *)input.data_ptr(), {batch * seq_len, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)key_w.data_ptr(), {nchannels, nchannels}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)k.data_ptr(), {batch * seq_len, nchannels}, {k.strides()[1], 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = true, .prepack_b = true});
    addmm(kutacc::to_bf16(1), Tensor((__bf16 *)value_w.data_ptr(), {nchannels, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)input.data_ptr(), {nchannels, batch * seq_len}, {1, nchannels}, 2, kBF16),
          kutacc::to_bf16(0),
          Tensor((__bf16 *)v.data_ptr(), {nchannels, batch * seq_len}, {v.strides()[1], 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = true, .prepack_b = true});
    addmm(kutacc::to_bf16(1),
          Tensor((__bf16 *)input.data_ptr(), {batch * seq_len, nchannels}, {nchannels, 1}, 2, kBF16),
          Tensor((__bf16 *)gating_w.data_ptr(), {nchannels, nchannels}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)gate.data_ptr(), {batch * seq_len, nchannels}, {gate.strides()[1], 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = true, .prepack_b = true, .row_bias = true, .bias = gating_b.data_ptr()});
    int64_t nblocks = (seq_len + block_size - 1) / block_size;
    float scale = (float)(1 / std::sqrt(head_size));
    KUTACC_CHECK(batch <= INT64_MAX / nheads / nblocks, "parallel end overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    parallel_for(0, batch * nheads * nblocks, nblocks, [&](int64_t start, int64_t end) {
        int64_t bi, hi, block_i;
        int64_t kv_buf_id = -1;
        data_index_init(start, bi, batch, hi, nheads, block_i, nblocks);
        auto logits_buf = kutacc::alloc<__bf16>(block_size * seq_len);
        auto softmax_buf = kutacc::alloc<float>(seq_len);
        auto softmax_buf_data = softmax_buf.get();
        auto k_buf = kutacc::alloc<__bf16>(seq_len * head_size);
        auto v_buf = kutacc::alloc<__bf16>(seq_len * head_size);
        KUTACC_CHECK(logits_buf != nullptr && softmax_buf != nullptr && k_buf != nullptr && v_buf != nullptr,
                     "gating_attention inner alloc memory failed!");
        if (kutacc::kutacc_check_err_set == true) {
            return;
        }
        for (int64_t _i = start; _i < end; _i++) {
            int64_t q_start = block_i * block_size;
            int64_t q_end = std::min(q_start + block_size, seq_len);
            auto bias_data = (__bf16 *)bias.data_ptr() + bi * bias.strides()[0];
            auto logits = Tensor(logits_buf.get(), {q_end - q_start, seq_len}, {seq_len, 1}, 2, kBF16);
            if (kv_buf_id != bi * nheads + hi) {
                gemm_pack<__bf16, __bf16>('A', 'T', 'N', (int)seq_len, 0, (int)head_size, (int)nchannels, 0,
                                          (__bf16 *)k.data_ptr() + bi * k.strides()[0] + hi * k.strides()[2],
                                          k_buf.get());
            }
            addmm(kutacc::to_bf16(1),
                  Tensor((__bf16 *)q.data_ptr() + bi * q.strides()[0] + q_start * q.strides()[1] + hi * q.strides()[2],
                         {q_end - q_start, head_size}, {q.strides()[1], 1}, 2, kBF16),
                  Tensor(k_buf.get(), {head_size, seq_len}, {1, head_size}, 2, kBF16), kutacc::to_bf16(0), logits,
                  BlasExtendParams{.num_threads = 1, .prepack_a = false, .prepack_b = true});
            for (int64_t qi = q_start; qi < q_end; qi++) {
                auto nonbatched_bias_data = (nonbatched_bias.sizes()[0] == 0) ? nullptr :
                                                                                (__bf16 *)nonbatched_bias.data_ptr() +
                                                                                    hi * nonbatched_bias.strides()[0] +
                                                                                    qi * nonbatched_bias.strides()[1];
                auto data = (__bf16 *)logits.data_ptr() + (qi - q_start) * seq_len;
                auto vl = svcntw();
                svfloat32_t reduce = svdup_f32(-INFINITY);
                for (int64_t si = 0; si < seq_len; si += vl) {
                    svbool_t pg = svwhilelt_b32(si, seq_len);
                    svfloat32_t values = kutacc::svld1<float, __bf16>(pg, &data[si]);
                    values = svmul_x(pg, values, scale);
                    values = svadd_x(pg, values, kutacc::svld1<float, __bf16>(pg, &bias_data[si]));
                    if (nonbatched_bias_data) {
                        values = svadd_x(pg, values, kutacc::svldnt1<float, __bf16>(pg, &nonbatched_bias_data[si]));
                    }
                    reduce = svmax_m(pg, reduce, values);
                    svst1_f32(pg, &softmax_buf_data[si], values);
                }
                kutacc::softmax_with_max(softmax_buf_data, data, seq_len, svmaxv(svptrue_b32(), reduce));
            }
            if (kv_buf_id != bi * nheads + hi) {
                gemm_pack<__bf16, __bf16>('A', 'T', 'N', (int)head_size, 0, (int)seq_len, (int)(batch * seq_len), 0,
                                          (__bf16 *)v.data_ptr() + hi * v.strides()[0] + bi * v.strides()[2],
                                          v_buf.get());
            }
            addmm(kutacc::to_bf16(1), logits, Tensor(v_buf.get(), {seq_len, head_size}, {1, seq_len}, 2, kBF16),
                  kutacc::to_bf16(0),
                  Tensor((__bf16 *)weighted_avg.data_ptr() + bi * weighted_avg.strides()[0] +
                             q_start * weighted_avg.strides()[1] + hi * weighted_avg.strides()[2],
                         {q_end - q_start, head_size}, {weighted_avg.strides()[1], 1}, 2, kBF16),
                  BlasExtendParams{.num_threads = 1, .prepack_a = false, .prepack_b = true});
            for (int64_t qi = q_start; qi < q_end; qi++) {
                int64_t vl = (int64_t)svcntw();
                auto gate_data = (__bf16 *)gate.data_ptr() + bi * gate.strides()[0] + qi * gate.strides()[1] +
                                 hi * gate.strides()[2];
                auto weighted_avg_data = (__bf16 *)weighted_avg.data_ptr() + bi * weighted_avg.strides()[0] +
                                         qi * weighted_avg.strides()[1] + hi * weighted_avg.strides()[2];
                for (int64_t hsi = 0; hsi < head_size; hsi += vl) {
                    auto pg = svwhilelt_b32(hsi, head_size);
                    auto gate_values = kutacc::svld1<float, __bf16>(pg, &gate_data[hsi]);
                    auto values = kutacc::svld1<float, __bf16>(pg, &weighted_avg_data[hsi]);
                    values = svmul_x(pg, values, kutacc::Sigmoid::call(pg, gate_values));
                    kutacc::svst1<__bf16, float>(pg, &weighted_avg_data[hsi], values);
                }
            }
            kv_buf_id = bi * nheads + hi;
            data_index_step(bi, batch, hi, nheads, block_i, nblocks);
        }
    });
    addmm(kutacc::to_bf16(1),
          Tensor((__bf16 *)weighted_avg.data_ptr(), {batch * seq_len, nchannels}, {weighted_avg.strides()[1], 1}, 2,
                 kBF16),
          Tensor((__bf16 *)output_w.data_ptr(), {nchannels, nchannels}, {1, nchannels}, 2, kBF16), kutacc::to_bf16(0),
          Tensor((__bf16 *)out.data_ptr(), {batch * seq_len, nchannels}, {out.strides()[1], 1}, 2, kBF16),
          BlasExtendParams{.prepack_a = false, .prepack_b = true, .row_bias = true, .bias = output_b.data_ptr()});
}
} // namespace kutacc
kutacc_export void kutacc_af2_gating_attention(kutacc_tensor_h input, kutacc_af2_attention_inputs_t *q_based_ptr,
                                               kutacc_tensor_h bias, kutacc_tensor_h nonbatched_bias,
                                               kutacc_af2_attention_weights_t *weight_ptr, kutacc_tensor_h out,
                                               int64_t block_size)
{
    KUTACC_CHECK(input != nullptr && q_based_ptr != nullptr && bias != nullptr && nonbatched_bias != nullptr &&
                     weight_ptr != nullptr && out != nullptr,
                 "kutacc_af2_gating_attention: input args nullptr error");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    kutacc_tensor_h q = q_based_ptr->q;
    kutacc_tensor_h k = q_based_ptr->k;
    kutacc_tensor_h v = q_based_ptr->v;
    kutacc_tensor_h gate = q_based_ptr->gate;
    kutacc_tensor_h weight_avg = q_based_ptr->avg;
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

    kutacc::gating_attention_kernel(
        *kutacc::convertKutaccTensor(input), *kutacc::convertKutaccTensor(q), *kutacc::convertKutaccTensor(k),
        *kutacc::convertKutaccTensor(v), *kutacc::convertKutaccTensor(gate), *kutacc::convertKutaccTensor(weight_avg),
        batch, seq_len, *kutacc::convertKutaccTensor(bias), *kutacc::convertKutaccTensor(nonbatched_bias),
        *kutacc::convertKutaccTensor(query_w), *kutacc::convertKutaccTensor(key_w),
        *kutacc::convertKutaccTensor(value_w), *kutacc::convertKutaccTensor(gating_w),
        *kutacc::convertKutaccTensor(gating_b), *kutacc::convertKutaccTensor(output_w),
        *kutacc::convertKutaccTensor(output_b), *kutacc::convertKutaccTensor(out), block_size, head_size, nheads,
        nchannels);
}
