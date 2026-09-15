/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <arm_sve.h>

#include "attention/flash_mla/decode/params.h"

namespace kutacc {

namespace flash_mla {

template <typename DecodeParams, typename KernelTraits, bool is_no_split>
void store(const DecodeParams &params, const int64_t req_idx, const int64_t q_token_idx,
           const int64_t q_block_idx, const int64_t split_idx, float *block_o, float *row_max, float *row_sum)
{
    constexpr auto q_block_size = KernelTraits::q_block_size;
    constexpr auto head_dim = KernelTraits::head_dim;
    constexpr auto head_dim_v = KernelTraits::head_dim_v;
    
    const int split_offset = params.meta->num_splits[req_idx];

    using ScalarTypeO = std::conditional_t<is_no_split, bfloat16_t, float>;
    ScalarTypeO *o =
        is_no_split ?
            (ScalarTypeO *)params.o->data_ptr() + req_idx * params.o->stride(0) + q_token_idx * params.o->stride(1) +
                q_block_idx * q_block_size * params.o->stride(2) :
            (ScalarTypeO *)params.oaccum.data_ptr() + (split_offset + split_idx) * params.oaccum.stride(0) +
                q_token_idx * params.oaccum.stride(1) + q_block_idx * q_block_size * params.oaccum.stride(2);

    float *lse =
        is_no_split ?
            params.softmax_lse->data_ptr() + req_idx * params.softmax_lse->stride(0) +
                q_token_idx * params.softmax_lse->stride(1) +
                q_block_idx * q_block_size * params.softmax_lse->stride(2) :
            params.softmax_lseaccum.data_ptr() + (split_offset + split_idx) * params.softmax_lseaccum.stride(0) +
                q_token_idx * params.softmax_lseaccum.stride(1) +
                q_block_idx * q_block_size * params.softmax_lseaccum.stride(2);

    const int64_t o_head_stride = is_no_split ? params.o->stride(2) : params.oaccum.stride(2);
    const int64_t lse_head_stride = is_no_split ? params.softmax_lse->stride(2) : params.softmax_lseaccum.stride(2);

    for (int i = 0; i < q_block_size && i < params.num_heads_q - q_block_idx * q_block_size; ++i) {
        float sum = row_sum[i];
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        float cur_lse =
            (sum == 0.f || sum != sum) ? (is_no_split ? INFINITY : -INFINITY) :
                row_max[i] * params.softmax_scale + logf(sum);
        lse[i * lse_head_stride] = cur_lse;
        svbool_t ptrue = svptrue_b16();
        if constexpr (is_no_split) {
            float scale = inv_sum;
            if constexpr (std::is_same_v<DecodeParams, SparseDecodeParams>) {
                if (params.attn_sink->has_value()) {
                    float attn_sink_value = params.attn_sink->value().index({q_block_idx * q_block_size + i});
                    scale *= 1.0 / (1.0 + std::exp(attn_sink_value - cur_lse));
                }
            }
            for (int j = 0; j < head_dim_v; j += 32) {
                svfloat32_t sve0 = svld1(ptrue, block_o + i * head_dim_v + j);
                svfloat32_t sve1 = svld1(ptrue, block_o + i * head_dim_v + j + 16);
                sve0 = svmul_m(ptrue, sve0, scale);
                sve1 = svmul_m(ptrue, sve1, scale);

                svbfloat16_t sve = svuzp1(svcvt_bf16_x(ptrue, sve0), svcvt_bf16_x(ptrue, sve1));

                svst1(ptrue, o + i * o_head_stride + j, sve);
            }
        } else {
            for (int j = 0; j < head_dim_v; j += 16) {
                svfloat32_t sve = svld1(ptrue, block_o + i * head_dim_v + j);
                sve = svmul_m(ptrue, sve, inv_sum);
                svst1(ptrue, o + i * o_head_stride + j, sve);
            }
        }
    }
}

} // namespace flash_mla

} // namespace kutacc