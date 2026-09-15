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
#include <type_traits>
#include <cmath>
#include <cstring>
#include "kutacc.h"

#include "attention/flash_mla/decode/params.h"

namespace kutacc {

namespace flash_mla {

template <typename DecodeParams, typename KernelTraits, int max_num_splits>
void splitkv_mla_combine_kernel(const DecodeParams &params)
{
    constexpr auto head_dim_v = KernelTraits::head_dim_v;
    if (params.meta->num_splits[params.batch_size] == params.batch_size) {
        return ;
    }
    const auto task_num = params.batch_size * params.seqlen_q * params.num_heads_q;
    kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
        for (int64_t task_idx = start; task_idx < end; ++task_idx) {
            const int64_t batch_idx = task_idx / params.num_heads_q / params.seqlen_q;
            const int64_t token_idx = task_idx / params.num_heads_q % params.seqlen_q;
            const int64_t head_idx = task_idx % params.num_heads_q;

            const auto split_offset = params.meta->num_splits[batch_idx];
            const auto actual_num_splits = params.meta->num_splits[batch_idx + 1] - split_offset;

            FLASH_MLA_CHECK(actual_num_splits <= max_num_splits, "");

            if (actual_num_splits == 1) {
                continue;
            }

            const float *lseaccum = &params.softmax_lseaccum.index({split_offset, token_idx, head_idx});

            const int64_t lseaccum_split_stride = params.softmax_lseaccum.stride(0);

            float *lse = &params.softmax_lse->index({batch_idx, token_idx, head_idx});

            alignas(ALIGNMENT) float local_lse[max_num_splits];

            float max_lse = -INFINITY;
            for (int64_t split = 0; split < actual_num_splits; ++split) {
                local_lse[split] = lseaccum[split * lseaccum_split_stride];
                if (max_lse < local_lse[split]) {
                    max_lse = local_lse[split];
                }
            }

            float sum_lse_exp = 0;
            for (int64_t split = 0; split < actual_num_splits; ++split) {
                sum_lse_exp += expf(local_lse[split] - max_lse);
            }

            float global_lse =
                (sum_lse_exp == 0.f || sum_lse_exp != sum_lse_exp) ? INFINITY : logf(sum_lse_exp) + max_lse;

            *lse = global_lse;

            float attn_sink_scale = 1.0;
            if constexpr (std::is_same_v<DecodeParams, SparseDecodeParams>) {
                if (params.attn_sink->has_value()) {
                    float attn_sink_value = params.attn_sink->value().index({head_idx});
                    attn_sink_scale = 1.0 / (1.0 + std::exp(attn_sink_value - global_lse));
                }
            }

            alignas(ALIGNMENT) float scale[max_num_splits];
            for (int64_t split = 0; split < actual_num_splits; ++split) {
                scale[split] = expf(local_lse[split] - global_lse) * attn_sink_scale;
            }

            const float *oaccum = &params.oaccum.index({split_offset, token_idx, head_idx, 0});

            const int64_t oaccum_split_stride = params.oaccum.stride(0);

            bfloat16_t *o = &params.o->index({batch_idx, token_idx, head_idx, 0});

            svbool_t ptrue = svptrue_b16();
            for (int i = 0; i < head_dim_v; i += 32) {
                svfloat32_t sve00 = svdup_f32(0);
                svfloat32_t sve01 = svdup_f32(0);
                for (int64_t split = 0; split < actual_num_splits; ++split) {
                    sve00 = svadd_m(ptrue, sve00,
                        svmul_m(ptrue, svld1(ptrue, oaccum + split * oaccum_split_stride + i), scale[split]));
                    sve01 = svadd_m(ptrue, sve01,
                        svmul_m(ptrue, svld1(ptrue, oaccum + split * oaccum_split_stride + i + 16), scale[split]));
                }
                svfloat32_t sve02 = svuzp1(sve00, sve01);
                svfloat32_t sve03 = svuzp2(sve00, sve01);
                svbfloat16_t sve0 = svcvt_bf16_x(ptrue, sve02);
                sve0 = svcvtnt_bf16_x(sve0, ptrue, sve03);
                svst1(ptrue, o + i, sve0);
            }
        }
    });
}

template <typename DecodeParams, typename KernelTraits>
void splitkv_mla_combine_launch(const DecodeParams &params)
{
    FLASH_MLA_NUM_SPLITS_SWITCH(params.meta->num_thread_parts, max_num_splits, [&] {
        splitkv_mla_combine_kernel<DecodeParams, KernelTraits, max_num_splits>(params);
    });
}

} // namespace flash_mla

} // namespace kutacc