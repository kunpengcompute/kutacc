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

#include <cmath>
#include <arm_sve.h>

#include "kutacc.h"

namespace kutacc {

template <typename Kernel_traits, bool Split>
void mla_store(const FlashMLAFwdParams &params, const int bidb, const int bidh, const int m_block,
    const int n_split_idx, typename Kernel_traits::ElementAccum *block_o,
    const typename Kernel_traits::ElementAccum *row_max, const typename Kernel_traits::ElementAccum *row_sum)
{
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int split_offset = params.num_splits_ptr[bidb];

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride +
        bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q +
        m_block * kBlockM) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q +
        m_block * kBlockM;
 
    ElementO *O = reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) +
        (Split ? row_offset_oaccum : row_offset_o);
    ElementAccum *LSE =
        reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) +
        (Split ? row_offset_lseaccum : row_offset_lse);

    const index_t o_row_stride = Split ? params.d_v : params.o_row_stride;

    bool Is_dropout = false;
    float rp_dropout = 1.0;
    for (int i = 0; i < kBlockM && i < params.seqlen_q - m_block * kBlockM; ++i) {
        float sum = row_sum[i];
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        LSE[i] =
            (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max[i] * params.scale_softmax + logf(sum);
        float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
        svbool_t ptrue = svptrue_b16();
        if constexpr(!Split) {
            for (int j = 0; j < kHeadDimV; j += 32) {
                svfloat32_t sve0 = svld1(ptrue, block_o + i * kHeadDimV + j);
                svfloat32_t sve1 = svld1(ptrue, block_o + i * kHeadDimV + j + 16);
                sve0 = svmul_n_f32_m(ptrue, sve0, scale);
                sve1 = svmul_n_f32_m(ptrue, sve1, scale);

                svbfloat16_t sve = svuzp1(svcvt_bf16_x(ptrue, sve0), svcvt_bf16_x(ptrue, sve1));

                svst1(ptrue, O + i * o_row_stride + j, sve);
            }
        } else {
            for (int j = 0; j < kHeadDimV; j += 16) {
                svfloat32_t sve = svld1(ptrue, block_o + i * kHeadDimV + j);
                sve = svmul_n_f32_m(ptrue, sve, scale);
                svst1(ptrue, O + i * o_row_stride + j, sve);
            }
        }
    }
}

} // namespace kutacc