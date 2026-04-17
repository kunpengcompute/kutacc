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
#include <arm_sme.h>

#include "kernel_common.h"
#include "mla_store_bf16.h"

namespace kutacc {

template <typename Kernel_traits, bool Is_causal>
void compute_attn_1rowblock_bf16_32x64(const FlashMLAFwdParams &params,
                                       const int bidb, const int bidh, const int m_block,
                                       const int n_split_idx, const int seqlen_k,
                                       const int n_block_min, const int n_block_max, const bool NoSplit,
                                       char *tiling_buffer_ptr)
{
    static constexpr int kBlockM = Kernel_traits::kBlockM;
    static constexpr int kBlockN = Kernel_traits::kBlockN;
    static constexpr int kHeadDim = Kernel_traits::kHeadDim;
    static constexpr int kHeadDimV = Kernel_traits::kHeadDimV;

    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    const int *block_table = params.block_table + bidb * params.block_table_batch_stride;

    const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride +
        bidh * params.q_head_stride;

    static_assert(kBlockM == 32);
    static_assert(kBlockN == 64);

    KUTACC_CHECK(params.h == 1, ""); // Guaranteed to be MQA

    static_assert(
        (32 * 576 + 32 * 576) * sizeof(Element) + (32 * 128 + 32 * 512 + 32 + 32 + 32) * sizeof(ElementAccum) <=
        kutacc::extra_buffer_size_per_thread);

    char *thread_ptr = tiling_buffer_ptr + kutacc::get_thread_id() * kutacc::extra_buffer_size_per_thread;
    Element *block_q = (Element *)alloc_buffer(thread_ptr, 32 * 576 * sizeof(Element));
    Element *block_k = (Element *)alloc_buffer(thread_ptr, 32 * 576 * sizeof(Element));
    ElementAccum *block_s = (ElementAccum *)alloc_buffer(thread_ptr, 32 * 128 * sizeof(ElementAccum));
    Element *block_p = (Element *)block_s;
    ElementAccum *block_o = (ElementAccum *)alloc_buffer(thread_ptr, 32 * 512 * sizeof(ElementAccum));
    ElementAccum *row_max = (ElementAccum *)alloc_buffer(thread_ptr, 32 * sizeof(ElementAccum));
    ElementAccum *row_sum = (ElementAccum *)alloc_buffer(thread_ptr, 32 * sizeof(ElementAccum));
    ElementAccum *scale_o = (ElementAccum *)alloc_buffer(thread_ptr, 32 * sizeof(ElementAccum));
    ElementAccum *out_o = (ElementAccum *)block_q;

    const Element *Q = reinterpret_cast<Element *>(params.q_ptr);
    const Element *K = reinterpret_cast<Element *>(params.k_ptr);

    MATRIX_ON();

    const svbool_t ptrue = svptrue_b16();

    if (likely((m_block + 1) * 32 <= params.seqlen_q)) {
        for (int j = 0; j < 576; j += 64) {
            const bfloat16_t *q0 = Q + row_offset_q + j;
            const bfloat16_t *q1 = Q + row_offset_q + 16 * 576 + j;
            for (int i = 0; i < 16; ++i) {
                if (likely(j + 128 < 576)) {
                    prefetch_L1(q0 + 128);
                    prefetch_L1(q0 + 128 + 32);
                    prefetch_L1(q1 + 128);
                    prefetch_L1(q1 + 128 + 32);
                }
                svld1_hor_za32(0, i, ptrue, q0);
                svld1_hor_za32(1, i, ptrue, q0 + 32);
                svld1_hor_za32(2, i, ptrue, q1);
                svld1_hor_za32(3, i, ptrue, q1 + 32);
                q0 += 576;
                q1 += 576;
            }
            for (int i = 0; i < 16; ++i) {
                svst1_ver_za32(0, i, ptrue, block_q + j * 32 + i * 64);
                svst1_ver_za32(1, i, ptrue, block_q + j * 32 + (i + 16) * 64);
                svst1_ver_za32(2, i, ptrue, block_q + j * 32 + i * 64 + 32);
                svst1_ver_za32(3, i, ptrue, block_q + j * 32 + (i + 16) * 64 + 32);
            }
        }
    } else {
        for (int j = 0; j < 576; j += 2) {
            for (int i = 0; i < 32 && i < params.seqlen_q - m_block * 32; ++i) {
                block_q[j * 32 + 2 * i] = Q[row_offset_q + i * 576 + j];
                block_q[j * 32 + 2 * i + 1] = Q[row_offset_q + i * 576 + j + 1];
            }
        }
    }

    const bool is_odd = (n_block_max - n_block_min) & 1;
    const float &scale_softmax_log2 = params.scale_softmax_log2;
    for (int n_block = n_block_min; n_block < n_block_max;) {
        const index_t row_offset_k = bidh * params.k_head_stride;
        const bool is_first_step = n_block == n_block_min;
        const int tiled_n = is_first_step && is_odd ? 64 : 128;
        const bfloat16_t *kv_ptr[4];
        kv_ptr[0] = K + block_table[n_block] * params.k_batch_stride + row_offset_k;
        kv_ptr[1] =
            n_block + 1 < n_block_max ? K + block_table[n_block + 1] * params.k_batch_stride + row_offset_k : nullptr;
        kv_ptr[2] =
            n_block + 2 < n_block_max ? K + block_table[n_block + 2] * params.k_batch_stride + row_offset_k : nullptr;
        kv_ptr[3] =
            n_block + 3 < n_block_max ? K + block_table[n_block + 3] * params.k_batch_stride + row_offset_k : nullptr;
        const bfloat16_t *prefetch_ptr = nullptr;
        bool is_odd_first = is_first_step && is_odd;
        if (is_odd_first) {
            prefetch_ptr = kv_ptr[1];
        } else {
            prefetch_ptr = kv_ptr[1] + 16 * 576;
        }
        
        for (int offset_n = 0; offset_n < tiled_n; offset_n += 32) {
            const bfloat16_t *cur_kv = kv_ptr[offset_n / 64] + offset_n % 64 * 576;
            for (int j = 0; j < 576; j += 64) {
                const bfloat16_t *k0 = cur_kv + j;
                const bfloat16_t *k1 = cur_kv + 16 * 576 + j;
                for (int i = 0; i < 16; ++i) {
                    if (likely(j + 64 < 576)) {
                        prefetch_L1(k0 + 64);
                        prefetch_L1(k0 + 64 + 32);
                        prefetch_L1(k1 + 64);
                        prefetch_L1(k1 + 64 + 32);
                    }
                    svld1_hor_za32(0, i, ptrue, k0);
                    svld1_hor_za32(1, i, ptrue, k0 + 32);
                    svld1_hor_za32(2, i, ptrue, k1);
                    svld1_hor_za32(3, i, ptrue, k1 + 32);
                    k0 += 576;
                    k1 += 576;
                }
                for (int i = 0; i < 16; ++i) {
                    svst1_ver_za32(0, i, ptrue, block_k + j * 32 + i * 64);
                    svst1_ver_za32(1, i, ptrue, block_k + j * 32 + (i + 16) * 64);
                    svst1_ver_za32(2, i, ptrue, block_k + j * 32 + i * 64 + 32);
                    svst1_ver_za32(3, i, ptrue, block_k + j * 32 + (i + 16) * 64 + 32);
                }
            }

            if (offset_n == 96) {
                prefetch_ptr = kv_ptr[2];
            }

            svzero_za();
            bfloat16_t *data_atmp = block_q;
            bfloat16_t *data_btmp = block_k;
            for (int i = 0; i < 576 / 2; ++i) {
                prefetch_L1(data_atmp + 64 * 6);
                prefetch_L1(data_atmp + 64 * 6 + 32);
                prefetch_L1(data_btmp + 64 * 6);
                prefetch_L1(data_btmp + 64 * 6 + 32);
                
                svbfloat16_t va0 = svld1_bf16(ptrue, data_atmp);
                svbfloat16_t vb0 = svld1_bf16(ptrue, data_btmp);
                svmopa_za32_bf16_m(0, ptrue, ptrue, va0, vb0);
                svbfloat16_t vb1 = svld1_bf16(ptrue, data_btmp + 32);
                svmopa_za32_bf16_m(1, ptrue, ptrue, va0, vb1);
                svbfloat16_t va1 = svld1_bf16(ptrue, data_atmp + 32);
                svmopa_za32_bf16_m(2, ptrue, ptrue, va1, vb0);
                svmopa_za32_bf16_m(3, ptrue, ptrue, va1, vb1);
                data_atmp += 64;
                data_btmp += 64;
                if (likely(prefetch_ptr != nullptr)) {
                    prefetch_L2(prefetch_ptr);
                    prefetch_ptr += 32;
                }
            }

            float *matd0 = block_s + offset_n * 16;
            float *matd1 = block_s + offset_n * 16 + 16 * 16;
            float *matd2 = block_s + offset_n * 16 + 16 * tiled_n;
            float *matd3 = block_s + offset_n * 16 + 16 * tiled_n + 16 * 16;
            for (int t = 0; t < 16; ++t) {
                svst1_ver_za32(0, t, ptrue, matd0);
                svst1_ver_za32(1, t, ptrue, matd1);
                svst1_ver_za32(2, t, ptrue, matd2);
                svst1_ver_za32(3, t, ptrue, matd3);
                matd0 += 16;
                matd1 += 16;
                matd2 += 16;
                matd3 += 16;
            }
        }

        for (int offset_m = 0; offset_m < 32; offset_m += 16) {
            float *block_s_begin = block_s + offset_m * tiled_n;
            bfloat16_t *block_p_begin = block_p + offset_m * 2;

            svfloat32_t sve_row_max = unlikely(is_first_step) ? svdup_f32(-INFINITY) : svld1(ptrue, row_max + offset_m);
            svfloat32_t sve_max_prev = sve_row_max;

            int end = seqlen_k - n_block * kBlockN;
            if (end > tiled_n) {
                end = tiled_n;
            }

            int causal_row_start = params.seqlen_q - 1 - (m_block * kBlockM + offset_m) - (params.ngroups - 1) -
                (seqlen_k - 1 - n_block * kBlockN) * params.ngroups;
            for (int i = 0; i < end; ++i) {
                auto pred = ptrue;
                if constexpr(Is_causal) {
                    pred = svwhilege_b32(15, causal_row_start + i * params.ngroups);
                }
                sve_row_max = svmax_m(pred, sve_row_max, svld1(pred, block_s_begin + i * 16));
            }
            svfloat32_t sve_scale_o;
            svfloat32_t sve_row_sum;
            if (unlikely(is_first_step)) {
                sve_scale_o = svdup_f32(0);
                sve_row_sum = svdup_f32(0);
            } else {
                auto pred = ptrue;
                if constexpr(Is_causal) {
                    pred = svwhilege_b32(15, causal_row_start);
                }
                sve_scale_o = kutacc::fast_exp2(
                    pred, svmul_m(pred, svsub_m(pred, sve_max_prev, sve_row_max), scale_softmax_log2));
                if constexpr(Is_causal) {
                    sve_scale_o = svdup_f32_m(sve_scale_o, svwhilelt_b32(0, causal_row_start), 1);
                }
                sve_row_sum = svmul_m(pred, svld1(ptrue, row_sum + offset_m), sve_scale_o);
            }
            for (int i = 0; i < (end + 1) / 2; ++i) {
                if (offset_m + 16 < 32) {
                    prefetch_L1(block_s_begin + i * 2 * 16 + 16 * tiled_n);
                    prefetch_L1(block_s_begin + (i * 2 + 1) * 16 + 16 * tiled_n);
                }
                svfloat32_t sve_s0;
                svfloat32_t sve_s1;
                auto pred0 = ptrue;
                auto pred1 = ptrue;
                if constexpr(Is_causal) {
                    pred0 = svwhilege_b32(15, causal_row_start + i * 2 * params.ngroups);
                    pred1 = svwhilege_b32(15, causal_row_start + (i * 2 + 1) * params.ngroups);
                }
                sve_s0 = svld1(pred0, block_s_begin + i * 2 * 16);
                sve_s0 = svsub_m(pred0, sve_s0, sve_row_max);
                sve_s0 = svmul_m(pred0, sve_s0, scale_softmax_log2);
                sve_s0 = kutacc::fast_exp2(pred0, sve_s0);
                sve_row_sum = svadd_m(pred0, sve_row_sum, sve_s0);
                if (likely(i * 2 + 1 < end)) {
                    sve_s1 = svld1(pred1, block_s_begin + (i * 2 + 1) * 16);
                    sve_s1 = svsub_m(pred1, sve_s1, sve_row_max);
                    sve_s1 = svmul_m(pred1, sve_s1, scale_softmax_log2);
                    sve_s1 = kutacc::fast_exp2(pred1, sve_s1);
                    sve_row_sum = svadd_m(pred1, sve_row_sum, sve_s1);
                } else {
                    sve_s1 = svdup_f32(0);
                }
                if (likely(prefetch_ptr != nullptr)) {
                    prefetch_L2(prefetch_ptr);
                    prefetch_ptr += 32;
                }
                svbfloat16_t sve = svcvt_bf16_f32_z(pred0, sve_s0);
                sve = svcvtnt_bf16_f32_m(sve, pred1, sve_s1);
                svst1(ptrue, block_p_begin + i * 64, sve);
            }
            svst1(ptrue, scale_o + offset_m, sve_scale_o);
            svst1(ptrue, row_max + offset_m, sve_row_max);
            svst1(ptrue, row_sum + offset_m, sve_row_sum);
            svbfloat16_t sve_zero = svdup_bf16(0);
            for (int i = (end + 1) / 2; i < tiled_n / 2; ++i) {
                svst1(ptrue, block_p_begin + i * 64, sve_zero);
            }
        }

        for (int offset_n = 0; offset_n < 512; offset_n += 32) {
            svzero_za();
            bfloat16_t *data_atmp = block_p;
            for (int j = 0; j < tiled_n; j += 64) {
                if (unlikely(!is_odd_first && kv_ptr[2] != nullptr && prefetch_ptr == kv_ptr[2] + 64 * 576)) {
                    prefetch_ptr = kv_ptr[3];
                }
                const bfloat16_t *data_btmp = kv_ptr[j / 64] + offset_n;
                for (int i = 0; i < 64 / 2; ++i) {
                    svbfloat16_t vb0 = svldnt1_bf16(ptrue, data_btmp);
                    svbfloat16_t vb1 = svldnt1_bf16(ptrue, data_btmp + 576);
                    svbfloat16_t vb2 = svzip1_bf16(vb0, vb1);
                    svbfloat16_t va0 = svld1_bf16(ptrue, data_atmp);
                    svmopa_za32_bf16_m(0, ptrue, ptrue, va0, vb2);
                    svbfloat16_t vb3 = svzip2_bf16(vb0, vb1);
                    svmopa_za32_bf16_m(1, ptrue, ptrue, va0, vb3);
                    svbfloat16_t va1 = svld1_bf16(ptrue, data_atmp + 32);
                    svmopa_za32_bf16_m(2, ptrue, ptrue, va1, vb2);
                    svmopa_za32_bf16_m(3, ptrue, ptrue, va1, vb3);
                    data_atmp += 64;
                    data_btmp += 2 * 576;
                    if (likely(prefetch_ptr != nullptr)) {
                        prefetch_L2(prefetch_ptr);
                        prefetch_ptr += 32;
                    }
                }
            }
            float *matd0 = block_o + offset_n * 32;
            float *matd1 = block_o + offset_n * 32 + 16;
            float *matd2 = block_o + offset_n * 32 + 16 * 32;
            float *matd3 = block_o + offset_n * 32 + 16 * 32 + 16;
            for (int t = 0; t < 16; ++t) {
                svfloat32_t sve0 = svread_hor_za32_m(sve0, ptrue, 0, t);
                svfloat32_t sve1 = svread_hor_za32_m(sve1, ptrue, 1, t);
                svfloat32_t sve2 = svread_hor_za32_m(sve2, ptrue, 2, t);
                svfloat32_t sve3 = svread_hor_za32_m(sve3, ptrue, 3, t);

                if (likely(!is_first_step)) {
                    svfloat32_t vd0 = svld1(ptrue, matd0);
                    svfloat32_t vd1 = svld1(ptrue, matd1);
                    svfloat32_t vd2 = svld1(ptrue, matd2);
                    svfloat32_t vd3 = svld1(ptrue, matd3);

                    sve0 = svadd_m(ptrue, sve0, svmul_m(ptrue, vd0, scale_o[t]));
                    sve1 = svadd_m(ptrue, sve1, svmul_m(ptrue, vd1, scale_o[t]));
                    sve2 = svadd_m(ptrue, sve2, svmul_m(ptrue, vd2, scale_o[16 + t]));
                    sve3 = svadd_m(ptrue, sve3, svmul_m(ptrue, vd3, scale_o[16 + t]));
                }

                svst1(ptrue, matd0, sve0);
                svst1(ptrue, matd1, sve1);
                svst1(ptrue, matd2, sve2);
                svst1(ptrue, matd3, sve3);

                matd0 += 32;
                matd1 += 32;
                matd2 += 32;
                matd3 += 32;
            }
        }
        n_block += tiled_n / 64;
    }

    for (int i = 0; i < kBlockM; ++i) {
        for (int j = 0; j < 512; j += 32) {
            svst1(ptrue, out_o + i * 512 + j, svld1(ptrue, block_o + j * kBlockM + i * 32));
            svst1(ptrue, out_o + i * 512 + j + 16, svld1(ptrue, block_o + j * kBlockM + i * 32 + 16));
        }
    }

    MATRIX_OFF();

    if (NoSplit) {
        mla_store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, out_o, row_max, row_sum);
    } else {
        mla_store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, out_o, row_max, row_sum);
    }
}

} // namespace kutacc