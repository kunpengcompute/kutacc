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
#include <cstring>

#include "math/fast_exp.h"
#include "attention/flash_mla/utils.h"
#include "attention/flash_mla/config.h"
#include "attention/flash_mla/decode/params.h"

#include "store.h"
#include "common.h"

namespace kutacc {

namespace flash_mla {

template <typename KernelTraits>
void sparse_mla_h64_kernel(
    const SparseDecodeParams &params,
    const int64_t req_idx,
    const int64_t q_token_idx,
    const int64_t q_block_idx,
    const int64_t split_idx,
    const int64_t seqlen_kv,
    const int64_t begin_kv_token_idx,
    const int64_t end_kv_token_idx,
    double *time_record
)
{
    constexpr auto q_block_size = KernelTraits::q_block_size;
    constexpr auto head_dim = KernelTraits::head_dim;
    constexpr auto head_dim_v = KernelTraits::head_dim_v;

    static_assert(q_block_size == 64);
    static_assert(head_dim % 32 == 0);
    static_assert(head_dim_v % 32 == 0);

    constexpr int max_tile_len = 256;

    constexpr auto block_q_bytes = q_block_size * head_dim * sizeof(bfloat16_t);
    constexpr auto block_s_bytes = q_block_size * max_tile_len * sizeof(float);
    constexpr auto block_o_bytes = q_block_size * head_dim_v * sizeof(float);
    constexpr auto micro_block_k_bytes = 16 * head_dim * sizeof(bfloat16_t);
    constexpr auto softmax_bytes = q_block_size * sizeof(float);
    constexpr auto kv_token_addr_bytes = (max_tile_len + 16) * sizeof(bfloat16_t *);

    constexpr auto size0 = std::max(block_q_bytes + block_s_bytes, block_o_bytes);
    static_assert(size0 + micro_block_k_bytes + block_o_bytes + softmax_bytes * 3 + kv_token_addr_bytes <=
        MAX_THREAD_BUFFER_BYTES);

    int8_t *thread_buffer = (int8_t *)params.threads_buffer + get_thread_id() * MAX_THREAD_BUFFER_BYTES;
    bfloat16_t *block_q = (bfloat16_t *)thread_buffer;
    float *out_o = (float *)thread_buffer;
    float *block_s = (float *)(thread_buffer + block_q_bytes);
    bfloat16_t *block_p = (bfloat16_t *)block_s;
    bfloat16_t *micro_block_k = (bfloat16_t *)(thread_buffer += size0);
    float *block_o = (float *)(thread_buffer += micro_block_k_bytes);
    float *row_max = (float *)(thread_buffer += block_o_bytes);
    float *row_sum = (float *)(thread_buffer += softmax_bytes);
    float *scale_o = (float *)(thread_buffer += softmax_bytes);
    bfloat16_t **kv_token_addr = (bfloat16_t **)(thread_buffer += softmax_bytes);

    const float softmax_scale_log2 = params.softmax_scale_log2;

    const auto indices = &params.indices->index({req_idx, q_token_idx, 0});

    const auto extra_indices =
        !params.extra_indices->has_value() ? nullptr :
            &params.extra_indices->value().index({req_idx, q_token_idx, 0});

    bfloat16_t *kvcache = params.kvcache.data_ptr();

    bfloat16_t *extra_kvcache = params.extra_kvcache.has_value() ? params.extra_kvcache.value().data_ptr() : nullptr;

    const bfloat16_t *q = &params.q->index({req_idx, q_token_idx, q_block_idx * q_block_size, 0});

    const int64_t q_head_stride = params.q->stride(2);

    const auto topk = params.topk_length->has_value() ? params.topk_length->value().index({req_idx}) : params.topk;

    MATRIX_ON();

    const svbool_t ptrue = svptrue_b16();

    if (likely((q_block_idx + 1) * q_block_size <= params.num_heads_q)) {
        for (int j = 0; j < head_dim; j += 32) {
            for (int head_idx = 0; head_idx < q_block_size; head_idx += 16) {
                auto q0 = q + head_idx * q_head_stride + j;
                for (int i = 0; i < 16; ++i) {
                    svld1_hor_za32(0, i, ptrue, q0 + i * q_head_stride);
                }
                auto q1 = block_q + j * q_block_size + head_idx * 2;
                for (int i = 0; i < 16; ++i) {
                    svst1_ver_za32(0, i, ptrue, q1 + i * q_block_size * 2);
                }
            }
        }
    } else {
        FLASH_MLA_CHECK(0, "");
    }

    auto kv0_begin = begin_kv_token_idx;
    auto kv0_end = std::min(topk, end_kv_token_idx);
    kv_back_check(indices, kv0_begin, kv0_end);
    const auto kv0_len = kv0_end - kv0_begin;

    auto kv1_begin = std::max<int64_t>(0, begin_kv_token_idx - topk);
    auto kv1_end = end_kv_token_idx - topk;
    kv_back_check(extra_indices, kv1_begin, kv1_end);
    const auto kv1_len = kv1_end - kv1_begin;

    const auto total_kv_len = kv0_len + kv1_len;

    bool is_first_step = true;

    for (int64_t kv_idx = 0; kv_idx < total_kv_len; kv_idx += max_tile_len) {
        const int kv_len = std::min<int64_t>(max_tile_len, total_kv_len - kv_idx);
        const auto ceil_kv_len = ceil(kv_len, 16);
        const auto pad_kv_len = ceil_kv_len + 16;

        const auto cur_kv0_begin = kv0_begin + kv_idx;
        const auto cur_kv0_end = std::min(kv0_end, kv0_begin + kv_idx + pad_kv_len);
        const auto cur_kv0_len = std::max<int64_t>(0, cur_kv0_end - cur_kv0_begin);

        const auto cur_kv1_begin = std::max(kv1_begin, kv1_begin + (kv_idx - kv0_len));
        const auto cur_kv1_end = std::min(kv1_end, kv1_begin + (kv_idx - kv0_len) + pad_kv_len);
        const auto cur_kv1_len = std::max<int64_t>(0, cur_kv1_end - cur_kv1_begin);

        if (cur_kv0_len > 0) {
            get_kv_token_addr<head_dim>(kvcache, indices + cur_kv0_begin, kv_token_addr, cur_kv0_len);
        }

        if (cur_kv1_len > 0) {
            get_kv_token_addr<head_dim>(
                extra_kvcache, extra_indices + cur_kv1_begin, kv_token_addr + cur_kv0_len, cur_kv1_len);
        }

        const auto pad_begin = cur_kv0_len + cur_kv1_len;
        for (int64_t i = pad_begin; i < pad_kv_len; ++i) {
            kv_token_addr[i] = kv_token_addr[pad_begin - 1];
        }

        for (int offset_n = 0; offset_n < kv_len; offset_n += 16) {
            auto cur_kv_token_addr = kv_token_addr + offset_n;
            for (int j = 0; j < head_dim; j += 32) {
                for (int i = 0; i < 16; ++i) {
                    svld1_hor_za32(0, i, ptrue, cur_kv_token_addr[i] + j);
                }
                auto k1 = micro_block_k + j * 16;
                for (int i = 0; i < 16; ++i) {
                    svst1_ver_za32(0, i, ptrue, k1 + i * 32);
                }
            }
            svzero_za();
            auto prf_kv_token_addr = kv_token_addr + offset_n + 16;
            bfloat16_t *a = block_q;
            bfloat16_t *b = micro_block_k;
            for (int k = 0; k < head_dim / 2; ++k) {
                svprfh(ptrue,
                    prf_kv_token_addr[k / (head_dim / 32)] + k % (head_dim / 32) * 32, SV_PLDL2KEEP);
                svbfloat16_t vb = svld1(ptrue, b);
                svbfloat16_t va0 = svld1(ptrue, a);
                svmopa_za32_bf16_m(0, ptrue, ptrue, va0, vb);
                svbfloat16_t va1 = svld1(ptrue, a + 32);
                svmopa_za32_bf16_m(1, ptrue, ptrue, va1, vb);
                svbfloat16_t va2 = svld1(ptrue, a + 64);
                svmopa_za32_bf16_m(2, ptrue, ptrue, va2, vb);
                svbfloat16_t va3 = svld1(ptrue, a + 96);
                svmopa_za32_bf16_m(3, ptrue, ptrue, va3, vb);
                a += 128;
                b += 32;
            }

            float *c0 = block_s + offset_n * 16;
            float *c1 = block_s + offset_n * 16 + 16 * ceil_kv_len;
            float *c2 = block_s + offset_n * 16 + 32 * ceil_kv_len;
            float *c3 = block_s + offset_n * 16 + 48 * ceil_kv_len;
            for (int i = 0; i < 16; ++i) {
                svst1_ver_za32(0, i, ptrue, c0 + i * 16);
                svst1_ver_za32(1, i, ptrue, c1 + i * 16);
                svst1_ver_za32(2, i, ptrue, c2 + i * 16);
                svst1_ver_za32(3, i, ptrue, c3 + i * 16);
            }
        }

        for (int offset_m = 0; offset_m < 64; offset_m += 16) {
            float *block_s_begin = block_s + offset_m * ceil_kv_len;
            bfloat16_t *block_p_begin = block_p + offset_m * ceil_kv_len;

            svfloat32_t sve_row_max =
                unlikely(is_first_step) ? svdup_f32(-INFINITY) : svld1(ptrue, row_max + offset_m);
            svfloat32_t sve_max_prev = sve_row_max;

            const auto end = kv_len;

            for (int i = 0; i < end; ++i) {
                sve_row_max = svmax_m(ptrue, sve_row_max, svld1(ptrue, block_s_begin + i * 16));
            }
            svfloat32_t sve_scale_o;
            svfloat32_t sve_row_sum;
            if (unlikely(is_first_step)) {
                sve_scale_o = svdup_f32(0);
                sve_row_sum = svdup_f32(0);
            } else {
                sve_scale_o =
                    kutacc::fast_exp2(ptrue,
                        svmul_m(ptrue, svsub_m(ptrue, sve_max_prev, sve_row_max), softmax_scale_log2));
                sve_row_sum = svmul_m(ptrue, svld1(ptrue, row_sum + offset_m), sve_scale_o);
            }
            for (int i = 0; i < (end + 1) / 2; ++i) {
                constexpr int prf_stride = 16 * 64;
                svprfw(ptrue, block_s_begin + i * 2 * 16 + prf_stride, SV_PLDL1KEEP);
                svprfw(ptrue, block_s_begin + (i * 2 + 1) * 16 + prf_stride, SV_PLDL1KEEP);
                svfloat32_t sve_s0;
                svfloat32_t sve_s1;
                sve_s0 = svld1(ptrue, block_s_begin + i * 2 * 16);
                sve_s0 = svsub_m(ptrue, sve_s0, sve_row_max);
                sve_s0 = svmul_m(ptrue, sve_s0, softmax_scale_log2);
                sve_s0 = kutacc::fast_exp2(ptrue, sve_s0);
                sve_row_sum = svadd_m(ptrue, sve_row_sum, sve_s0);
                if (likely(i * 2 + 1 < end)) {
                    sve_s1 = svld1(ptrue, block_s_begin + (i * 2 + 1) * 16);
                    sve_s1 = svsub_m(ptrue, sve_s1, sve_row_max);
                    sve_s1 = svmul_m(ptrue, sve_s1, softmax_scale_log2);
                    sve_s1 = kutacc::fast_exp2(ptrue, sve_s1);
                    sve_row_sum = svadd_m(ptrue, sve_row_sum, sve_s1);
                } else {
                    sve_s1 = svdup_f32(0);
                }
                svbfloat16_t sve = svcvt_bf16_f32_z(ptrue, sve_s0);
                sve = svcvtnt_bf16_f32_m(sve, ptrue, sve_s1);
                svst1(ptrue, block_p_begin + i * 32, sve);
            }
            svst1(ptrue, scale_o + offset_m, sve_scale_o);
            svst1(ptrue, row_max + offset_m, sve_row_max);
            svst1(ptrue, row_sum + offset_m, sve_row_sum);
            svbfloat16_t sve_zero = svdup_bf16(0);
            for (int i = (end + 1) / 2; i < ceil_kv_len / 2; ++i) {
                svst1(ptrue, block_p_begin + i * 32, sve_zero);
            }
        }

        for (int offset_n = 0; offset_n < head_dim_v; offset_n += 32) {
            {
                const int offset_m = 0;
                svzero_za();
                bfloat16_t *a0 = block_p + offset_m * ceil_kv_len;
                bfloat16_t *a1 = block_p + (offset_m + 16) * ceil_kv_len;
                for (int i = 0; i < kv_len; i += 2) {
                    if (likely(offset_n + 32 < head_dim_v)) {
                        svprfh(ptrue, kv_token_addr[i] + offset_n + 32, SV_PLDL2KEEP);
                    }
                    svbfloat16_t vb0 = svld1(ptrue, kv_token_addr[i] + offset_n);
                    svbfloat16_t vb1 = svld1(ptrue, kv_token_addr[i + 1] + offset_n);
                    svbfloat16_t vb2 = svzip1(vb0, vb1);
                    svbfloat16_t va0 = svld1(ptrue, a0);
                    svmopa_za32_bf16_m(0, ptrue, ptrue, va0, vb2);
                    svbfloat16_t vb3 = svzip2(vb0, vb1);
                    svmopa_za32_bf16_m(1, ptrue, ptrue, va0, vb3);
                    svbfloat16_t va1 = svld1(ptrue, a1);
                    svmopa_za32_bf16_m(2, ptrue, ptrue, va1, vb2);
                    svmopa_za32_bf16_m(3, ptrue, ptrue, va1, vb3);
                    a0 += 32;
                    a1 += 32;
                }
                float *c0 = block_o + offset_n * 64 + offset_m * 32;
                float *c1 = block_o + offset_n * 64 + offset_m * 32 + 16;
                float *c2 = block_o + offset_n * 64 + (offset_m + 16) * 32;
                float *c3 = block_o + offset_n * 64 + (offset_m + 16) * 32 + 16;
                for (int t = 0; t < 16; ++t) {
                    svfloat32_t vc0 = svread_hor_za32_m(vc0, ptrue, 0, t);
                    svfloat32_t vc1 = svread_hor_za32_m(vc1, ptrue, 1, t);
                    svfloat32_t vc2 = svread_hor_za32_m(vc2, ptrue, 2, t);
                    svfloat32_t vc3 = svread_hor_za32_m(vc3, ptrue, 3, t);

                    if (likely(!is_first_step)) {
                        svfloat32_t vd0 = svld1(ptrue, c0);
                        svfloat32_t vd1 = svld1(ptrue, c1);
                        svfloat32_t vd2 = svld1(ptrue, c2);
                        svfloat32_t vd3 = svld1(ptrue, c3);

                        vc0 = svadd_m(ptrue, vc0, svmul_m(ptrue, vd0, scale_o[offset_m + t]));
                        vc1 = svadd_m(ptrue, vc1, svmul_m(ptrue, vd1, scale_o[offset_m + t]));
                        vc2 = svadd_m(ptrue, vc2, svmul_m(ptrue, vd2, scale_o[offset_m + 16 + t]));
                        vc3 = svadd_m(ptrue, vc3, svmul_m(ptrue, vd3, scale_o[offset_m + 16 + t]));
                    }

                    svst1(ptrue, c0, vc0);
                    svst1(ptrue, c1, vc1);
                    svst1(ptrue, c2, vc2);
                    svst1(ptrue, c3, vc3);

                    c0 += 32;
                    c1 += 32;
                    c2 += 32;
                    c3 += 32;
                }
            }
            {
                const int offset_m = 32;
                svzero_za();
                bfloat16_t *a0 = block_p + offset_m * ceil_kv_len;
                bfloat16_t *a1 = block_p + (offset_m + 16) * ceil_kv_len;
                for (int i = 0; i < kv_len; i += 2) {
                    if (likely(offset_n + 32 < head_dim_v)) {
                        svprfh(ptrue, kv_token_addr[i + 1] + offset_n + 32, SV_PLDL2KEEP);
                    }
                    svbfloat16_t vb0 = SVLDNT1_INDEX_BF16(ptrue, kv_token_addr[i], offset_n);
                    svbfloat16_t vb1 = SVLDNT1_INDEX_BF16(ptrue, kv_token_addr[i + 1], offset_n);
                    svbfloat16_t vb2 = svzip1(vb0, vb1);
                    svbfloat16_t va0 = svld1(ptrue, a0);
                    svmopa_za32_bf16_m(0, ptrue, ptrue, va0, vb2);
                    svbfloat16_t vb3 = svzip2(vb0, vb1);
                    svmopa_za32_bf16_m(1, ptrue, ptrue, va0, vb3);
                    svbfloat16_t va1 = svld1(ptrue, a1);
                    svmopa_za32_bf16_m(2, ptrue, ptrue, va1, vb2);
                    svmopa_za32_bf16_m(3, ptrue, ptrue, va1, vb3);
                    a0 += 32;
                    a1 += 32;
                }
                float *c0 = block_o + offset_n * 64 + offset_m * 32;
                float *c1 = block_o + offset_n * 64 + offset_m * 32 + 16;
                float *c2 = block_o + offset_n * 64 + (offset_m + 16) * 32;
                float *c3 = block_o + offset_n * 64 + (offset_m + 16) * 32 + 16;
                for (int t = 0; t < 16; ++t) {
                    svfloat32_t vc0 = svread_hor_za32_m(vc0, ptrue, 0, t);
                    svfloat32_t vc1 = svread_hor_za32_m(vc1, ptrue, 1, t);
                    svfloat32_t vc2 = svread_hor_za32_m(vc2, ptrue, 2, t);
                    svfloat32_t vc3 = svread_hor_za32_m(vc3, ptrue, 3, t);

                    if (likely(!is_first_step)) {
                        svfloat32_t vd0 = svld1(ptrue, c0);
                        svfloat32_t vd1 = svld1(ptrue, c1);
                        svfloat32_t vd2 = svld1(ptrue, c2);
                        svfloat32_t vd3 = svld1(ptrue, c3);

                        vc0 = svadd_m(ptrue, vc0, svmul_m(ptrue, vd0, scale_o[offset_m + t]));
                        vc1 = svadd_m(ptrue, vc1, svmul_m(ptrue, vd1, scale_o[offset_m + t]));
                        vc2 = svadd_m(ptrue, vc2, svmul_m(ptrue, vd2, scale_o[offset_m + 16 + t]));
                        vc3 = svadd_m(ptrue, vc3, svmul_m(ptrue, vd3, scale_o[offset_m + 16 + t]));
                    }

                    svst1(ptrue, c0, vc0);
                    svst1(ptrue, c1, vc1);
                    svst1(ptrue, c2, vc2);
                    svst1(ptrue, c3, vc3);

                    c0 += 32;
                    c1 += 32;
                    c2 += 32;
                    c3 += 32;
                }
            }
        }
        is_first_step = false;
    }
    for (int i = 0; i < q_block_size; ++i) {
        for (int j = 0; j < head_dim_v; j += 32) {
            svst1(ptrue, out_o + i * head_dim_v + j, svld1(ptrue, block_o + j * q_block_size + i * 32));
            svst1(ptrue, out_o + i * head_dim_v + j + 16, svld1(ptrue, block_o + j * q_block_size + i * 32 + 16));
        }
    }
    MATRIX_OFF();

    if (is_first_step) {
        memset(out_o, 0, sizeof(float) * q_block_size * head_dim_v);
        for (int i = 0; i < q_block_size; ++i) {
            row_sum[i] = 0;
            row_max[i] = -INFINITY;
        }
    }

    const bool is_no_split = begin_kv_token_idx == 0 && end_kv_token_idx == seqlen_kv;

    if (is_no_split) {
        store<SparseDecodeParams, KernelTraits, true>(
            params, req_idx, q_token_idx, q_block_idx, split_idx, out_o, row_max, row_sum);
    } else {
        store<SparseDecodeParams, KernelTraits, false>(
            params, req_idx, q_token_idx, q_block_idx, split_idx, out_o, row_max, row_sum);
    }
}

} // namespace flash_mla

} // namespace kutacc