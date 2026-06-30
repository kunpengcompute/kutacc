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

#include "math/fast_exp.h"
#include "attention/flash_mla/utils.h"
#include "attention/flash_mla/config.h"
#include "attention/flash_mla/decode/params.h"

#include "store.h"
#include "common.h"

namespace kutacc {

namespace flash_mla {

template <typename KernelTraits>
void dense_mla_h64_kernel(
    const DenseDecodeParams &params,
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
    constexpr auto kv_block_size = KernelTraits::kv_block_size;
    constexpr auto head_dim = KernelTraits::head_dim;
    constexpr auto head_dim_v = KernelTraits::head_dim_v;
    constexpr auto is_causal = KernelTraits::is_causal;

    static_assert(q_block_size == 64);
    static_assert(kv_block_size == 64);
    static_assert(head_dim == 576);
    static_assert(head_dim_v == 512);

    const int n_block_min = begin_kv_token_idx / kv_block_size;
    const int n_block_max = ceil_div<int64_t>(end_kv_token_idx, kv_block_size);

    const int *block_table = &params.block_table->index({req_idx, 0});
    const int64_t row_offset_q = req_idx * params.q->stride(0) + q_block_idx * q_block_size * params.q->stride(2);
    static constexpr int max_tiled_blocks = 4;
    static constexpr int max_tiled_n = 64 * max_tiled_blocks;
    constexpr int size0 = std::max(
        q_block_size * 576 * sizeof(bfloat16_t) + q_block_size * max_tiled_n * sizeof(float),
        q_block_size * 512 * sizeof(float));
    static_assert(
        size0 + 16 * 576 * sizeof(bfloat16_t) +
            (q_block_size * 512 + q_block_size + q_block_size + q_block_size) * sizeof(float) <=
        MAX_THREAD_BUFFER_BYTES);
    char *thread_ptr = (char *)params.threads_buffer + get_thread_id() * MAX_THREAD_BUFFER_BYTES;
    auto alloc_buffer = [](char *&pool_ptr, int size) {
        pool_ptr += size;
        return pool_ptr - size;
    };
    bfloat16_t *block_q = (bfloat16_t *)alloc_buffer(thread_ptr, size0);
    float *out_o = (float *)block_q;
    float *block_s = (float *)(block_q + q_block_size * 576);
    bfloat16_t *block_p = (bfloat16_t *)block_s;
    bfloat16_t *block_k = (bfloat16_t *)alloc_buffer(thread_ptr, 16 * 576 * sizeof(bfloat16_t));
    float *block_o = (float *)alloc_buffer(thread_ptr, q_block_size * 512 * sizeof(float));
    float *row_max = (float *)alloc_buffer(thread_ptr, q_block_size * sizeof(float));
    float *row_sum = (float *)alloc_buffer(thread_ptr, q_block_size * sizeof(float));
    float *scale_o = (float *)alloc_buffer(thread_ptr, q_block_size * sizeof(float));
    const bfloat16_t *Q = reinterpret_cast<bfloat16_t *>(params.q->data_ptr());
    const bfloat16_t *K = reinterpret_cast<bfloat16_t *>(params.kcache->data_ptr());
    MATRIX_ON();
    const svbool_t ptrue = svptrue_b16();
    if (likely((q_block_idx + 1) * 64 <= params.num_heads_q)) {
        for (int j = 0; j < 576; j += 32) {
            for (int head_idx = 0; head_idx < 64; head_idx += 16) {
                auto q0 = Q + row_offset_q + head_idx * 576 + j;
                for (int i = 0; i < 16; ++i) {
                    svld1_hor_za32(0, i, ptrue, q0 + i * 576);
                }
                auto q1 = block_q + j * 64 + head_idx * 2;
                for (int i = 0; i < 16; ++i) {
                    svst1_ver_za32(0, i, ptrue, q1 + i * 64 * 2);
                }
            }
        }
    } else {
        FLASH_MLA_CHECK(0, "");
    }
    const float &softmax_scale_log2 = params.softmax_scale_log2;
    constexpr int snoop_group_size = 2;
    const int num_tiled_blocks = (n_block_max - n_block_min + max_tiled_blocks - 1) / max_tiled_blocks;
    const int tiled_block_offset = num_tiled_blocks * (q_block_idx / snoop_group_size) /
        ((params.num_heads_q + 64 * snoop_group_size - 1) / (64 * snoop_group_size));
    for (int tiled_block_id = 0; tiled_block_id < num_tiled_blocks; ++tiled_block_id) {
        const bool is_first_step = tiled_block_id == 0;
        int n_block = n_block_min + (tiled_block_id + tiled_block_offset) % num_tiled_blocks * max_tiled_blocks;
        int tiled_blocks = std::min(max_tiled_blocks, n_block_max - n_block);
        int tiled_n = tiled_blocks * 64;
        const bfloat16_t *nxt_kv_ptr = tiled_block_id + 1 < num_tiled_blocks ?
            K + block_table[n_block_min + (tiled_block_id + 1 + tiled_block_offset) % num_tiled_blocks *
                max_tiled_blocks] * params.kcache->stride(0) :
            nullptr;
        int len_n = std::min<int64_t>(seqlen_kv - n_block * 64, tiled_n);
        for (int offset_n = 0; offset_n < len_n; offset_n += 16) {
            const bfloat16_t *cur_kv =
                K + block_table[n_block + offset_n / 64] * params.kcache->stride(0) + offset_n % 64 * 576;
            for (int j = 0; j < 576; j += 32) {
                const bfloat16_t *k0 = cur_kv + j;
                for (int i = 0; i < 16; ++i) {
                    svld1_hor_za32(0, i, ptrue, k0 + i * 576);
                }
                auto k1 = block_k + j * 16;
                for (int i = 0; i < 16; ++i) {
                    svst1_ver_za32(0, i, ptrue, k1 + i * 32);
                }
            }
            const int prf_offset_n = offset_n + 16;
            const bfloat16_t *p_b = prf_offset_n < len_n ?
                K + block_table[n_block + prf_offset_n / 64] * params.kcache->stride(0) + (prf_offset_n % 64) * 576 :
                nxt_kv_ptr;
            svzero_za();
            bfloat16_t *a = block_q;
            bfloat16_t *b = block_k;
            for (int k = 0; k < 576 / 2; ++k) {
                if (likely(p_b != nullptr)) {
                    svprfh(ptrue, p_b, SV_PLDL2KEEP);
                    p_b += 32;
                }
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
            float *c1 = block_s + offset_n * 16 + 16 * tiled_n;
            float *c2 = block_s + offset_n * 16 + 32 * tiled_n;
            float *c3 = block_s + offset_n * 16 + 48 * tiled_n;
            for (int i = 0; i < 16; ++i) {
                svst1_ver_za32(0, i, ptrue, c0);
                svst1_ver_za32(1, i, ptrue, c1);
                svst1_ver_za32(2, i, ptrue, c2);
                svst1_ver_za32(3, i, ptrue, c3);
                c0 += 16;
                c1 += 16;
                c2 += 16;
                c3 += 16;
            }
        }
        for (int offset_m = 0; offset_m < 64; offset_m += 16) {
            float *block_s_begin = block_s + offset_m * tiled_n;
            bfloat16_t *block_p_begin = block_p + offset_m * tiled_n;
            svfloat32_t sve_row_max = unlikely(is_first_step) ? svdup_f32(-INFINITY) : svld1(ptrue, row_max + offset_m);
            svfloat32_t sve_max_prev = sve_row_max;
            int end = len_n;
            int64_t causal_row_start = params.num_heads_q - 1 - (q_block_idx * q_block_size + offset_m) -
                (params.casual_group_size - 1) - (seqlen_kv - 1 - n_block * kv_block_size) * params.casual_group_size;
            for (int i = 0; i < end; ++i) {
                auto pred = ptrue;
                if constexpr(is_causal) {
                    pred = svwhilege_b32(15LL, causal_row_start + i * params.casual_group_size);
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
                if constexpr(is_causal) {
                    pred = svwhilege_b32(15LL, causal_row_start);
                }
                sve_scale_o = kutacc::fast_exp2(
                    pred, svmul_m(pred, svsub_m(pred, sve_max_prev, sve_row_max), softmax_scale_log2));
                if constexpr(is_causal) {
                    sve_scale_o = svdup_f32_m(sve_scale_o, svwhilelt_b32(0LL, causal_row_start), 1);
                }
                sve_row_sum = svmul_m(pred, svld1(ptrue, row_sum + offset_m), sve_scale_o);
            }
            for (int i = 0; i < (end + 1) / 2; ++i) {
                constexpr int prf_stride = 16 * 64;
                svprfw(ptrue, block_s_begin + i * 2 * 16 + prf_stride, SV_PLDL1KEEP);
                svprfw(ptrue, block_s_begin + (i * 2 + 1) * 16 + prf_stride, SV_PLDL1KEEP);
                svfloat32_t sve_s0;
                svfloat32_t sve_s1;
                auto pred0 = ptrue;
                auto pred1 = ptrue;
                if constexpr(is_causal) {
                    pred0 = svwhilege_b32(15LL, causal_row_start + i * 2 * params.casual_group_size);
                    pred1 = svwhilege_b32(15LL, causal_row_start + (i * 2 + 1) * params.casual_group_size);
                }
                sve_s0 = svld1(pred0, block_s_begin + i * 2 * 16);
                sve_s0 = svsub_m(pred0, sve_s0, sve_row_max);
                sve_s0 = svmul_m(pred0, sve_s0, softmax_scale_log2);
                sve_s0 = kutacc::fast_exp2(pred0, sve_s0);
                sve_row_sum = svadd_m(pred0, sve_row_sum, sve_s0);
                if (likely(i * 2 + 1 < end)) {
                    sve_s1 = svld1(pred1, block_s_begin + (i * 2 + 1) * 16);
                    sve_s1 = svsub_m(pred1, sve_s1, sve_row_max);
                    sve_s1 = svmul_m(pred1, sve_s1, softmax_scale_log2);
                    sve_s1 = kutacc::fast_exp2(pred1, sve_s1);
                    sve_row_sum = svadd_m(pred1, sve_row_sum, sve_s1);
                } else {
                    sve_s1 = svdup_f32(0);
                }
                svbfloat16_t sve = svcvt_bf16_f32_z(pred0, sve_s0);
                sve = svcvtnt_bf16_f32_m(sve, pred1, sve_s1);
                svst1(ptrue, block_p_begin + i * 32, sve);
            }
            svst1(ptrue, scale_o + offset_m, sve_scale_o);
            svst1(ptrue, row_max + offset_m, sve_row_max);
            svst1(ptrue, row_sum + offset_m, sve_row_sum);
            svbfloat16_t sve_zero = svdup_bf16(0);
            for (int i = (end + 1) / 2; i < tiled_n / 2; ++i) {
                svst1(ptrue, block_p_begin + i * 32, sve_zero);
            }
        }
        for (int offset_n = 0; offset_n < 512; offset_n += 32) {
            for (int offset_m = 0; offset_m < 64; offset_m += 32) {
                svzero_za();
                bfloat16_t *a0 = block_p + offset_m * tiled_n;
                bfloat16_t *a1 = block_p + (offset_m + 16) * tiled_n;
                for (int n_tiled_block = 0; n_tiled_block < tiled_blocks; ++n_tiled_block) {
                    const bfloat16_t *b =
                        K + block_table[n_block + n_tiled_block] * params.kcache->stride(0) + offset_n;
                    for (int k = 0; k < 64 / 2; ++k) {
                        svbfloat16_t vb0 = SVLDNT1_BF16(ptrue, b);
                        svbfloat16_t vb1 = SVLDNT1_INDEX_BF16(ptrue, b, 576);
                        svbfloat16_t vb2 = svzip1_bf16(vb0, vb1);
                        svbfloat16_t va0 = svld1_bf16(ptrue, a0);
                        svmopa_za32_bf16_m(0, ptrue, ptrue, va0, vb2);
                        svbfloat16_t vb3 = svzip2_bf16(vb0, vb1);
                        svmopa_za32_bf16_m(1, ptrue, ptrue, va0, vb3);
                        svbfloat16_t va1 = svld1_bf16(ptrue, a1);
                        svmopa_za32_bf16_m(2, ptrue, ptrue, va1, vb2);
                        svmopa_za32_bf16_m(3, ptrue, ptrue, va1, vb3);
                        a0 += 32;
                        a1 += 32;
                        b += 576 * 2;
                    }
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
    }
    for (int i = 0; i < q_block_size; ++i) {
        for (int j = 0; j < 512; j += 32) {
            svst1(ptrue, out_o + i * 512 + j, svld1(ptrue, block_o + j * q_block_size + i * 32));
            svst1(ptrue, out_o + i * 512 + j + 16, svld1(ptrue, block_o + j * q_block_size + i * 32 + 16));
        }
    }
    MATRIX_OFF();

    const bool is_no_split = begin_kv_token_idx == 0 && end_kv_token_idx == seqlen_kv;

    if (is_no_split) {
        store<DenseDecodeParams, KernelTraits, true>(
            params, req_idx, q_token_idx, q_block_idx, split_idx, out_o, row_max, row_sum);
    } else {
        store<DenseDecodeParams, KernelTraits, false>(
            params, req_idx, q_token_idx, q_block_idx, split_idx, out_o, row_max, row_sum);
    }
}

} // namespace flash_mla

} // namespace kutacc