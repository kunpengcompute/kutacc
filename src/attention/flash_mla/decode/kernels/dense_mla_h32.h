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
void dense_mla_h32_kernel(
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

    static_assert(q_block_size == 32);
    static_assert(kv_block_size == 64);
    static_assert(head_dim == 576);
    static_assert(head_dim_v == 512);

    const int n_block_min = begin_kv_token_idx / kv_block_size;
    const int n_block_max = ceil_div<int64_t>(end_kv_token_idx, kv_block_size);

    const int *block_table = &params.block_table->index({req_idx, 0});
    const int64_t row_offset_q = req_idx * params.q->stride(0) + q_block_idx * q_block_size * params.q->stride(2);
    static_assert(
        (32 * 576 + 32 * 576) * sizeof(bfloat16_t) + (32 * 128 + 32 * 512 + 32 + 32 + 32) * sizeof(float) <=
        MAX_THREAD_BUFFER_BYTES);
    char *thread_ptr = (char *)params.threads_buffer + get_thread_id() * MAX_THREAD_BUFFER_BYTES;
    auto alloc_buffer = [](char *&pool_ptr, int size) {
        pool_ptr += size;
        return pool_ptr - size;
    };
    bfloat16_t *block_q = (bfloat16_t *)alloc_buffer(thread_ptr, 32 * 576 * sizeof(bfloat16_t));
    bfloat16_t *block_k = (bfloat16_t *)alloc_buffer(thread_ptr, 32 * 576 * sizeof(bfloat16_t));
    float *block_s = (float *)alloc_buffer(thread_ptr, 32 * 128 * sizeof(float));
    bfloat16_t *block_p = (bfloat16_t *)block_s;
    float *block_o = (float *)alloc_buffer(thread_ptr, 32 * 512 * sizeof(float));
    float *row_max = (float *)alloc_buffer(thread_ptr, 32 * sizeof(float));
    float *row_sum = (float *)alloc_buffer(thread_ptr, 32 * sizeof(float));
    float *scale_o = (float *)alloc_buffer(thread_ptr, 32 * sizeof(float));
    float *out_o = (float *)block_q;
    const bfloat16_t *Q = reinterpret_cast<bfloat16_t *>(params.q->data_ptr());
    const bfloat16_t *K = reinterpret_cast<bfloat16_t *>(params.kcache->data_ptr());
    MATRIX_ON();
    const svbool_t ptrue = svptrue_b16();
    if (likely((q_block_idx + 1) * 32 <= params.num_heads_q)) {
        for (int j = 0; j < 576; j += 32) {
            for (int head_idx = 0; head_idx < 32; head_idx += 16) {
                auto q0 = Q + row_offset_q + head_idx * 576 + j;
                for (int i = 0; i < 16; ++i) {
                    svld1_hor_za32(0, i, ptrue, q0 + i * 576);
                }
                auto q1 = block_q + j * 32 + head_idx * 2;
                for (int i = 0; i < 16; ++i) {
                    svst1_ver_za32(0, i, ptrue, q1 + i * 32 * 2);
                }
            }
        }
    } else {
        for (int j = 0; j < 576; j += 2) {
            for (int i = 0; i < 32 && i < params.num_heads_q - q_block_idx * 32; ++i) {
                block_q[j * 32 + 2 * i] = Q[row_offset_q + i * 576 + j];
                block_q[j * 32 + 2 * i + 1] = Q[row_offset_q + i * 576 + j + 1];
            }
        }
    }
    const bool is_odd = (n_block_max - n_block_min) & 1;
    const float &softmax_scale_log2 = params.softmax_scale_log2;
    auto get_block_id = [=](int n_block) {
        if (n_block >= n_block_max - 2) {
            return n_block;
        }
        constexpr int g = 2;
        int l = n_block_min;
        int r = n_block_max - 2;
        int p = l + (r - l) * (q_block_idx / g) / ((params.num_heads_q + 32 * g - 1) / (32 * g));
        if constexpr (is_causal) {
            if (params.num_heads_q / params.casual_group_size - 1 > 65 && (p - l) % 2) {
                p += 1;
            }
        }
        if (n_block - l < r - p) {
            return n_block + (p - l);
        } else {
            return n_block - (r - p);
        }
    };
    for (int n_block = n_block_min; n_block < n_block_max;) {
        const bool is_first_step = n_block == n_block_min;
        const int tiled_n = is_first_step && is_odd ? 64 : 128;
        const bool is_last_step = n_block + tiled_n / 64 == n_block_max;
        int block_id = get_block_id(n_block);
        int len_n = seqlen_kv - block_id * 64;
        if (len_n > tiled_n) {
            len_n = tiled_n;
        }
        const bfloat16_t *kv_ptr[4];
        kv_ptr[0] = K + block_table[block_id] * params.kcache->stride(0);
        kv_ptr[1] =
            n_block + 1 < n_block_max ? K + block_table[get_block_id(n_block + 1)] * params.kcache->stride(0) : nullptr;
        kv_ptr[2] =
            n_block + 2 < n_block_max ? K + block_table[get_block_id(n_block + 2)] * params.kcache->stride(0) : nullptr;
        kv_ptr[3] =
            n_block + 3 < n_block_max ? K + block_table[get_block_id(n_block + 3)] * params.kcache->stride(0) : nullptr;
        const bfloat16_t *prefetch_ptr = nullptr;
        bool is_odd_first = is_first_step && is_odd;
        if (is_odd_first) {
            prefetch_ptr = kv_ptr[1];
        } else {
            prefetch_ptr = kv_ptr[1] + 16 * 576;
        }
        for (int offset_n = 0; offset_n < len_n; offset_n += 32) {
            const bfloat16_t *cur_kv = kv_ptr[offset_n / 64] + offset_n % 64 * 576;
            for (int j = 0; j < 576; j += 32) {
                for (int i0 = 0; i0 < 32; i0 += 16) {
                    const bfloat16_t *k0 = cur_kv + i0 * 576 + j;
                    for (int i = 0; i < 16; ++i) {
                        svld1_hor_za32(0, i, ptrue, k0 + i * 576);
                    }
                    auto k1 = block_k + j * 32 + i0 * 2;
                    for (int i = 0; i < 16; ++i) {
                        svst1_ver_za32(0, i, ptrue, k1 + i * 64);
                    }
                }
            }
            if (offset_n == 96) {
                prefetch_ptr = kv_ptr[2];
            }
            svzero_za();
            bfloat16_t *data_atmp = block_q;
            bfloat16_t *data_btmp = block_k;
            for (int i = 0; i < 576 / 2; ++i) {
                constexpr int prf_stride = 64 * 6;
                svbfloat16_t va0 = svld1(ptrue, data_atmp);
                svbfloat16_t vb0 = SVLDNT1_BF16(ptrue, data_btmp);
                svprfh(ptrue, data_atmp + prf_stride, SV_PLDL1STRM);
                svmopa_za32_bf16_m(0, ptrue, ptrue, va0, vb0);
                svbfloat16_t vb1 = SVLDNT1_INDEX_BF16(ptrue, data_btmp, 32);
                svprfh(ptrue, data_atmp + prf_stride + 32, SV_PLDL1STRM);
                svmopa_za32_bf16_m(1, ptrue, ptrue, va0, vb1);
                svbfloat16_t va1 = svld1(ptrue, data_atmp + 32);
                svprfh(ptrue, data_btmp + prf_stride, SV_PLDL1STRM);
                svmopa_za32_bf16_m(2, ptrue, ptrue, va1, vb0);
                svprfh(ptrue, data_btmp + prf_stride + 32, SV_PLDL1STRM);
                svmopa_za32_bf16_m(3, ptrue, ptrue, va1, vb1);
                data_atmp += 64;
                data_btmp += 64;
                if (likely(prefetch_ptr != nullptr)) {
                    svprfh(ptrue, prefetch_ptr, SV_PLDL2KEEP);
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
        if (unlikely(is_last_step)) {
            prefetch_ptr = nullptr;
        }
        for (int offset_m = 0; offset_m < 32; offset_m += 16) {
            float *block_s_begin = block_s + offset_m * tiled_n;
            bfloat16_t *block_p_begin = block_p + offset_m * 2;
            svfloat32_t sve_row_max = unlikely(is_first_step) ? svdup_f32(-INFINITY) : svld1(ptrue, row_max + offset_m);
            svfloat32_t sve_max_prev = sve_row_max;
            int end = len_n;
            int causal_row_start = params.num_heads_q - 1 - (q_block_idx * q_block_size + offset_m) -
                (params.casual_group_size - 1) - (seqlen_kv - 1 - block_id * kv_block_size) * params.casual_group_size;
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
                svprfw(ptrue, block_s_begin + i * 2 * 16 + prf_stride, SV_PLDL1STRM);
                svprfw(ptrue, block_s_begin + (i * 2 + 1) * 16 + prf_stride, SV_PLDL1STRM);
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
                if (likely(prefetch_ptr != nullptr)) {
                    svprfh(ptrue, prefetch_ptr, SV_PLDL2KEEP);
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
            for (int j = 0; j < len_n; j += 64) {
                if (unlikely(!is_odd_first && kv_ptr[2] != nullptr && prefetch_ptr == kv_ptr[2] + 64 * 576)) {
                    prefetch_ptr = kv_ptr[3];
                }
                const bfloat16_t *data_btmp = kv_ptr[j / 64] + offset_n;
                for (int i = 0; i < 64; i += 2) {
                    svbfloat16_t vb0 = SVLDNT1_BF16(ptrue, data_btmp);
                    svbfloat16_t vb1 = SVLDNT1_INDEX_BF16(ptrue, data_btmp, 576);
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
                        svprfh(ptrue, prefetch_ptr, SV_PLDL2KEEP);
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
                svstnt1(ptrue, matd0, sve0);
                svstnt1(ptrue, matd1, sve1);
                svstnt1(ptrue, matd2, sve2);
                svstnt1(ptrue, matd3, sve3);
                matd0 += 32;
                matd1 += 32;
                matd2 += 32;
                matd3 += 32;
            }
        }
        n_block += tiled_n / 64;
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