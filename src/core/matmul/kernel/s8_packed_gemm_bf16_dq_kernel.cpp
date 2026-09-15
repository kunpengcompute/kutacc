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
#include <cmath>
#include <algorithm>
#include <arm_sve.h>
#include <arm_sme.h>

#include "kernel.h"
#include "kernel_common.h"

namespace kutacc {

inline void preload_L2(int bm, int em, int k_num, int k, int8_t *a)
{
    for (int mi = bm; mi + 32 <= em; mi += 32) {
        for (int ki = 0; ki < k_num / 4; ki++) {
            svprfb(svptrue_b8(), a + mi * k + ki * 64, SV_PLDL2KEEP);
            svprfb(svptrue_b8(), a + (mi + 16) * k + ki * 64, SV_PLDL2KEEP);
        }
    }
}

inline void s8_packed_gemm_bf16_dq_M_lt_128_N_K_eq_2048_macro_kernel(int block_index_m, int block_end_m,
    int block_index_n, int block_end_n, int k, int block_index_k, int block_stride_k, int ldc, int8_t *a, int8_t *b,
    bfloat16_t *c, const float *rscales, const float *cscales) __arm_inout("za") __arm_streaming
{
    // check (block_end_m - block_index_m) % 64 == 0
    // check (block_end_n - block_index_n) % 32 == 0
    constexpr int svls = 16;
    constexpr int n_stride = 2 * svls;
    constexpr int m_stride = 2 * svls;
    const int igemm_prefetch_k_stride = (block_stride_k / 4) / 2 / 2; // 128
    int block_prefetch_k_start = block_stride_k / 4 * ((block_end_m - block_index_m + svls - 1) / 32);
    int block_prefetch_k_stride = std::max(block_stride_k - block_prefetch_k_start, 0);
    preload_L2(block_index_n, block_index_n + n_stride, block_prefetch_k_start, k, b + block_index_k * svls);
    if (block_stride_k % 448) {
        preload_L2(block_index_m, block_end_m, block_stride_k, k, a + block_index_k * svls);
    }
    int ni = block_index_n;
    for (; ni + n_stride <= block_end_n; ni += n_stride) {
        preload_L2(ni, ni + n_stride, block_prefetch_k_stride, k, b + (block_prefetch_k_start + block_index_k) * svls);
        int mi = block_index_m;
        for (; mi + svls < block_end_m; mi += m_stride) {
            int64_t m_num = std::min(64, (block_end_m - mi - svls) * 4);
            svzero_za();
            s8_packed_gemm_2VL_2VL_prf_no_insert(a + mi * k + block_index_k * svls, b + ni * k + block_index_k * svls,
                k, block_stride_k, (mi - block_index_m) / m_stride, igemm_prefetch_k_stride, m_num);
            s8_packed_gemm_bf16_dq_2VL_2VL_masked(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num);
        }
        for (; mi < block_end_m; mi += svls) {
            int m_num = std::min(64, (block_end_m - mi) * 4);
            svzero_za();
            s8_packed_gemm_2VL_1VL(
                a + mi * k + block_index_k * m_num / 4, b + ni * k + block_index_k * svls, k, block_stride_k, m_num);
            s8_packed_gemm_bf16_dq_2VL_1VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num);
        }
    }
    for (; ni < block_end_n; ni += svls) {
        int64_t n_num = std::min(64, (block_end_n - ni) * 4);
        int mi = block_index_m;
        for (; mi + m_stride <= block_end_m; mi += m_stride) {
            svzero_za();
            s8_packed_gemm_1VL_2VL(
                a + mi * k + block_index_k * svls, b + ni * k + block_index_k * n_num / 4, k, block_stride_k, n_num);
            s8_packed_gemm_bf16_dq_1VL_2VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, n_num);
        }
        for (; mi < block_end_m; mi += svls) {
            int m_num = std::min(64, (block_end_m - mi) * 4);
            svzero_za();
            s8_packed_gemm_1VL_1VL(a + mi * k + block_index_k * m_num / 4, b + ni * k + block_index_k * n_num / 4,
                block_stride_k, m_num, n_num);
            s8_packed_gemm_bf16_dq_1VL_1VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num, n_num);
        }
    }
}

inline void s8_packed_gemm_bf16_dq_macro_kernel(int bm, int em, int bn, int en, int k, int k_offset, int k_num,
    int ldc, int8_t *a, int8_t *b, bfloat16_t *c, const float *rscales,
    const float *cscales) __arm_inout("za") __arm_streaming
{
    int prefetch_dis = (k == 128 ? 128 : (k == 4096 ? 48 : 16));
    if (k >= 1024) {
        for (int ni = bn; ni < en;) {
            if (ni + 32 <= en) {
                int mi = bm;
                for (; mi + 32 <= em; mi += 32) {
                    svzero_za();
                    s8_packed_gemm_2VL_2VL(
                        a + mi * k + k_offset * 16, b + ni * k + k_offset * 16, k, k_num, prefetch_dis);
                    s8_packed_gemm_bf16_dq_2VL_2VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni);
                }
                for (; mi < em; mi += 16) {
                    int m_num = std::min(64, (em - mi) * 4);
                    svzero_za();
                    s8_packed_gemm_2VL_1VL(
                        a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * 16, k, k_num, m_num);
                    s8_packed_gemm_bf16_dq_2VL_1VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num);
                }
                ni += 32;
            } else {
                int n_num = std::min(64, (en - ni) * 4), mi = bm;
                for (; mi + 64 <= em; mi += 64) {
                    svzero_za();
                    s8_packed_gemm_1VL_4VL(
                        a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 4, k, k_num, n_num);
                    s8_packed_gemm_bf16_dq_1VL_4VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, n_num);
                }
                for (; mi < em; mi += 16) {
                    int m_num = std::min(64, (em - mi) * 4);
                    svzero_za();
                    s8_packed_gemm_1VL_1VL(
                        a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * n_num / 4, k_num, m_num, n_num);
                    s8_packed_gemm_bf16_dq_1VL_1VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num, n_num);
                }
                ni += 16;
            }
        }
    } else {
        for (int mi = bm; mi < em;) {
            if (mi + 32 <= em) {
                int ni = bn;
                for (; ni + 32 <= en; ni += 32) {
                    svzero_za();
                    s8_packed_gemm_2VL_2VL(
                        a + mi * k + k_offset * 16, b + ni * k + k_offset * 16, k, k_num, prefetch_dis);
                    s8_packed_gemm_bf16_dq_2VL_2VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni);
                }
                for (; ni < en; ni += 16) {
                    int n_num = std::min(64, (en - ni) * 4);
                    svzero_za();
                    s8_packed_gemm_1VL_1VL(
                        a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 4, k_num, 64, n_num);
                    s8_packed_gemm_bf16_dq_1VL_1VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, 64, n_num);
                    svzero_za();
                    s8_packed_gemm_1VL_1VL(
                        a + (mi + 16) * k + k_offset * 16, b + ni * k + k_offset * n_num / 4, k_num, 64, n_num);
                    s8_packed_gemm_bf16_dq_1VL_1VL(
                        rscales + ni, cscales + (mi + 16), ldc, c + (mi + 16) * ldc + ni, 64, n_num);
                }
                mi += 32;
            } else {
                int m_num = std::min(64, (em - mi) * 4), ni = bn;
                for (; ni + 32 <= en; ni += 32) {
                    svzero_za();
                    s8_packed_gemm_2VL_1VL(
                        a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * 16, k, k_num, m_num);
                    s8_packed_gemm_bf16_dq_2VL_1VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num);
                }
                for (; ni < en; ni += 16) {
                    int n_num = std::min(64, (en - ni) * 4);
                    svzero_za();
                    s8_packed_gemm_1VL_1VL(
                        a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * n_num / 4, k_num, m_num, n_num);
                    s8_packed_gemm_bf16_dq_1VL_1VL(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num, n_num);
                }
                mi += 16;
            }
        }
    }
}

__arm_new("za") void s8_packed_gemm_bf16_dq_alpha1_beta0(int m, int n, int ldc, int k, int8_t *a, int8_t *b,
    bfloat16_t *c, const float *rscales, const float *cscales, int macro_kernel_m, int macro_kernel_n,
    int macro_kernel_k) __arm_streaming
{
    if (macro_kernel_k < k)
        macro_kernel_k = k;
    for (int bm = 0; bm < m; bm += macro_kernel_m) {
        int em = std::min(bm + macro_kernel_m, m);
        for (int k_offset = 0; k_offset < k; k_offset += macro_kernel_k) {
            int k_num = std::min(macro_kernel_k, k - k_offset);
            preload_L2(bm, em, k_num, k, a + k_offset * 16);
            for (int bn = 0; bn < n; bn += macro_kernel_n) {
                int en = std::min(bn + macro_kernel_n, n);
                s8_packed_gemm_bf16_dq_macro_kernel(bm, em, bn, en, k, k_offset, k_num, ldc, a, b, c, rscales, cscales);
            }
        }
    }
}

__arm_new("za") void s8_packed_gemm_bf16_dq_alpha1_beta0_m_lt_128_k_2048(int m, int n, int ldc, int k, int8_t *a,
    int8_t *b, bfloat16_t *c, const float *rscales, const float *cscales, int macro_kernel_m, int macro_kernel_n,
    int macro_kernel_k) __arm_streaming
{
    for (int block_index_m = 0; block_index_m < m; block_index_m += macro_kernel_m) {
        int block_end_m = std::min(block_index_m + macro_kernel_m, m);
        for (int block_index_k = 0; block_index_k < k; block_index_k += macro_kernel_k) {
            int block_stride_k = std::min(macro_kernel_k, k - block_index_k);
            for (int block_index_n = 0; block_index_n < n; block_index_n += macro_kernel_n) {
                int block_end_n = std::min(block_index_n + macro_kernel_n, n);
                s8_packed_gemm_bf16_dq_M_lt_128_N_K_eq_2048_macro_kernel(block_index_m, block_end_m, block_index_n,
                    block_end_n, k, block_index_k, block_stride_k, ldc, a, b, c, rscales, cscales);
            }
        }
    }
}

inline void preload_L2_1VL_LD(int m, int n, int8_t *src, int lds, int8_t *dst, int ldd)
{
    for (int mi = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi);
        int dst_off = mi * ldd;
        for (int ni = 0; ni < n; ni += 64) {
#pragma unroll(16)
            for (int i = 0; i < 16; i++) {
                svprfb(svptrue_b8(), src + (mi + i) * lds + ni, svprfop::SV_PLDL2KEEP);
                svprfb(svptrue_b8(), dst + dst_off + (ni + i * 4) * m_num, svprfop::SV_PSTL2KEEP);
            }
        }
    }
}

__arm_new("za") void s8_gemm_pack_kernel(int m, int n, int8_t *src, int lds, int8_t *dst, int ldd) __arm_streaming
{
    preload_L2_1VL_LD(m, n, src, lds, dst, ldd);
    auto prefetch_dis = 64;
    for (int mi = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi);
        int dst_off = mi * ldd;
        svbool_t pg_m = svwhilelt_b32(mi, m);
        for (int ni = 0; ni < n; ni += 64) {
            svzero_za();
            svbool_t pg_n = svwhilelt_b32(0, (n - ni) / 4);
#pragma unroll(8)
            for (int i = 0; i < 16 && mi + i < m; i++) {
                svprfb(svptrue_b8(), src + (mi + i) * lds + ni + prefetch_dis, svprfop::SV_PLDL1KEEP);
                svld1_hor_za32(0, i, pg_n, src + (mi + i) * lds + ni);
            }
#pragma unroll(8)
            for (int i = 0; i < 16 && ni + i * 4 < n; i++) {
                svprfb(svptrue_b8(), dst + dst_off + (ni + prefetch_dis + i * 4) * m_num, svprfop::SV_PSTL1KEEP);
                svst1_ver_za32(0, i, pg_m, dst + dst_off + (ni + i * 4) * m_num);
            }
        }
    }
}

__arm_new("za") void s8_gemm_pack_with_idx_kernel(int m, int n, int8_t *src, int lds, int m_off, int *idx, int8_t *dst,
    int ldd) __arm_streaming
{
    auto prefetch_dis = 128;
    for (int mi = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi);
        int dst_off = mi * ldd;
        svbool_t pg_m = svwhilelt_b32(mi, m);
        for (int ni = 0; ni < n; ni += 64) {
            svzero_za();
            svbool_t pg_n = svwhilelt_b32(0, (n - ni) / 4);
#pragma unroll(16)
            for (int i = 0; i < 16 && mi + i < m; i++) {
                svprfb(svptrue_b8(), src + idx[m_off + mi + i] * lds + ni + prefetch_dis, svprfop::SV_PLDL2KEEP);
                svld1_hor_za32(0, i, pg_n, src + idx[m_off + mi + i] * lds + ni);
            }
#pragma unroll(16)
            for (int i = 0; i < 16 && ni + i * 4 < n; i++) {
                // dst + dst_off + (ni + i * 4) * m_num + prefetch_dis / 64 * 16 * 4 * m_num
                svprfb(
                    svptrue_b8(), dst + dst_off + (ni + i * 4) * m_num + prefetch_dis * m_num, svprfop::SV_PSTL2KEEP);
                svst1_ver_za32(0, i, pg_m, dst + dst_off + (ni + i * 4) * m_num);
            }
        }
    }
}

} // namespace kutacc
