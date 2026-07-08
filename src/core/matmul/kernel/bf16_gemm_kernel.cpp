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
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>

#include "kernel.h"
#include "kernel_common.h"
#include "utils/check.h"

namespace kutacc {

inline void bf16_packed_gemm_macro_kernel(int bm, int em, int bn, int en, int k, int k_offset, int k_num, int ldc,
    bfloat16_t *a, bfloat16_t *b, bfloat16_t *c) __arm_inout("za") __arm_streaming
{
    for (int mi = bm; mi < em;) {
        if (mi + 64 <= em) {
            int ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bf16_packed_gemm_4VL_1VL(
                    a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 2, k, k_num, n_num);
                bf16_packed_gemm_4VL_1VL_save(ldc, c + mi * ldc + ni, n_num);
            }
            mi += 64;
        } else {
            int m_num = std::min(32, (em - mi) * 2), ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bf16_packed_gemm_1VL_1VL(
                    a + mi * k + k_offset * m_num / 2, b + ni * k + k_offset * n_num / 2, k, k_num, m_num, n_num);
                bf16_packed_gemm_1VL_1VL_save(ldc, c + mi * ldc + ni, m_num, n_num);
            }
            mi += 16;
        }
    }
}

template <bool add, int pred_k, bool usebias>
inline void bf16_packed_gemm_macro_kernel_64(int bm, int em, int bn, int en, int k, int k_offset, int k_num, int ldc,
    bfloat16_t *a, bfloat16_t *b, bfloat16_t *c, float *sub_bias, bool row_bias) __arm_inout("za") __arm_streaming
{
    KUTACC_CHECK(bm + 64 == em, "");
    KUTACC_CHECK(k_num % 2 == 0, "");
    for (int ni = bn; ni < en; ni += 32) {
        int n_num0 = std::min(16, en - ni);
        int n_num1 = std::min(16, en - ni - 16);
        for (int mi = bm; mi < em; mi += 32) {
            svzero_za();
            const bfloat16_t *a0 = a + mi * k + k_offset * 16;
            const bfloat16_t *a1 = a + (mi + 16) * k + k_offset * 16;
            const bfloat16_t *b0 = b + ni * k + k_offset * n_num0;
            const bfloat16_t *b1 = b + (ni + 16) * k + k_offset * n_num1;
            svbool_t ptrue = svptrue_b16();
            svbool_t pb0 = svwhilelt_b16(0, n_num0 * 2);
            svbool_t pb1 = svwhilelt_b16(0, n_num1 * 2);
            const bfloat16_t *prf_ptr = b + (ni + 32 + (mi - bm) / 2) * k + k_offset * 16;
            for (int ki = 0; ki < (pred_k < 0 ? k_num : pred_k); ki += 2) {
                svprfh(ptrue, prf_ptr, SV_PLDL2KEEP);
                prf_ptr += 32;
                svbfloat16_t va0 = svld1(ptrue, a0);
                svbfloat16_t vb0 = svld1(pb0, b0);
                svmopa_za32_bf16_m(0, ptrue, pb0, va0, vb0);
                svbfloat16_t vb1 = svld1(pb1, b1);
                svmopa_za32_bf16_m(1, ptrue, pb1, va0, vb1);
                svbfloat16_t va1 = svld1(ptrue, a1);
                svmopa_za32_bf16_m(2, ptrue, pb0, va1, vb0);
                svmopa_za32_bf16_m(3, ptrue, pb1, va1, vb1);
                a0 += 32;
                a1 += 32;
                b0 += 2 * n_num0;
                b1 += 2 * n_num1;
            }
            bfloat16_t *c0 = c + mi * ldc + ni;
            bfloat16_t *c1 = c + (mi + 16) * ldc + ni;
            svbool_t pc = svwhilelt_b16(0, std::min(32, en - ni));
            svbfloat16_t vz = svdup_bf16(0);
            for (int i = 0; i < 16; i++) {
                svfloat32_t vc0 = svread_hor_za32_f32_m({}, ptrue, 0, i);
                svfloat32_t vc1 = svread_hor_za32_f32_m({}, ptrue, 1, i);
                svfloat32_t vc2 = svread_hor_za32_f32_m({}, ptrue, 2, i);
                svfloat32_t vc3 = svread_hor_za32_f32_m({}, ptrue, 3, i);
                if constexpr (add) {
                    svbfloat16_t vt0 = svld1(ptrue, c0);
                    svbfloat16_t vt1 = svld1(ptrue, c1);
                    svfloat32_t vd0 = svreinterpret_f32(svzip1(vz, vt0));
                    svfloat32_t vd1 = svreinterpret_f32(svzip2(vz, vt0));
                    svfloat32_t vd2 = svreinterpret_f32(svzip1(vz, vt1));
                    svfloat32_t vd3 = svreinterpret_f32(svzip2(vz, vt1));
                    vc0 = svadd_x(ptrue, vc0, vd0);
                    vc1 = svadd_x(ptrue, vc1, vd1);
                    vc2 = svadd_x(ptrue, vc2, vd2);
                    vc3 = svadd_x(ptrue, vc3, vd3);
                }
                if constexpr (usebias) {
                    if (row_bias) {
                        svbool_t pg_bias = svwhilelt_b32(0, 16);
                        svfloat32_t bias_vec0 = svld1_f32(pg_bias, sub_bias + ni);
                        svfloat32_t bias_vec1 = svld1_f32(pg_bias, sub_bias + ni + 16);
                        vc0 = svadd_f32_z(pg_bias, vc0, bias_vec0);
                        vc1 = svadd_f32_z(pg_bias, vc1, bias_vec1);
                        vc2 = svadd_f32_z(pg_bias, vc2, bias_vec0);
                        vc3 = svadd_f32_z(pg_bias, vc3, bias_vec1);
                    } else {
                        int row0 = mi + i;
                        int row1 = mi + i + 16;
                        int row2 = mi + i + 32;
                        int row3 = mi + i + 48;
                        svbool_t pg_f32 = svptrue_b32();
                        svfloat32_t bias_vec0 = svdup_f32(sub_bias[row0]);
                        svfloat32_t bias_vec1 = svdup_f32(sub_bias[row1]);
                        svfloat32_t bias_vec2 = svdup_f32(sub_bias[row2]);
                        svfloat32_t bias_vec3 = svdup_f32(sub_bias[row3]);
                        vc0 = svadd_f32_z(pg_f32, vc0, bias_vec0);
                        vc1 = svadd_f32_z(pg_f32, vc1, bias_vec1);
                        vc2 = svadd_f32_z(pg_f32, vc2, bias_vec2);
                        vc3 = svadd_f32_z(pg_f32, vc3, bias_vec3);
                    }
                }
                svstnt1(pc, c0, svuzp1(svcvt_bf16_x(ptrue, vc0), svcvt_bf16_x(ptrue, vc1)));
                svstnt1(pc, c1, svuzp1(svcvt_bf16_x(ptrue, vc2), svcvt_bf16_x(ptrue, vc3)));
                c0 += ldc;
                c1 += ldc;
            }
        }
    }
}

template <bool add, int pred_k, bool hasbias>
inline void bf16_packed_gemm_macro_kernel_16(int bm, int em, int bn, int en, int k, int k_offset, int k_num, int ldc,
    bfloat16_t *a, bfloat16_t *b, bfloat16_t *c, float *sub_bias, bool row_bias) __arm_inout("za") __arm_streaming
{
    KUTACC_CHECK(em - bm <= 16, "");
    KUTACC_CHECK(k_num % 2 == 0, "");
    int m_num = em - bm;
    for (int ni = bn; ni < en; ni += 32) {
        int n_num0 = std::min(16, en - ni);
        int n_num1 = std::min(16, en - ni - 16);
        svzero_za();
        const bfloat16_t *a0 = a + bm * k + k_offset * m_num;
        const bfloat16_t *b0 = b + ni * k + k_offset * n_num0;
        const bfloat16_t *b1 = b + (ni + 16) * k + k_offset * n_num1;
        svbool_t ptrue = svptrue_b16();
        svbool_t pa = svwhilelt_b16(0, m_num * 2);
        svbool_t pb0 = svwhilelt_b16(0, n_num0 * 2);
        svbool_t pb1 = svwhilelt_b16(0, n_num1 * 2);
        const bfloat16_t *prf_ptr0 = b + (ni + 32) * k + k_offset * 16;
        const bfloat16_t *prf_ptr1 = b + (ni + 48) * k + k_offset * 16;
        for (int ki = 0; ki < (pred_k < 0 ? k_num : pred_k); ki += 2) {
            if (__builtin_expect(ni + 64 <= en, 1)) {
                svprfh(ptrue, prf_ptr0, SV_PLDL2KEEP);
                prf_ptr0 += 32;
                svprfh(ptrue, prf_ptr1, SV_PLDL2KEEP);
                prf_ptr1 += 32;
            }
            svbfloat16_t va0 = svld1(pa, a0);
            svbfloat16_t vb0 = svldnt1(pb0, b0);
            svmopa_za32_bf16_m(0, pa, pb0, va0, vb0);
            svbfloat16_t vb1 = svldnt1(pb1, b1);
            svmopa_za32_bf16_m(1, pa, pb1, va0, vb1);
            a0 += 2 * m_num;
            b0 += 2 * n_num0;
            b1 += 2 * n_num1;
        }
        bfloat16_t *c0 = c + bm * ldc + ni;
        svbool_t pc = svwhilelt_b16(0, std::min(32, en - ni));
        svbfloat16_t vz = svdup_bf16(0);
        for (int i = 0; i < m_num; i++) {
            svfloat32_t vc0 = svread_hor_za32_f32_m({}, ptrue, 0, i);
            svfloat32_t vc1 = svread_hor_za32_f32_m({}, ptrue, 1, i);
            if constexpr (add) {
                svbfloat16_t vt0 = svld1(pc, c0);
                svfloat32_t vd0 = svreinterpret_f32(svzip1(vz, vt0));
                svfloat32_t vd1 = svreinterpret_f32(svzip2(vz, vt0));
                vc0 = svadd_x(ptrue, vc0, vd0);
                vc1 = svadd_x(ptrue, vc1, vd1);
            }
            if constexpr (hasbias) {
                if (row_bias) {
                    svbool_t pg_bias = svwhilelt_b32(0, 16); 
                    svfloat32_t bias_vec0 = svld1_f32(pg_bias, sub_bias + ni);
                    svfloat32_t bias_vec1 = svld1_f32(pg_bias, sub_bias + ni + 16);
                    vc0 = svadd_f32_z(pg_bias, vc0, bias_vec0);
                    vc1 = svadd_f32_z(pg_bias, vc1, bias_vec1);
                } else {
                    int row = bm + i;
                    svbool_t pg_f32 = svptrue_b32();
                    svfloat32_t bias_vec = svdup_f32(sub_bias[row]);
                    vc0 = svadd_f32_z(pg_f32, vc0, bias_vec);
                    vc1 = svadd_f32_z(pg_f32, vc1, bias_vec);
                }
            }
            svstnt1(pc, c0, svuzp1(svcvt_bf16_x(ptrue, vc0), svcvt_bf16_x(ptrue, vc1)));
            c0 += ldc;
        }
    }
}

inline void bf16_packed_gemm_prf_opt_kernel(int m, int n, int ldc, int k, bfloat16_t *a, bfloat16_t *b,
    bfloat16_t *c, float *sub_bias, bool row_bias) __arm_inout("za") __arm_streaming
{
    int bm = 0;
    for (; bm + 64 <= m; bm += 64) {
        constexpr int k_block = 7168 / 8;
        for (int k_offset = 0; k_offset < k; k_offset += k_block) {
            int k_num = std::min(k_block, k - k_offset);
            bool use_k_block = (k_block == k_num);
            bool has_k_off = (k_offset > 0);
            bool has_bias = (sub_bias != nullptr);
            auto kernel = has_k_off
                ? (use_k_block
                    ? (has_bias ? bf16_packed_gemm_macro_kernel_64<true, k_block, true>
                                    : bf16_packed_gemm_macro_kernel_64<true, k_block, false>)
                    : (has_bias ? bf16_packed_gemm_macro_kernel_64<true, -1, true>
                                    : bf16_packed_gemm_macro_kernel_64<true, -1, false>))
                : (use_k_block
                    ? (has_bias ? bf16_packed_gemm_macro_kernel_64<false, k_block, true>
                                    : bf16_packed_gemm_macro_kernel_64<false, k_block, false>)
                    : (has_bias ? bf16_packed_gemm_macro_kernel_64<false, -1, true>
                                    : bf16_packed_gemm_macro_kernel_64<false, -1, false>));
            kernel(bm, bm + 64, 0, n, k, k_offset, k_num, ldc, a, b, c, sub_bias, row_bias);
        }
    }
    for (; bm < m; bm += 16) {
        constexpr int k_block = 7168 / 32;
        int em = std::min(bm + 16, m);
        for (int k_offset = 0; k_offset < k; k_offset += k_block) {
            int k_num = std::min(k_block, k - k_offset);
            bool use_k_block = (k_block == k_num);
            bool has_k_off = (k_offset > 0);
            bool has_bias = (sub_bias != nullptr);
            auto kernel = has_k_off
                ? (use_k_block
                    ? (has_bias ? bf16_packed_gemm_macro_kernel_16<true, k_block, true>
                                    : bf16_packed_gemm_macro_kernel_16<true, k_block, false>)
                    : (has_bias ? bf16_packed_gemm_macro_kernel_16<true, -1, true>
                                    : bf16_packed_gemm_macro_kernel_16<true, -1, false>))
                : (use_k_block
                    ? (has_bias ? bf16_packed_gemm_macro_kernel_16<false, k_block, true>
                                    : bf16_packed_gemm_macro_kernel_16<false, k_block, false>)
                    : (has_bias ? bf16_packed_gemm_macro_kernel_16<false, -1, true>
                                    : bf16_packed_gemm_macro_kernel_16<false, -1, false>));

            kernel(bm, em, 0, n, k, k_offset, k_num, ldc, a, b, c, sub_bias, row_bias);
        }
    }
}

inline void preload_L1(int bm, int em, int k_num, int k, bfloat16_t *a)
{
    for (int mi = bm; mi + 32 <= em; mi += 32)
        for (int ki = 0; ki < k_num / 2; ki++) {
            svprfh(svptrue_b16(), a + mi * k + ki * 32, SV_PLDL1KEEP);
            svprfh(svptrue_b16(), a + (mi + 16) * k + ki * 32, SV_PLDL1KEEP);
        }
}

inline void preload_L2(int bm, int em, int k_num, int k, bfloat16_t *a)
{
    for (int mi = bm; mi + 32 <= em; mi += 32)
        for (int ki = 0; ki < k_num / 2; ki++) {
            svprfh(svptrue_b16(), a + mi * k + ki * 32, SV_PLDL2KEEP);
            svprfh(svptrue_b16(), a + (mi + 16) * k + ki * 32, SV_PLDL2KEEP);
        }
}

__arm_new("za") void bf16_packed_gemm_alpha1_beta0(int m, int n, int ldc, int k, bfloat16_t *a, bfloat16_t *b,
    bfloat16_t *c, float *sub_bias, bool row_bias, int macro_kernel_m, int macro_kernel_n, int macro_kernel_k) __arm_streaming
{
    bf16_packed_gemm_prf_opt_kernel(m, n, ldc, k, a, b, c, sub_bias, row_bias);
    return ;
    if (macro_kernel_k < k)
        macro_kernel_k = k;
    for (int bm = 0; bm < m; bm += macro_kernel_m) {
        int em = std::min(bm + macro_kernel_m, m);
        for (int k_offset = 0; k_offset < k; k_offset += macro_kernel_k) {
            int k_num = std::min(macro_kernel_k, k - k_offset);
            for (int bn = 0; bn < n; bn += macro_kernel_n) {
                int en = std::min(bn + macro_kernel_n, n);
                bf16_packed_gemm_macro_kernel(bm, em, bn, en, k, k_offset, k_num, ldc, a, b, c);
            }
        }
    }
}

__arm_new("za") void bf16_gemm_pack_kernel(int m, int n, bfloat16_t *src, int lds, bfloat16_t *dst,
    int ldd) __arm_streaming
{
    for (int mi = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi);
        int dst_off = mi * ldd;
        svbool_t pg_m = svwhilelt_b32(mi, m);
        for (int ni = 0; ni < n; ni += 32) {
            svzero_za();
            svbool_t pg_n = svwhilelt_b32(0, (n - ni) / 2);
            for (int i = 0; i < 16 && mi + i < m; i++)
                svld1_hor_za32(0, i, pg_n, src + (mi + i) * lds + ni);
            for (int i = 0; i < 16 && ni + i * 2 < n; i++)
                svst1_ver_za32(0, i, pg_m, dst + dst_off + (ni + i * 2) * m_num);
        }
    }
}

} // namespace kutacc
