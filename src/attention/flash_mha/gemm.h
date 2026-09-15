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

#include <arm_sme.h>
#include <arm_sve.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <optional>

#include "attention.h"
#include "utils/bf16.h"

namespace kutacc {
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

__arm_new("za") inline void qk_gemm_4vl_1vl(bfloat16_t* temp_q, bfloat16_t* temp_k, float* s, int64_t q_len,
    int64_t k_len, int64_t qk_dim, int64_t q_start, int64_t k_start, int prefill_start) __arm_streaming
{
    constexpr int svls = 16;
    constexpr int svlh = 32;
    svbool_t pg_32_all = svptrue_b32();
    svbool_t pg_16_all = svptrue_b16();
    constexpr int row_single_step = svls;
    constexpr int row_step = 4 * svls;
    constexpr int col_single_step = svls;
    constexpr int col_step = 1 * svls;
    constexpr int prf_stride = 12;
    int left_rows = q_len % row_step;
    int intact_row_max = q_len - left_rows;
    for (int row_index = 0; row_index < intact_row_max; row_index += row_step) {
        for (int col_index = 0; col_index < k_len; col_index += col_step) {
            if (unlikely(q_start + prefill_start == k_start && col_index + 1 > row_index + row_step)) {
                continue;
            }
            const bfloat16_t* k_base0 = temp_k + col_index * qk_dim;

            const bfloat16_t* q_base0 = temp_q + row_index * qk_dim;
            const bfloat16_t* q_base1 = q_base0 + row_single_step * qk_dim;
            const bfloat16_t* q_base2 = q_base1 + row_single_step * qk_dim;
            const bfloat16_t* q_base3 = q_base2 + row_single_step * qk_dim;

            const int col0_offset = col_index * q_len;

            const int row0_offset = row_index;
            const int row1_offset = row_index + svls;
            const int row2_offset = row_index + 2 * svls;
            const int row3_offset = row_index + 3 * svls;

            svzero_za();
#pragma unroll(6)
            for (int k_dim = 0; k_dim < svls * qk_dim; k_dim += svlh) {
                svbfloat16_t b0_data0 = svld1(pg_16_all, k_base0 + k_dim);
                svprfh(pg_16_all, k_base0 + k_dim + (prf_stride)*svlh, svprfop::SV_PLDL1KEEP);

                svbfloat16_t a0_data0 = svld1(pg_16_all, q_base0 + k_dim);
                svbfloat16_t a0_data1 = svld1(pg_16_all, q_base1 + k_dim);
                svbfloat16_t a0_data2 = svld1(pg_16_all, q_base2 + k_dim);
                svbfloat16_t a0_data3 = svld1(pg_16_all, q_base3 + k_dim);
                svmopa_za32_bf16_m(0, pg_16_all, pg_16_all, a0_data0, b0_data0);
                svmopa_za32_bf16_m(1, pg_16_all, pg_16_all, a0_data1, b0_data0);
                svmopa_za32_bf16_m(2, pg_16_all, pg_16_all, a0_data2, b0_data0);
                svmopa_za32_bf16_m(3, pg_16_all, pg_16_all, a0_data3, b0_data0);
            }

#pragma unroll(8)
            for (int row_loop = 0; row_loop < svls; row_loop++) {
                svfloat32_t out_za_s0 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 0, row_loop);
                svfloat32_t out_za_s1 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 1, row_loop);
                svfloat32_t out_za_s2 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 2, row_loop);
                svfloat32_t out_za_s3 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 3, row_loop);

                const int loop_row_offset = row_loop * q_len;

                svst1(pg_16_all, s + col0_offset + row0_offset + loop_row_offset, out_za_s0);
                svst1(pg_16_all, s + col0_offset + row1_offset + loop_row_offset, out_za_s1);
                svst1(pg_16_all, s + col0_offset + row2_offset + loop_row_offset, out_za_s2);
                svst1(pg_16_all, s + col0_offset + row3_offset + loop_row_offset, out_za_s3);
            }
        }
    }
    for (int row_index = intact_row_max; row_index < intact_row_max + left_rows; row_index += row_single_step) {
        for (int col_index = 0; col_index < k_len; col_index += col_step) {
            if (unlikely(q_start + prefill_start == k_start && col_index + 1 > row_index + row_step)) {
                continue;
            }
            const bfloat16_t* k_base0 = temp_k + col_index * qk_dim;

            const bfloat16_t* q_base0 = temp_q + row_index * qk_dim;

            const int col0_offset = col_index * q_len;

            const int row0_offset = row_index;

            svzero_za();
#pragma unroll(8)
            for (int k_dim = 0; k_dim < svls * qk_dim; k_dim += svlh) {
                svbfloat16_t b0_data0 = svld1(pg_16_all, k_base0 + k_dim);
                svprfh(pg_16_all, k_base0 + k_dim + (prf_stride)*svlh, svprfop::SV_PLDL1KEEP);

                svbfloat16_t a0_data0 = svld1(pg_16_all, q_base0 + k_dim);
                svmopa_za32_bf16_m(0, pg_16_all, pg_16_all, a0_data0, b0_data0);
            }

#pragma unroll(8)
            for (int row_loop = 0; row_loop < svls; row_loop++) {
                svfloat32_t out_za_s0 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 0, row_loop);

                const int loop_row_offset = row_loop * q_len;

                svst1(pg_16_all, s + col0_offset + row0_offset + loop_row_offset, out_za_s0);
            }
        }
    }
}

__arm_new("za") inline void pv_gemm_2vl_2vl_out_gemv(float* mat1, bfloat16_t* mat2, float* result, int64_t q_len,
    int64_t vo_head_dim, int64_t k_len, float* old_result, float* old_maxs, int64_t q_start, int64_t k_start,
    [[maybe_unused]] bfloat16_t* out_ptr, int64_t bc, int prefill_start) __arm_streaming
{
    const int svls = 16;
    const int svlh = 32;
    svbool_t pg_32_all = svptrue_b32();
    svbool_t pg_16_all = svptrue_b16();
    const int row_single_step = svls;
    const int row_step = 2 * svls;
    const int col_single_step = svls;
    const int col_step = 2 * svls;
    const int prf_stride = 12;
    int rest_rows = q_len % row_step;
    int intact_row_max = q_len - rest_rows;
    for (int col_index = 0; col_index < vo_head_dim; col_index += col_step) {
        const bfloat16_t* mat2_base0 = mat2 + col_index * bc / 2;
        const bfloat16_t* mat2_base1 = mat2 + col_index * bc / 2 + 64 * bc;

        for (int row_index = 0; row_index < intact_row_max; row_index += row_step) {
            float* mat1_base0 = mat1 + row_index;
            float* mat1_base1 = mat1 + (row_index + svls);

            float* result_row0_base = result + row_index * vo_head_dim + col_index;
            float* result_row1_base = result + row_index * vo_head_dim + col_index + svls;
            float* result_row2_base = result + (row_index + svls) * vo_head_dim + col_index;
            float* result_row3_base = result + (row_index + svls) * vo_head_dim + col_index + svls;

            float* old_result_row0_base = old_result + row_index * vo_head_dim + col_index;
            float* old_result_row1_base = old_result + row_index * vo_head_dim + col_index + svls;
            float* old_result_row2_base = old_result + (row_index + svls) * vo_head_dim + col_index;
            float* old_result_row3_base = old_result + (row_index + svls) * vo_head_dim + col_index + svls;

            svzero_za();
#pragma unroll(2)
            for (int k_dim = 0; k_dim < k_len; k_dim += 2) {
                if (unlikely(q_start + prefill_start == k_start && k_dim > row_index + row_step)) {
                    continue;
                }
                svbfloat16_t v_data0 = svld1(pg_16_all, mat2_base0 + (k_dim % bc) * 16);
                svbfloat16_t v_data1 = svld1(pg_16_all, mat2_base1 + (k_dim % bc) * 16);

                svbfloat16_t input0 = svld1(pg_16_all, reinterpret_cast<bfloat16_t*>(mat1_base0 + k_dim * q_len));
                svbfloat16_t input1 = svld1(pg_16_all, reinterpret_cast<bfloat16_t*>(mat1_base1 + k_dim * q_len));

                svprfh(pg_16_all, mat2_base0 + ((k_dim % bc) + prf_stride) * 16, svprfop::SV_PLDL1KEEP);
                svprfh(pg_16_all, mat2_base1 + ((k_dim % bc) + prf_stride) * 16, svprfop::SV_PLDL1KEEP);

                svmopa_za32_bf16_m(0, pg_16_all, pg_16_all, input0, v_data0);
                svmopa_za32_bf16_m(1, pg_16_all, pg_16_all, input0, v_data1);

                svmopa_za32_bf16_m(2, pg_16_all, pg_16_all, input1, v_data0);
                svmopa_za32_bf16_m(3, pg_16_all, pg_16_all, input1, v_data1);
            }

#pragma unroll(4)
            for (int row_loop = 0; row_loop < svls; row_loop++) {
                const int loop_row_offset = row_loop * vo_head_dim;

                float vec_scale0 = old_maxs[row_index + row_loop];
                float vec_scale1 = old_maxs[row_index + svls + row_loop];

                svfloat32_t old_out_zas0 = svld1(pg_32_all, old_result_row0_base + loop_row_offset);
                svfloat32_t old_out_zas1 = svld1(pg_32_all, old_result_row1_base + loop_row_offset);
                svfloat32_t old_out_zas2 = svld1(pg_32_all, old_result_row2_base + loop_row_offset);
                svfloat32_t old_out_zas3 = svld1(pg_32_all, old_result_row3_base + loop_row_offset);

                svfloat32_t out_zas0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, row_loop);
                svfloat32_t out_zas1 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 1, row_loop);
                svfloat32_t out_zas2 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 2, row_loop);
                svfloat32_t out_zas3 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 3, row_loop);

                out_zas0 = svmla_x(pg_32_all, out_zas0, old_out_zas0, vec_scale0);
                out_zas1 = svmla_x(pg_32_all, out_zas1, old_out_zas1, vec_scale0);
                out_zas2 = svmla_x(pg_32_all, out_zas2, old_out_zas2, vec_scale1);
                out_zas3 = svmla_x(pg_32_all, out_zas3, old_out_zas3, vec_scale1);

                svst1(pg_32_all, result_row0_base + loop_row_offset, out_zas0);
                svst1(pg_32_all, result_row1_base + loop_row_offset, out_zas1);
                svst1(pg_32_all, result_row2_base + loop_row_offset, out_zas2);
                svst1(pg_32_all, result_row3_base + loop_row_offset, out_zas3);
            }
        }

        for (int row_index = intact_row_max; row_index < intact_row_max + rest_rows; row_index += row_single_step) {
            float* mat1_base0 = mat1 + row_index;

            float* result_row0_base = result + row_index * vo_head_dim + col_index;
            float* result_row1_base = result + row_index * vo_head_dim + col_index + svls;

            float* old_result_row0_base = old_result + row_index * vo_head_dim + col_index;
            float* old_result_row1_base = old_result + row_index * vo_head_dim + col_index + svls;

            svzero_za();
#pragma unroll(2)
            for (int k_dim = 0; k_dim < k_len; k_dim += 2) {
                if (unlikely(q_start + prefill_start == k_start && k_dim > row_index + row_step)) {
                    continue;
                }
                svbfloat16_t v_data0 = svld1(pg_16_all, mat2_base0 + (k_dim % bc) * 16);
                svbfloat16_t v_data1 = svld1(pg_16_all, mat2_base1 + (k_dim % bc) * 16);

                svbfloat16_t input0 = svld1(pg_16_all, reinterpret_cast<bfloat16_t*>(mat1_base0 + k_dim * q_len));

                svprfh(pg_16_all, mat2_base0 + ((k_dim % bc) + prf_stride) * 16, svprfop::SV_PLDL1KEEP);
                svprfh(pg_16_all, mat2_base1 + ((k_dim % bc) + prf_stride) * 16, svprfop::SV_PLDL1KEEP);

                svmopa_za32_bf16_m(0, pg_16_all, pg_16_all, input0, v_data0);
                svmopa_za32_bf16_m(1, pg_16_all, pg_16_all, input0, v_data1);
            }

#pragma unroll(4)
            for (int row_loop = 0; row_loop < svls; row_loop++) {
                const int loop_row_offset = row_loop * vo_head_dim;

                float vec_scale0 = old_maxs[row_index + row_loop];

                svfloat32_t old_out_zas0 = svld1(pg_32_all, old_result_row0_base + loop_row_offset);
                svfloat32_t old_out_zas1 = svld1(pg_32_all, old_result_row1_base + loop_row_offset);

                svfloat32_t out_zas0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, row_loop);
                svfloat32_t out_zas1 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 1, row_loop);

                out_zas0 = svmla_x(pg_32_all, out_zas0, old_out_zas0, vec_scale0);
                out_zas1 = svmla_x(pg_32_all, out_zas1, old_out_zas1, vec_scale0);

                svst1(pg_32_all, result_row0_base + loop_row_offset, out_zas0);
                svst1(pg_32_all, result_row1_base + loop_row_offset, out_zas1);
            }
        }
    }
}
}  // namespace attn
