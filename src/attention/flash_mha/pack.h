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
#include <optional>

#include "kutacc.h"
#include "utils/bf16.h"
#include "utils/check.h"

namespace kutacc {

constexpr int64_t br = 128;  // flashattn v2中q k S o m l五组数据大小总和小于1636的768KB L2/Core
constexpr int64_t bc = 128;

std::tuple<int64_t, int64_t> get_flash_attention_block();

__arm_new("za") inline void block_pack_bf16_1VL_trans_gemm_varlen(bfloat16_t* mat, bfloat16_t* pack_mat, int64_t m,
    int64_t n, int64_t row_strides, int64_t block_size) __arm_streaming
{
    const int svls = 16;
    const int svlh = 32;
    svbool_t pg_32_all = svptrue_b32();
    svbool_t pg_16_all = svptrue_b16();
    const int row_block = block_size;
    const int row_single_step = svls;
    const int row_step = 2 * svls;
    const int col_single_step = svlh;
    const int col_step = 2 * svlh;
    const int prf_stride = 128;
    const int pack_mat_row_size = row_single_step * n;

    int rest_slpit_len = m % row_step;
    int actual_row_step = m - rest_slpit_len;
    int rest_slpit_single_len = m % row_single_step;
    int actual_row_single_step = m - rest_slpit_single_len;
    // rowstep可以完整步进的部分
    for (int outer_row_index = 0; outer_row_index < actual_row_step; outer_row_index += row_step) {
        int group_row_index = outer_row_index / block_size;
        int block_base = group_row_index * block_size * n;
        int fst_vl_row_index = (outer_row_index % block_size) / row_single_step;
        int snd_vl_row_index = fst_vl_row_index + 1;
        for (int col_index = 0; col_index < n; col_index += col_step) {
#pragma unroll(8)
            for (int row_index = 0; row_index < row_single_step; row_index += 1) {
                svfloat32_t n_data0 = svld1(pg_32_all,
                    reinterpret_cast<float*>(&mat[(row_index + outer_row_index) * row_strides + col_index]));
                svfloat32_t n_data1 = svld1(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + col_single_step]));
                svfloat32_t n_data2 = svld1(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index + row_single_step) * row_strides + col_index]));
                svfloat32_t n_data3 = svld1(pg_32_all,
                    reinterpret_cast<float*>(&mat[(row_index + outer_row_index + row_single_step) * row_strides +
                                                  col_index + col_single_step]));

                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + prf_stride]),
                    svprfop::SV_PLDL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + prf_stride + col_single_step]),
                    svprfop::SV_PLDL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index + row_single_step) * row_strides + col_index + prf_stride]),
                    svprfop::SV_PLDL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(&mat[(row_index + outer_row_index + row_single_step) * row_strides +
                                                  col_index + prf_stride + col_single_step]),
                    svprfop::SV_PLDL1KEEP);

                svwrite_hor_za32_f32_m(0, row_index, pg_32_all, n_data0);
                svwrite_hor_za32_f32_m(1, row_index, pg_32_all, n_data1);
                svwrite_hor_za32_f32_m(2, row_index, pg_32_all, n_data2);
                svwrite_hor_za32_f32_m(3, row_index, pg_32_all, n_data3);
            }
#pragma unroll(8)
            for (int row_index = 0; row_index < row_single_step; row_index += 1) {
                svfloat32_t n_data0 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 0, row_index);
                svfloat32_t n_data1 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 1, row_index);
                svfloat32_t n_data2 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 2, row_index);
                svfloat32_t n_data3 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 3, row_index);

                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data0);
                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data1);
                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + snd_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data2);
                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + snd_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data3);

                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + snd_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + snd_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
            }
        }
    }
    // rowstep不可以完整步进的部分且剩余部分rownum大于svls就再算一个row_single_step
    if (rest_slpit_len >= row_single_step) {
        int outer_row_index = actual_row_step;
        int group_row_index = outer_row_index / block_size;
        int block_base = group_row_index * block_size * n;
        int fst_vl_row_index = (outer_row_index % block_size) / row_single_step;
        int snd_vl_row_index = fst_vl_row_index + 1;
        for (int col_index = 0; col_index < n; col_index += col_step) {
// 块0-0,0-1是完整的，装入时为完整装入ZA寄存器，取出时由于转置因此完整取出但标注谓词
#pragma unroll(8)
            for (int row_index = 0; row_index < row_single_step; row_index += 1) {
                svfloat32_t n_data0 = svld1(pg_32_all,
                    reinterpret_cast<float*>(&mat[(row_index + outer_row_index) * row_strides + col_index]));
                svfloat32_t n_data1 = svld1(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + col_single_step]));

                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + prf_stride]),
                    svprfop::SV_PLDL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + prf_stride + col_single_step]),
                    svprfop::SV_PLDL1KEEP);

                svwrite_hor_za32_f32_m(0, row_index, pg_32_all, n_data0);
                svwrite_hor_za32_f32_m(1, row_index, pg_32_all, n_data1);
            }
#pragma unroll(8)
            for (int row_index = 0; row_index < row_single_step; row_index += 1) {
                svfloat32_t n_data0 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 0, row_index);
                svfloat32_t n_data1 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 1, row_index);

                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data0);
                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data1);

                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
            }
        }
    }
    if (rest_slpit_single_len) {
        // 算剩下N小于slvs的部分
        int outer_row_index = actual_row_single_step;
        int group_row_index = outer_row_index / block_size;
        int block_base = group_row_index * block_size * n;
        int fst_vl_row_index = (outer_row_index % block_size) / row_single_step;
        for (int col_index = 0; col_index < n; col_index += col_step) {
            // 块0-0,0-1是不完整的，装入时为不完整装入ZA寄存器，取出时由于转置因此完整取出但标注谓词
            svzero_za();
            for (int row_index = 0; row_index < rest_slpit_single_len; row_index += 1) {
                svfloat32_t n_data0 = svld1(pg_32_all,
                    reinterpret_cast<float*>(&mat[(row_index + outer_row_index) * row_strides + col_index]));
                svfloat32_t n_data1 = svld1(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + col_single_step]));

                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + prf_stride]),
                    svprfop::SV_PLDL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &mat[(row_index + outer_row_index) * row_strides + col_index + prf_stride + col_single_step]),
                    svprfop::SV_PLDL1KEEP);

                svwrite_hor_za32_f32_m(0, row_index, pg_32_all, n_data0);
                svwrite_hor_za32_f32_m(1, row_index, pg_32_all, n_data1);
            }

#pragma unroll(8)
            for (int row_index = 0; row_index < row_single_step; row_index += 1) {
                svfloat32_t n_data0 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 0, row_index);
                svfloat32_t n_data1 = svread_ver_za32_f32_m(svfloat32_t(), pg_32_all, 1, row_index);

                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data0);
                svst1(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step) * row_single_step + row_index * col_single_step]),
                    n_data1);

                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 0 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
                svprfw(pg_32_all,
                    reinterpret_cast<float*>(
                        &pack_mat[block_base + fst_vl_row_index * pack_mat_row_size +
                                  (col_index + 1 * col_single_step + prf_stride) * row_single_step +
                                  row_index * col_single_step]),
                    svprfop::SV_PSTL1KEEP);
            }
        }
    }
}

__arm_new("za") inline void block_pack_bf16_2VL_gemm_varlen(bfloat16_t* mat, bfloat16_t* pack_mat, int64_t m, int64_t n,
    int64_t row_strides, int64_t block_size) __arm_streaming
{
    svbool_t pg_32_all = svptrue_b32();
    svbool_t pg_16_all = svptrue_b16();
    const int svls = 16;
    const int svlh = 32;
    const int col_single_step = svlh;
    const int col_step = svlh;
    int row_block = 0;
    int col_block = 0;
    int len_after_padding = (m + block_size - 1) / block_size * block_size;
    int rest_slpit_len = m % 2;
    int actual_row_step = m - rest_slpit_len;
    for (int col_index = 0; col_index < n; col_index += svlh) {
        for (int row_index = 0; row_index < actual_row_step; row_index += 2) {
            row_block = row_index / block_size;
            svbfloat16_t n_data0 = svld1(pg_16_all, mat + row_index * row_strides + col_index);
            svbfloat16_t n_data1 = svld1(pg_16_all, mat + (row_index + 1) * row_strides + col_index);
            svbfloat16_t first_svb16 = svzip1(n_data0, n_data1);
            svbfloat16_t second_svb16 = svzip2(n_data0, n_data1);
            svst1(pg_16_all,
                pack_mat + row_block * n * block_size + (col_index / 32) * 16 * block_size +
                    (row_index % block_size) * 16,
                first_svb16);
            svst1(pg_16_all,
                pack_mat + row_block * n * block_size + (col_index / 32 + n / 32) * 16 * block_size +
                    (row_index % block_size) * 16,
                second_svb16);
        }
    }
    for (int col_index = 0; col_index < n; col_index += svlh) {
        for (int row_index = actual_row_step; row_index < len_after_padding; row_index += 2) {
            row_block = row_index / block_size;
            svst1(pg_16_all,
                pack_mat + row_block * n * block_size + (col_index / 32) * 16 * block_size +
                    (row_index % block_size) * 16,
                svdup_bf16(0));
            svst1(pg_16_all,
                pack_mat + row_block * n * block_size + (col_index / 32 + n / 32) * 16 * block_size +
                    (row_index % block_size) * 16,
                svdup_bf16(0));
        }
    }
    if (rest_slpit_len == 1) {
        for (int col_index = 0; col_index < n; col_index += svlh) {
            for (int row_index = actual_row_step; row_index < actual_row_step + 1; row_index += 2) {
                row_block = row_index / block_size;
                svbfloat16_t n_data0 = svld1(pg_16_all, mat + row_index * row_strides + col_index);
                svbfloat16_t n_data1 = svdup_bf16(0);
                svbfloat16_t first_svb16 = svzip1(n_data0, n_data1);
                svbfloat16_t second_svb16 = svzip2(n_data0, n_data1);
                svst1(pg_16_all,
                    pack_mat + row_block * n * block_size + (col_index / 32) * 16 * block_size +
                        (row_index % block_size) * 16,
                    first_svb16);
                svst1(pg_16_all,
                    pack_mat + row_block * n * block_size + (col_index / 32 + n / 32) * 16 * block_size +
                        (row_index % block_size) * 16,
                    second_svb16);
            }
        }
    }
}

inline int64_t get_len_per_head(int64_t num_heads, int64_t output_len)
{
    return (output_len / 1024) * 1024 / num_heads;
}

void flash_attention_v_block_pack(int64_t kv_len, int64_t num_heads, int64_t vo_head_dim, int64_t output_len,
    int64_t input_stride0, int64_t input_stride1, bfloat16_t* input, bfloat16_t* output);

void flash_attention_k_block_pack(int64_t kv_len, int64_t num_heads, int64_t qk_head_dim, int64_t output_len,
    int64_t input_stride0, int64_t input_stride1, bfloat16_t* input, bfloat16_t* output);
}  // namespace attn
