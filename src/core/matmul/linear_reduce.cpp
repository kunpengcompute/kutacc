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
#include "linear_reduce.h"

namespace kutacc {
template <int64_t unroll_num>
inline void prefetch_L2_reduce(int64_t m, int64_t n, int64_t bk, bfloat16_t *origin_c, bfloat16_t *tmpc,
    int64_t start, int64_t end)
{
    for (int64_t i = start; i < end; i += svcnth()) {
        svprfb(svptrue_b8(), origin_c + i, SV_PSTL2KEEP);
#pragma unroll(unroll_num)
        for (int64_t k = 1; k < bk; ++k) {
            svprfb(svptrue_b8(), tmpc + i + k * m * n, SV_PLDL2KEEP);
        }
    }
}

template <int64_t unroll_num>
inline void prefetch_L2_reduce_stride(int64_t m, int64_t n, int64_t bk, bfloat16_t *origin_c, bfloat16_t *tmpc,
    int64_t start, int64_t end, int64_t n_start, int64_t n_length)
{
    for (int64_t i = start; i < end; i += svcnth()) {
        int64_t cur_row = i / n_length;
        int64_t cur_col = i % n_length;
        int64_t actual_bias = cur_row * n + n_start + cur_col;
        svprfb(svptrue_b8(), origin_c + actual_bias, SV_PSTL2KEEP);
#pragma unroll(unroll_num)
        for (int64_t k = 1; k < bk; ++k) {
            svprfb(svptrue_b8(), tmpc + actual_bias + k * m * n, SV_PLDL2KEEP);
        }
    }
}

template <int64_t unroll_num>
void reduce_c_reused(int64_t m, int64_t n, int64_t bk, bfloat16_t *origin_c, bfloat16_t *tmpc,
    int64_t start, int64_t end)
{
    svbfloat16_t zero = svdup_bf16(0);
    auto prefetch_dis = 32 + 256 / unroll_num; // 经验公式，不同场景下性能不同
    prefetch_L2_reduce<unroll_num>(m, n, bk, origin_c, tmpc, start, end);
    for (int64_t i = start; i < end; i += svcnth()) {
        svbool_t p = svwhilelt_b16(i, end);
        svbfloat16_t s = svld1(p, origin_c + i);
        svprfb(p, origin_c + i + prefetch_dis, SV_PSTL1KEEP);
        svfloat32_t s0 = svreinterpret_f32(svzip1(zero, s));
        svfloat32_t s1 = svreinterpret_f32(svzip2(zero, s));
#pragma unroll(unroll_num)
        for (int64_t k = 1; k < bk; ++k) {
            svbfloat16_t v = svld1(p, tmpc + i + k * m * n);
            svprfb(p, tmpc + i + prefetch_dis + k * m * n, SV_PLDL1KEEP);
            s0 = svadd_m(svptrue_b32(), s0, svreinterpret_f32(svzip1(zero, v)));
            s1 = svadd_m(svptrue_b32(), s1, svreinterpret_f32(svzip2(zero, v)));
        }
        svst1(p, origin_c + i, svuzp1(svcvt_bf16_x(svptrue_b32(), s0), svcvt_bf16_x(svptrue_b32(), s1)));
    }
}

inline void reduce_for_kblocks_c_reused(int64_t m, int64_t n, int64_t blocks_in_k, bfloat16_t *output_ptr,
    bfloat16_t *tmpc, int64_t start, int64_t end)
{
    if (blocks_in_k == 2) {
        reduce_c_reused<1>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    } else if (blocks_in_k == 4) {
        reduce_c_reused<3>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    } else if (blocks_in_k >= 8 && blocks_in_k % 8 == 0) {
        reduce_c_reused<7>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    } else {
        reduce_c_reused<1>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    }
}

template <int64_t unroll_num>
void reduce(int64_t m, int64_t n, int64_t bk, bfloat16_t *origin_c, bfloat16_t *tmpc, int64_t start, int64_t end)
{
    svbfloat16_t zero = svdup_bf16(0);
    auto prefetch_dis = 32 + 256 / unroll_num; // 经验公式，不同场景下性能不同
    for (int64_t i = start; i < end; i += svcnth()) {
        svbool_t p = svwhilelt_b16(i, end);
        svbfloat16_t s = svld1(p, tmpc + i);
        svprfb(p, tmpc + i + prefetch_dis, SV_PLDL1KEEP);
        svfloat32_t s0 = svreinterpret_f32(svzip1(zero, s));
        svfloat32_t s1 = svreinterpret_f32(svzip2(zero, s));
#pragma unroll(unroll_num)
        for (int64_t k = 1; k < bk; ++k) {
            svbfloat16_t v = svld1(p, tmpc + i + k * m * n);
            svprfb(p, tmpc + i + prefetch_dis + k * m * n, SV_PLDL1KEEP);
            s0 = svadd_m(svptrue_b32(), s0, svreinterpret_f32(svzip1(zero, v)));
            s1 = svadd_m(svptrue_b32(), s1, svreinterpret_f32(svzip2(zero, v)));
        }
        svst1(p, origin_c + i, svuzp1(svcvt_bf16_x(svptrue_b32(), s0), svcvt_bf16_x(svptrue_b32(), s1)));
    }
}

inline void reduce_for_kblocks(int64_t m, int64_t n, int64_t blocks_in_k, bfloat16_t *output_ptr, bfloat16_t *tmpc,
    int64_t start, int64_t end)
{
    if (blocks_in_k == 2) {
        reduce<2>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    } else if (blocks_in_k == 4) {
        reduce<4>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    } else if (blocks_in_k >= 8 && blocks_in_k % 8 == 0) {
        reduce<8>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    } else {
        reduce<1>(m, n, blocks_in_k, output_ptr, tmpc, start, end);
    }
}

template <int64_t unroll_num>
void reduce_stride_c_reused(int64_t m, int64_t n, int64_t bk, bfloat16_t *origin_c, bfloat16_t *tmpc,
    int64_t start, int64_t end, int64_t n_start, int64_t n_length)
{
    svbfloat16_t zero = svdup_bf16(0);
    auto prefetch_dis = 32 + 256 / unroll_num; // 经验公式，不同场景下性能不同
    prefetch_L2_reduce_stride<unroll_num>(m, n, bk, origin_c, tmpc, start, end, n_start, n_length);
    for (int64_t i = start; i < end; i += svcnth()) {
        int64_t cur_row = i / n_length;
        int64_t cur_col = i % n_length;
        int64_t actual_bias = cur_row * n + n_start + cur_col;
        svbool_t p = svwhilelt_b16(cur_col, n_length);
        svbfloat16_t s = svld1(p, origin_c + actual_bias);
        svprfb(p, origin_c + actual_bias + prefetch_dis, SV_PSTL1KEEP);
        svfloat32_t s0 = svreinterpret_f32(svzip1(zero, s));
        svfloat32_t s1 = svreinterpret_f32(svzip2(zero, s));
#pragma unroll(unroll_num)
        for (int64_t k = 1; k < bk; ++k) {
            svbfloat16_t v = svld1(p, tmpc + actual_bias + k * m * n);
            svprfb(p, tmpc + actual_bias + prefetch_dis + k * m * n, SV_PLDL1KEEP);
            s0 = svadd_m(svptrue_b32(), s0, svreinterpret_f32(svzip1(zero, v)));
            s1 = svadd_m(svptrue_b32(), s1, svreinterpret_f32(svzip2(zero, v)));
        }
        svst1(p, origin_c + actual_bias, svuzp1(svcvt_bf16_x(svptrue_b32(), s0), svcvt_bf16_x(svptrue_b32(), s1)));
    }
}

inline void reduce_stride_for_kblocks_c_reused(int64_t m, int64_t n, int64_t blocks_in_k, bfloat16_t *output_ptr,
    bfloat16_t *tmpc, int64_t start, int64_t end, int64_t n_start, int64_t n_length)
{
    if (blocks_in_k == 2) {
        reduce_stride_c_reused<1>(m, n, blocks_in_k, output_ptr, tmpc, start, end, n_start, n_length);
    } else if (blocks_in_k == 4) {
        reduce_stride_c_reused<3>(m, n, blocks_in_k, output_ptr, tmpc, start, end, n_start, n_length);
    } else if (blocks_in_k >= 8 && blocks_in_k % 8 == 0) {
        reduce_stride_c_reused<7>(m, n, blocks_in_k, output_ptr, tmpc, start, end, n_start, n_length);
    } else {
        reduce_stride_c_reused<1>(m, n, blocks_in_k, output_ptr, tmpc, start, end, n_start, n_length);
    }
}

__attribute__((always_inline)) void reduce_stride_filter(int64_t m, int64_t n, int64_t k, int64_t tile_k,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t start, int64_t end, int64_t n_start, int64_t n_length)
{
    int64_t blocks_in_k = k / tile_k;
    
    bool k_splited = (blocks_in_k > 1);
    bool c_reused = (blocks_in_k <= 4);

    if (k_splited) {
        kutacc::parallel_barrier();
        if (!c_reused) {
            std::cerr << "wrong case!!! not yet suppported";
            // reduce_stride_for_kblocks(m, n, blocks_in_k, output_ptr, tmpc, n_start, n_length);//待实现
        } else {
            reduce_stride_for_kblocks_c_reused(m, n, blocks_in_k, output_ptr, tmpc, start, end, n_start, n_length);
        }
    }
}

__attribute__((always_inline)) void reduce_filter(int64_t m, int64_t n, int64_t k, int64_t tile_k,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t start, int64_t end)
{
    int64_t blocks_in_k = k / tile_k;
    
    bool k_splited = (blocks_in_k > 1);
    bool c_reused = (blocks_in_k <= 4);

    if (k_splited) {
        kutacc::parallel_barrier();
        if (!c_reused) {
            reduce_for_kblocks(m, n, blocks_in_k, output_ptr, tmpc, start, end);
        } else {
            reduce_for_kblocks_c_reused(m, n, blocks_in_k, output_ptr, tmpc, start, end);
        }
    }
}
}