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
#include "kutacc.h"

#include <arm_sve.h>
#include <cmath>

namespace kutacc {
const int max_seq_len = 8192;
const double PI = acos(-1);

template <typename scalar_t>
inline void odd_even_grouping_kernel(int64_t head_size, scalar_t* x, scalar_t* y, const scalar_t* x_prfm,
    const scalar_t* cos)
{
    int64_t i = 0;
    const scalar_t* sin = cos + head_size / 2;
#ifdef __ARM_FEATURE_SVE
    const int step = svcnth();
#pragma unroll(4)
    for (; i + step <= head_size; i += step) {
        __builtin_prefetch(x_prfm + i, 1, 3);
        svbool_t pg = svwhilelt_b16(step / 2, step);
        if constexpr (std::is_same<typename std::decay<scalar_t>::type, float16_t>::value) {
            svfloat16x2_t _x = svld2(pg, x + i);
            svfloat32_t x1 = svcvt_f32_x(svptrue_b32(), svzip1(svget2(_x, 0), svget2(_x, 0)));
            svfloat32_t x2 = svcvt_f32_x(svptrue_b32(), svzip1(svget2(_x, 1), svget2(_x, 1)));
            svfloat16_t _c = svld1(pg, cos + i / 2);
            svfloat16_t _s = svld1(pg, sin + i / 2);
            svfloat32_t c = svcvt_f32_x(svptrue_b32(), svzip1(_c, _c));
            svfloat32_t s = svcvt_f32_x(svptrue_b32(), svzip1(_s, _s));
            svfloat32_t y1 = svmls_x(svptrue_b32(), svmul_x(svptrue_b32(), x1, c), x2, s);
            svfloat32_t y2 = svmla_x(svptrue_b32(), svmul_x(svptrue_b32(), x2, c), x1, s);
            svfloat16_t z1 = svcvt_f16_x(svptrue_b32(), y1);
            svfloat16_t z2 = svcvt_f16_x(svptrue_b32(), y2);
            svst2(pg, y + i, svcreate2(svuzp1(z1, z1), svuzp1(z2, z2)));
        } else {
            svbfloat16_t zero_b = svdup_bf16(0);
            svbfloat16x2_t _x = svld2(pg, x + i);
            svfloat32_t x1 = svreinterpret_f32(svzip1(zero_b, svget2(_x, 0)));
            svfloat32_t x2 = svreinterpret_f32(svzip1(zero_b, svget2(_x, 1)));
            svbfloat16_t _c = svld1(pg, cos + i / 2);
            svbfloat16_t _s = svld1(pg, sin + i / 2);
            svfloat32_t c = svreinterpret_f32(svzip1(zero_b, _c));
            svfloat32_t s = svreinterpret_f32(svzip1(zero_b, _s));
            svfloat32_t y1 = svmls_x(svptrue_b32(), svmul_x(svptrue_b32(), x1, c), x2, s);
            svfloat32_t y2 = svmla_x(svptrue_b32(), svmul_x(svptrue_b32(), x2, c), x1, s);
            svbfloat16_t z1 = svcvt_bf16_x(svptrue_b32(), y1);
            svbfloat16_t z2 = svcvt_bf16_x(svptrue_b32(), y2);
            svst2(pg, y + i, svcreate2(svuzp1(z1, zero_b), svuzp1(z2, zero_b)));
        }
    }
#endif
    for (; i < head_size; i += 2) {
        float x1 = x[i];
        float x2 = x[i + 1];
        float c = cos[i / 2];
        float s = sin[i / 2];
        y[i] = (scalar_t)(x1 * c - x2 * s);
        y[i + 1] = (scalar_t)(x2 * c + x1 * s);
    }
}

template <>
inline void odd_even_grouping_kernel(int64_t head_size, bfloat16_t* x, bfloat16_t* y, const bfloat16_t* x_prfm,
    const bfloat16_t* cos)
{
    const bfloat16_t* sin = cos + head_size / 2;
    for (int64_t i = 0; i < head_size; i += 64) {
        __builtin_prefetch(x_prfm + i, 1, 3);
        __builtin_prefetch(x_prfm + i + 32, 1, 3);
        svbool_t ptrue = svptrue_b16();
        svbfloat16_t zero_b = svdup_bf16(0);
        svbool_t p1 = svwhilelt_b16(i, head_size);
        svbool_t p2 = svwhilelt_b16(i + 32, head_size);

        svbfloat16_t x1 = svldnt1(p1, x + i);                      // 0, 1, ..., 31
        svbfloat16_t x2 = svldnt1(p2, x + i + 32);                 // 32, 33, ..., 63
        svbfloat16_t x3 = svuzp1(x1, x2);                          // 0, 2, ..., 62
        svbfloat16_t x4 = svuzp2(x1, x2);                          // 1, 3, ..., 63
        svfloat32_t x1_1 = svreinterpret_f32(svzip1(zero_b, x3));  // 0, 2, ..., 30
        svfloat32_t x1_2 = svreinterpret_f32(svzip2(zero_b, x3));  // 32, 34, ..., 62
        svfloat32_t x2_1 = svreinterpret_f32(svzip1(zero_b, x4));  // 1, 3, ..., 31
        svfloat32_t x2_2 = svreinterpret_f32(svzip2(zero_b, x4));  // 33, 35, ..., 63

        svbfloat16_t c = svld1(svwhilelt_b16(i / 2, head_size / 2), cos + i / 2);  // 0, 1, ..., 31
        svbfloat16_t s = svld1(svwhilelt_b16(i / 2, head_size / 2), sin + i / 2);  // 0, 1, ..., 31
        svfloat32_t c_1 = svreinterpret_f32(svzip1(zero_b, c));                    // 0, 1, ..., 15
        svfloat32_t c_2 = svreinterpret_f32(svzip2(zero_b, c));                    // 16, 17, ..., 31
        svfloat32_t s_1 = svreinterpret_f32(svzip1(zero_b, s));                    // 0, 1, ..., 15
        svfloat32_t s_2 = svreinterpret_f32(svzip2(zero_b, s));                    // 16, 17, ..., 31

        svfloat32_t y1_1 = svmls_x(ptrue, svmul_x(ptrue, x1_1, c_1), x2_1, s_1);
        svfloat32_t y1_2 = svmls_x(ptrue, svmul_x(ptrue, x1_2, c_2), x2_2, s_2);
        svfloat32_t y2_1 = svmla_x(ptrue, svmul_x(ptrue, x2_1, c_1), x1_1, s_1);
        svfloat32_t y2_2 = svmla_x(ptrue, svmul_x(ptrue, x2_2, c_2), x1_2, s_2);

        svbfloat16_t y3 = svuzp1(svcvt_bf16_x(ptrue, y1_1), svcvt_bf16_x(ptrue, y1_2));  // 0, 2, ..., 62
        svbfloat16_t y4 = svuzp1(svcvt_bf16_x(ptrue, y2_1), svcvt_bf16_x(ptrue, y2_2));  // 1, 3, ..., 63

        svst1(p1, y + i, svzip1(y3, y4));
        svst1(p2, y + i + 32, svzip2(y3, y4));
    }
}

template <typename scalar_t>
void rope_out(
    const Tensor<int64_t, 1>& position_ids,
    const Tensor<scalar_t, 3>& q,
    const Tensor<scalar_t, 3>& k,
    const Tensor<scalar_t, 3>& q_out,
    const Tensor<scalar_t, 3>& k_out,
    const Tensor<scalar_t, 2>& cos_sin_cache)
{
    const int64_t prfm_dist = 2;
    int64_t n_tokens = position_ids.size(0);
    int64_t num_q_heads = q.size(1);
    int64_t num_k_heads = k.size(1);
    int64_t head_size = cos_sin_cache.size(1);

    int64_t q_stride0 = q.stride(0);
    int64_t q_stride1 = q.stride(1);
    int64_t k_stride0 = k.stride(0);
    int64_t k_stride1 = k.stride(1);
    int64_t q_out_stride0 = q_out.stride(0);
    int64_t q_out_stride1 = q_out.stride(1);
    int64_t k_out_stride0 = k_out.stride(0);
    int64_t k_out_stride1 = k_out.stride(1);

    scalar_t* q_data = q.data_ptr();
    scalar_t* k_data = k.data_ptr();
    scalar_t* q_out_data = q_out.data_ptr();
    scalar_t* k_out_data = k_out.data_ptr();
    scalar_t* cos_sin_cache_data = cos_sin_cache.data_ptr();
    int64_t* pid_data = position_ids.data_ptr();

    kutacc::parallel_for(0, n_tokens, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            int64_t pindex = i;
            scalar_t* q_start = q_data + i * q_stride0;
            scalar_t* k_start = k_data + i * k_stride0;
            scalar_t* q_out_start = q_out_data + i * q_out_stride0;
            scalar_t* k_out_start = k_out_data + i * k_out_stride0;
            
            scalar_t* q_prfm = i + prfm_dist < end ? q_start + prfm_dist * q_stride0 : q_start;
            scalar_t* k_prfm = i + prfm_dist < end ? k_start + prfm_dist * k_stride0 : k_start;

            for (int64_t hi = 0; hi < num_q_heads; hi++) {
                odd_even_grouping_kernel<scalar_t>(
                    head_size,
                    q_start + hi * q_stride1,
                    q_out_start + hi * q_out_stride1,
                    q_prfm + hi * q_stride1,
                    cos_sin_cache_data + pid_data[pindex] * head_size);
            }

            for (int64_t kv_hi = 0; kv_hi < num_k_heads; kv_hi++) {
                odd_even_grouping_kernel<scalar_t>(
                    head_size,
                    k_start + kv_hi * k_stride1,
                    k_out_start + kv_hi * k_out_stride1,
                    k_prfm + kv_hi * k_stride1,
                    cos_sin_cache_data + pid_data[pindex] * head_size);
            }
        }
    });
}

void rope(
    const Tensor<int64_t, 1>& position_ids,
    const Tensor<__bf16, 3>& q,
    const Tensor<__bf16, 3>& k,
    const Tensor<__bf16, 3>& q_out,
    const Tensor<__bf16, 3>& k_out,
    const Tensor<__bf16, 2>& cos_sin_cache)
{
        rope_out<__bf16>(position_ids, q, k, q_out, k_out, cos_sin_cache);
    };
}  // namespace kutacc
