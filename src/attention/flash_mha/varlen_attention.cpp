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
#include <arm_sve.h>
#include <unistd.h>
#include <memory>

#include <cstdlib>

#include "attention.h"
#include "utils/bf16.h"
#include "math/fast_exp.h"

namespace kutacc {

namespace {
inline float reduce_head(const bfloat16_t* q, const bfloat16_t* k, int64_t head_dim)
{
#ifdef __ARM_FEATURE_SVE
    const int64_t vl = svcnth();
    svfloat32_t sum_vec = svdup_f32(0);
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    for (int64_t i = 0; i < head_dim; i += vl) {
        svbfloat16_t q_vec = svld1(pg16, q + i);
        svbfloat16_t k_vec = svld1(pg16, k + i);
        svfloat32_t q_vec0 = svreinterpret_f32(svzip1(svdup_bf16(0), q_vec));
        svfloat32_t q_vec1 = svreinterpret_f32(svzip2(svdup_bf16(0), q_vec));
        svfloat32_t k_vec0 = svreinterpret_f32(svzip1(svdup_bf16(0), k_vec));
        svfloat32_t k_vec1 = svreinterpret_f32(svzip2(svdup_bf16(0), k_vec));
        sum_vec = svmla_m(pg32, sum_vec, q_vec0, k_vec0);
        sum_vec = svmla_m(pg32, sum_vec, q_vec1, k_vec1);
    }
    return svaddv(svptrue_b32(), sum_vec);
#endif
    float sum = 0;
    for (int64_t i = 0; i < head_dim; i++) {
        sum += to_float(q[i]) * to_float(k[i]);
    }
    return sum;
}

inline void mul_head(float attn_w, const bfloat16_t* v, float* out, int64_t head_dim, bool accumulate)
{
#ifdef __ARM_FEATURE_SVE
    const int64_t vl = svcnth();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    for (int64_t i = 0; i < head_dim; i += vl) {
        svbfloat16_t v_vec = svld1(pg16, v + i);
        svfloat32_t v_vec0 = svreinterpret_f32(svzip1(svdup_bf16(0), v_vec));
        svfloat32_t v_vec1 = svreinterpret_f32(svzip2(svdup_bf16(0), v_vec));
        svfloat32_t out_vec0;
        svfloat32_t out_vec1;
        if (accumulate) {
            out_vec0 = svld1(pg32, out + i);
            out_vec1 = svld1(pg32, out + i + svcntw());
            out_vec0 = svmla_x(pg32, out_vec0, v_vec0, attn_w);
            out_vec1 = svmla_x(pg32, out_vec1, v_vec1, attn_w);
        } else {
            out_vec0 = svmul_x(pg32, v_vec0, attn_w);
            out_vec1 = svmul_x(pg32, v_vec1, attn_w);
        }
        svst1(pg32, out + i, out_vec0);
        svst1(pg32, out + i + svcntw(), out_vec1);
    }
    return;
#endif
    for (int64_t i = 0; i < head_dim; i++) {
        if (accumulate) {
            out[i] += attn_w * to_float(v[i]);
        } else {
            out[i] = attn_w * to_float(v[i]);
        }
    }
}

inline void copy_out(const float* temp_out, bfloat16_t* out, int64_t size)
{
#ifdef __ARM_FEATURE_SVE
    const int64_t vl = svcnth();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    for (int64_t i = 0; i < size; i += vl) {
        svfloat32_t temp_out_vec0 = svld1(pg32, temp_out + i);
        svfloat32_t temp_out_vec1 = svld1(pg32, temp_out + i + svcntw());
        svbfloat16_t out_vec0 = svcvt_bf16_x(pg32, temp_out_vec0);
        svbfloat16_t out_vec1 = svcvt_bf16_x(pg32, temp_out_vec1);
        svbfloat16_t out_vec = svuzp1(out_vec0, out_vec1);
        svst1(pg16, out + i, out_vec);
    }
    return;
#endif
    for (int64_t i = 0; i < size; i++) {
        out[i] = to_bf16(temp_out[i]);
    }
}

void softmax_fusion_kernel(int64_t width, float* data, float scale, std::optional<int64_t> causal_width)
{
    const int64_t vl = svcntw();
    // mul scale & add mask & reduce max
    svfloat32_t reduce = svdup_f32(-INFINITY);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svfloat32_t values = svld1(pg, data + i);
        if (causal_width.has_value()) {
            svfloat32_t mask_values = svsel(svwhilelt_b32(i, causal_width.value()), svdup_f32(0), svdup_f32(-INFINITY));
            values = svmla_x(pg, mask_values, values, scale);
        } else {
            values = svmul_x(pg, values, scale);
        }
        reduce = svmax_m(pg, reduce, values);
        svst1(pg, data + i, values);
    }
    float max = svmaxv(svptrue_b32(), reduce);
    // sub max & exp & reduce sum
    reduce = svdup_f32(0.f);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svfloat32_t values = svld1(pg, data + i);
        values = fast_exp(pg, svsub_x(pg, values, max));
        reduce = svadd_m(pg, reduce, values);
        svst1(pg, data + i, values);
    }
    // mul sum_inv
    float sum_inv = 1 / svaddv(svptrue_b32(), reduce);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svfloat32_t values = svld1(pg, data + i);
        values = svmul_x(pg, values, sum_inv);
        svst1(pg, data + i, values);
    }
}

void attention_impl(const Tensor<bfloat16_t, 3>& q, const Tensor<bfloat16_t, 3>& k, const Tensor<bfloat16_t, 3>& v,
    bool causal, double softmax_scale, const Tensor<bfloat16_t, 3>& out, int64_t hi)
{
    auto q_len = q.size(0);
    auto k_len = k.size(0);
    auto qk_head_dim = q.size(2);
    auto nope_head_dim = v.size(2);
    const int64_t q_block_size = 64;
    auto attn_weights = std::make_unique<float[]>(q_block_size * k_len);
    auto temp_out = std::make_unique<float[]>(q_block_size * nope_head_dim);
    for (int64_t q_start = 0; q_start < q_len; q_start += q_block_size) {
        int64_t q_step = std::min(q_len - q_start, q_block_size);
        int64_t q_end = q_start + q_step;
        for (int64_t qi = q_start; qi < q_end; ++qi) {
            for (int64_t ki = 0; ki < k_len; ++ki) {
                auto q_data = q.data_ptr() + qi * q.stride(0) + hi * qk_head_dim;
                auto k_data = k.data_ptr() + ki * k.stride(0) + hi * qk_head_dim;
                auto a_pos = attn_weights.get() + (qi - q_start) * k_len + ki;
                *a_pos = reduce_head(q_data, k_data, qk_head_dim);
            }
        }

        for (int64_t qi = q_start; qi < q_end; ++qi) {
            auto a_data = attn_weights.get() + (qi - q_start) * k_len;
            std::optional<int64_t> causal_width;
            if (causal) {
                causal_width = std::make_optional(qi - q_len + k_len + 1);
            }
            softmax_fusion_kernel(k_len, a_data, softmax_scale, causal_width);
        }

        for (int64_t qi = q_start; qi < q_end; ++qi) {
            for (int64_t ki = 0; ki < k_len; ++ki) {
                float attn_w = attn_weights[(qi - q_start) * k_len + ki];
                auto v_data = v.data_ptr() + ki * v.stride(0) + hi * nope_head_dim;
                auto temp_out_data = temp_out.get() + (qi - q_start) * nope_head_dim;
                mul_head(attn_w, v_data, temp_out_data, nope_head_dim, ki != 0);
            }
            copy_out(temp_out.get() + (qi - q_start) * nope_head_dim,
                out.data_ptr() + qi * out.stride(0) + hi * nope_head_dim, nope_head_dim);
        }
    }
}

}  // namespace

void varlen_attention(
    const Tensor<bfloat16_t, 3> &(q),
    const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v),
    const Tensor<bfloat16_t, 3> &(out),
    bool causal,
    double softmax_scale,
    const Tensor<int, 1> &(cu_seqlens_q),
    const Tensor<int, 1> &(cu_seqlens_k)
    )
{
    int64_t num_heads = q.size(1);
    int64_t num_seqs = cu_seqlens_q.size(0) - 1;
    auto cu_seqlens_q_data = cu_seqlens_q.data_ptr();
    auto cu_seqlens_k_data = cu_seqlens_k.data_ptr();

    kutacc::parallel_for(0, num_seqs * num_heads, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            int64_t bi = i / num_heads;
            int64_t hi = i % num_heads;
            auto q_start = cu_seqlens_q_data[bi];
            auto q_end = cu_seqlens_q_data[bi + 1];
            auto k_start = cu_seqlens_k_data[bi];
            auto k_end = cu_seqlens_k_data[bi + 1];
            attention_impl(tensor_slice(q, 0, q_start, q_end), tensor_slice(k, 0, k_start, k_end),
                tensor_slice(v, 0, k_start, k_end), causal, softmax_scale,
                tensor_slice(out, 0, q_start, q_end), hi);
        }
    });
}

}  // namespace attn
