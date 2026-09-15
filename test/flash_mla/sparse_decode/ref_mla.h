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

#include <kutacc.h>

#include "../utils.h"
#include "testcase.h"

inline void ref_mla_sparse_decode(const TestcaseParams &p, const TestcaseData &d,
                                  kutacc::Tensor<bfloat16_t, 4> &o,
                                  kutacc::Tensor<float, 3> &softmax_lse,
                                  Allocator &allocator)
{
    int64_t num_threads = kutacc::get_thread_num();
    auto s = create_tensor<float, 3>(allocator, {num_threads, p.num_heads_q, p.topk + p.extra_topk});
    auto kv_token_ptr_buffer = create_tensor<bfloat16_t *, 2>(allocator, {num_threads, p.topk + p.extra_topk});
    int64_t task_num = p.batch_size * p.seqlen_q;
    kutacc::parallel_for(0, task_num, 1, [&](int64_t begin, int64_t end) {
        int64_t tid = kutacc::get_thread_id();
        for (int64_t task_idx = begin; task_idx < end; ++task_idx) {
            int64_t req_idx = task_idx / p.seqlen_q;
            int64_t q_token_idx = task_idx % p.seqlen_q;

            int64_t topk = p.have_topk_length ? d.topk_length.value().data_ptr()[req_idx] : p.topk;
            int64_t extra_topk =
                p.have_extra_topk_length ? d.extra_topk_length.value().data_ptr()[req_idx] : p.extra_topk;
            
            auto kv_token_ptr = &kv_token_ptr_buffer.index({tid});

            int seqlen_kv = 0;
            for (int i = 0; i < topk; ++i) {
                auto p = get_token_in_kvcache(d.kvcache, d.indices.index({req_idx, q_token_idx, i}));
                if (p != nullptr) {
                    kv_token_ptr[seqlen_kv++] = p;
                }
            }
            for (int i = 0; i < extra_topk; ++i) {
                auto p =
                    get_token_in_kvcache(d.extra_kvcache.value(),
                        d.extra_indices.value().index({req_idx, q_token_idx, i}));
                if (p != nullptr) {
                    kv_token_ptr[seqlen_kv++] = p;
                }
            }

            for (int i = 0; i < p.num_heads_q; ++i) {
                for (int j = 0; j < seqlen_kv; ++j) {
                    bfloat16_t *q_token_ptr = &d.q.index({req_idx, q_token_idx, i, 0});
                    float sum = 0;
                    for (int k = 0; k < p.head_dim; ++k) {
                        float a = q_token_ptr[k];
                        float b = kv_token_ptr[j][k];
                        sum += a * b;
                    }
                    s.index({tid, i, j}) = sum;
                }
            }

            for (int i = 0; i < p.num_heads_q; ++i) {
                float *s_ptr = &s.index({tid, i, 0});
                float mx = s_ptr[0];
                for (int j = 0; j < seqlen_kv; ++j) {
                    mx = std::max(mx, s_ptr[j]);
                }
                float sum = 0;
                for (int j = 0; j < seqlen_kv; ++j) {
                    sum += std::exp((s_ptr[j] - mx) * d.softmax_scale);
                }
                for (int j = 0; j < seqlen_kv; ++j) {
                    s_ptr[j] = std::exp((s_ptr[j] - mx) * d.softmax_scale) / sum;
                }
                softmax_lse.index({req_idx, q_token_idx, i}) =
                    seqlen_kv == 0 ? INFINITY : std::log(sum) + mx * d.softmax_scale;
            }

            for (int i = 0; i < p.num_heads_q; ++i) {
                for (int j = 0; j < p.head_dim_v; ++j) {
                    float sum = 0;
                    for (int k = 0; k < seqlen_kv; ++k) {
                        float a = s.index({tid, i, k});
                        float b = kv_token_ptr[k][j];
                        sum += a * b;
                    }
                    float attn_sink_scale = 1;
                    if (d.attn_sink.has_value()) {
                        float attn_sink_value = d.attn_sink.value().index({i});
                        float lse = softmax_lse.index({req_idx, q_token_idx, i});
                        attn_sink_scale = 1.0 / (1.0 + std::exp(attn_sink_value - lse));
                    }
                    o.index({req_idx, q_token_idx, i, j}) = seqlen_kv == 0 ? 0 : sum * attn_sink_scale;
                }
            }
        }
    });
}