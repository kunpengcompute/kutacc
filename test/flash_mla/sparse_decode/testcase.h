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

#include <random>
#include <kutacc.h>

#include "../utils.h"
#include "../allocator.h"

struct TestcaseParams {
    int64_t batch_size;
    int64_t seqlen_q;
    int64_t num_heads_q;

    int64_t head_dim;
    int64_t head_dim_v;

    int64_t seqlen_kv;
    int64_t page_block_size;
    int64_t topk;

    int64_t extra_seqlen_kv;
    int64_t extra_page_block_size;
    int64_t extra_topk;

    bool have_topk_length;
    bool have_extra_topk_length;

    bool have_attn_sink;

    bool is_indices_shuffle;

    bool check_correctness;
    int64_t num_runs;
};

struct TestcaseData {
    kutacc::Tensor<bfloat16_t, 4> q;                             // [batch_size, seqlen_q, num_heads_q, head_dim]

    kutacc::Tensor<bfloat16_t, 3> kvcache;                       // [num_blocks, page_block_size, head_dim]
    kutacc::Tensor<int, 3> indices;                              // [batch_size, seqlen_q, topk]
    std::optional<kutacc::Tensor<int, 1>> topk_length;           // [batch_size]

    std::optional<kutacc::Tensor<bfloat16_t, 3>> extra_kvcache;  // [extra_num_blocks, extra_page_block_size, head_dim]
    std::optional<kutacc::Tensor<int, 3>> extra_indices;         // [batch_size, seqlen_q, extra_topk]
    std::optional<kutacc::Tensor<int, 1>> extra_topk_length;     // [batch_size]

    std::optional<kutacc::Tensor<float, 1>> attn_sink;           // [num_heads_q]
    float softmax_scale;
};

template <typename T>
auto constant_generator(const T &x)
{
    return [&]() -> T { return x; };
}

inline void generate_kvcache(
    const TestcaseParams &p,
    int64_t seqlen_kv,
    int64_t page_block_size,
    int64_t topk,
    bool have_topk_length,
    kutacc::Tensor<bfloat16_t, 3> &kvcache,
    kutacc::Tensor<int, 3> &indices,
    std::optional<kutacc::Tensor<int, 1>> &topk_length,
    Allocator &allocator,
    std::mt19937 &rng
)
{
    std::normal_distribution<> normal_distr;
    std::uniform_int_distribution<> topk_distr(1, topk);
    const auto gen_kvcache = [&]() -> bfloat16_t {
        return std::min(std::max(-1.0, normal_distr(rng) / 10), 1.0);
    };
    const auto gen_topk_length = [&]() -> int {
        return topk_distr(rng);
    };
    int64_t num_blocks = p.batch_size * ceil_div(seqlen_kv, page_block_size);
    kvcache = generate_tensor<bfloat16_t, 3>(allocator,
        {num_blocks, page_block_size, p.head_dim}, gen_kvcache, HUGE_PAGE_SIZE);
    indices = generate_tensor<int, 3>(allocator, {p.batch_size, p.seqlen_q, topk}, constant_generator<int>(-1));
    if (have_topk_length) {
        topk_length = generate_tensor<int, 1>(allocator, {p.batch_size}, gen_topk_length);
    } else {
        topk_length = std::nullopt;
    }
    std::vector<int> perm(seqlen_kv);
    for (int i = 0; i < seqlen_kv; ++i) {
        perm[i] = i;
    }
    for (int i = 0; i < p.batch_size; ++i) {
        for (int j = 0; j < p.seqlen_q; ++j) {
            if (p.is_indices_shuffle) {
                std::shuffle(perm.begin(), perm.end(), rng);
            }
            int64_t actual_topk = have_topk_length ? topk_length.value().data_ptr()[i] : topk;
            actual_topk = std::min(actual_topk, seqlen_kv);
            auto cur_indices = &indices.index({i, j, 0});
            for (int k = 0; k < actual_topk; ++k) {
                cur_indices[k] = i * ceil(seqlen_kv, page_block_size) + perm[k];
            }
            std::sort(cur_indices, cur_indices + actual_topk);
        }
    }
}

inline TestcaseData generate_testcase_data(const TestcaseParams &p, Allocator &allocator, std::mt19937 &rng)
{
    std::normal_distribution<> normal_distr;
    const auto gen_q = [&]() -> bfloat16_t {
        return std::min(std::max(-1.0, normal_distr(rng)), 1.0);
    };
    const auto gen_attn_sink = [&]() -> bfloat16_t {
        auto x = normal_distr(rng);
        if (x > 0.5) {
            return INFINITY;
        } else if (x < -0.5) {
            return -INFINITY;
        } else {
            return x;
        }
    };

    kutacc::Tensor<bfloat16_t, 4> q;
    kutacc::Tensor<bfloat16_t, 3> kvcache;
    kutacc::Tensor<int, 3> indices;
    std::optional<kutacc::Tensor<int, 1>> topk_length;
    std::optional<kutacc::Tensor<float, 1>> attn_sink;

    q = generate_tensor<bfloat16_t, 4>(allocator, {p.batch_size, p.seqlen_q, p.num_heads_q, p.head_dim}, gen_q);

    if (p.have_attn_sink) {
        attn_sink = generate_tensor<float, 1>(allocator, {p.num_heads_q}, gen_attn_sink);
    } else {
        attn_sink = std::nullopt;
    }

    float softmax_scale = std::pow(p.head_dim, -0.55);

    generate_kvcache(p, p.seqlen_kv, p.page_block_size, p.topk, p.have_topk_length, kvcache,
        indices, topk_length, allocator, rng);

    if (p.extra_topk) {
        kutacc::Tensor<bfloat16_t, 3> extra_kvcache;
        kutacc::Tensor<int, 3> extra_indices;
        std::optional<kutacc::Tensor<int, 1>> extra_topk_length;
        generate_kvcache(p, p.extra_seqlen_kv, p.extra_page_block_size, p.extra_topk, p.have_extra_topk_length,
            extra_kvcache, extra_indices, extra_topk_length, allocator, rng);
        return {q, kvcache, indices, topk_length, extra_kvcache,
            extra_indices, extra_topk_length, attn_sink, softmax_scale};
    } else {
        return {q, kvcache, indices, topk_length, std::nullopt, std::nullopt,
            std::nullopt, attn_sink, softmax_scale};
    }
}

inline std::pair<int64_t, int64_t> count_flop_and_mem_vol(const TestcaseParams &p, const TestcaseData &d)
{
    int64_t num_kv_tokens = 0;
    for (int i = 0; i < p.batch_size; ++i) {
        num_kv_tokens +=
            (d.topk_length.has_value() ? d.topk_length.value().index({i}) : p.topk) +
                (d.extra_topk_length.has_value() ? d.extra_topk_length.value().index({i}) : p.extra_topk);
    }
    num_kv_tokens *= p.seqlen_q;
    int64_t flops = 2 * p.num_heads_q * num_kv_tokens * (p.head_dim + p.head_dim_v);
    int64_t mem_vol =
        2 * p.batch_size * p.seqlen_q * p.head_dim +
        2 * num_kv_tokens * p.head_dim +
        2 * p.batch_size * p.seqlen_q * p.head_dim_v;
    return {flops, mem_vol};
}