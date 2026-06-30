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

#include <type_traits>

#include "kutacc.h"

#include "attention/flash_mla/meta.h"
#include "attention/flash_mla/utils.h"
#include "attention/flash_mla/decode/traits.h"
#include "attention/flash_mla/decode/kernels/combine.h"
#include "attention/flash_mla/decode/kernels/dense_mla_h32.h"
#include "attention/flash_mla/decode/kernels/dense_mla_h64.h"
#include "attention/flash_mla/decode/kernels/dense_mla_h64_kvpacked.h"
#include "attention/flash_mla/decode/kernels/sparse_mla_h32.h"
#include "attention/flash_mla/decode/kernels/sparse_mla_h64.h"

#include "params.h"

namespace kutacc {

namespace flash_mla {

template <typename KernelTraits>
constexpr auto get_dense_mla_kernel()
{
    constexpr auto q_block_size = KernelTraits::q_block_size;
    constexpr auto is_kv_packed = KernelTraits::is_kv_packed;
    if constexpr (q_block_size == 32 && !is_kv_packed) {
        return dense_mla_h32_kernel<KernelTraits>;
    } else if constexpr (q_block_size == 64 && !is_kv_packed) {
        return dense_mla_h64_kernel<KernelTraits>;
    } else if constexpr (q_block_size == 64 && is_kv_packed) {
        return dense_mla_h64_kvpacked_kernel<KernelTraits>;
    } else {
        static_assert(0);
    }
}

template <typename KernelTraits>
constexpr auto get_sparse_mla_kernel()
{
    constexpr auto q_block_size = KernelTraits::q_block_size;
    if constexpr (q_block_size == 32) {
        return sparse_mla_h32_kernel<KernelTraits>;
    } else if constexpr (q_block_size == 64) {
        return sparse_mla_h64_kernel<KernelTraits>;
    } else {
        static_assert(0);
    }
}

template <typename DecodeParams, typename KernelTraits>
void splitkv_mla_kernel(const DecodeParams &params)
{
    constexpr auto q_block_size = KernelTraits::q_block_size;

    const auto num_q_blocks = ceil_div<int64_t>(params.num_heads_q, q_block_size);
    const auto num_tasks = params.meta->num_thread_parts * params.seqlen_q * num_q_blocks;

    kutacc::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
        double *thread_time_record = nullptr;
        for (int64_t task_idx = begin; task_idx < end; ++task_idx) {
            const auto [part_idx, q_token_idx, q_block_idx] =
                get_indices<3>(task_idx, {params.meta->num_thread_parts, params.seqlen_q, num_q_blocks});

            const auto &cur_meta = params.meta->tile_sched_meta[part_idx];

            for (int64_t req_idx = cur_meta.begin_req_idx; req_idx <= cur_meta.end_req_idx; ++req_idx) {
                const bool is_begin = req_idx == cur_meta.begin_req_idx;
                const int64_t split_idx = is_begin ? cur_meta.begin_split_idx : 0;
                
                int64_t seqlen_kv;
                if constexpr (std::is_same_v<DecodeParams, DenseDecodeParams>) {
                    seqlen_kv = params.seqlens_kv->index({req_idx});
                } else {
                    seqlen_kv =
                        (params.topk_length->has_value() ? params.topk_length->value().index({req_idx}) :
                            params.topk) +
                        (params.extra_topk_length->has_value() ? params.extra_topk_length->value().index({req_idx}) :
                            params.extra_topk);
                }
                const int64_t begin_kv_token_idx = is_begin ? cur_meta.begin_token_idx : 0;
                const int64_t end_kv_token_idx =
                    req_idx == cur_meta.end_req_idx ? cur_meta.end_token_idx : seqlen_kv;
                if constexpr (std::is_same_v<DecodeParams, DenseDecodeParams>) {
                    get_dense_mla_kernel<KernelTraits>()(params, req_idx, q_token_idx, q_block_idx, split_idx,
                        seqlen_kv, begin_kv_token_idx, end_kv_token_idx, thread_time_record);
                } else {
                    get_sparse_mla_kernel<KernelTraits>()(params, req_idx, q_token_idx, q_block_idx, split_idx,
                        seqlen_kv, begin_kv_token_idx, end_kv_token_idx, thread_time_record);
                }
            }
        }
    });
}

template <typename DecodeParams, typename KernelTraits>
void decode_splitkv_mla_launch(const DecodeParams &params)
{
    splitkv_mla_kernel<DecodeParams, KernelTraits>(params);
    splitkv_mla_combine_launch<DecodeParams, KernelTraits>(params);
}

inline void dense_decode_splitkv_mla(const DenseDecodeParams &params)
{
    const auto &kernel_meta = params.meta->kernel_meta;

    FLASH_MLA_CHECK(kernel_meta.kv_block_size == 64, "");
    FLASH_MLA_CHECK(kernel_meta.head_dim == 576, "");
    FLASH_MLA_CHECK(kernel_meta.head_dim_v == 512, "");

    const auto q_block_size = kernel_meta.q_block_size;
    const auto is_kv_packed = kernel_meta.is_kv_packed;

    FLASH_MLA_BOOL_SWITCH(params.is_casual, is_casual, [&] {
        if (q_block_size == 32 && !is_kv_packed) {
            decode_splitkv_mla_launch<
                DenseDecodeParams, KernelTraitsTemplate<32, 64, 576, 512, is_casual, false>>(params);
        } else if (q_block_size == 64 && !is_kv_packed) {
            decode_splitkv_mla_launch<
                DenseDecodeParams, KernelTraitsTemplate<64, 64, 576, 512, is_casual, false>>(params);
        } else if (q_block_size == 64 && is_kv_packed) {
            decode_splitkv_mla_launch<
                DenseDecodeParams, KernelTraitsTemplate<64, 64, 576, 512, is_casual, true>>(params);
        } else {
            FLASH_MLA_CHECK(false, "");
        }
    });
}

inline void sparse_decode_splitkv_mla(const SparseDecodeParams &params)
{
    const auto &kernel_meta = params.meta->kernel_meta;

    FLASH_MLA_CHECK(kernel_meta.kv_block_size == 64, "");
    FLASH_MLA_CHECK(kernel_meta.head_dim_v == 512, "");
    FLASH_MLA_CHECK(!kernel_meta.is_kv_packed, "");

    const auto q_block_size = kernel_meta.q_block_size;
    const auto head_dim = kernel_meta.head_dim;

    if (q_block_size == 64 && head_dim == 576) {
        decode_splitkv_mla_launch<SparseDecodeParams, KernelTraitsTemplate<64, 64, 576, 512, false, false>>(params);
    } else if (q_block_size == 64 && head_dim == 512) {
        decode_splitkv_mla_launch<SparseDecodeParams, KernelTraitsTemplate<64, 64, 512, 512, false, false>>(params);
    } else if (q_block_size == 32 && head_dim == 576) {
        decode_splitkv_mla_launch<SparseDecodeParams, KernelTraitsTemplate<32, 64, 576, 512, false, false>>(params);
    } else if (q_block_size == 32 && head_dim == 512) {
        decode_splitkv_mla_launch<SparseDecodeParams, KernelTraitsTemplate<32, 64, 512, 512, false, false>>(params);
    } else {
        FLASH_MLA_CHECK(false, "");
    }
}

} // namespace flash_mla

} // namespace kutacc