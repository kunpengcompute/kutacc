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

#include "attention/flash_mla/meta.h"
#include "attention/flash_mla/utils.h"
#include "attention/flash_mla/config.h"

namespace kutacc {

namespace flash_mla {

inline void decode_sched(const std::vector<int64_t> &seqlens_kv, int64_t &extra_bytes, FlashMLAMetaHandle meta)
{
    const auto &batch_size = meta->batch_size;
    const auto &kernel_meta = meta->kernel_meta;

    constexpr auto seq_cost = [](int64_t seqlen) {
        return ceil_div<int64_t>(seqlen, 16);
    };

    int64_t sum_seqlen_cost = 0;
    int64_t total_num_blocks = (batch_size - 1) * kernel_meta.fixed_overhead_num_blocks;
    for (int64_t i = 0; i < batch_size; ++i) {
        sum_seqlen_cost += seq_cost(seqlens_kv[i]);
        total_num_blocks += ceil_div<int64_t>(seqlens_kv[i], kernel_meta.kv_block_size);
    }

    auto block_cost = seq_cost(kernel_meta.kv_block_size);
    auto split_cost = kernel_meta.fixed_overhead_num_blocks * block_cost;

    auto check = [&](int64_t payload, bool allow_split) {
        int64_t req_idx = 0;
        int64_t token_idx = 0;
        int64_t split_idx = 0;
        int64_t cum_num_splits = 0;
        meta->num_splits[0] = 0;
        for (int64_t i = 0; i < meta->num_thread_parts; ++i) {
            auto &cur_meta = meta->tile_sched_meta[i];
            cur_meta.begin_req_idx = req_idx;
            cur_meta.begin_token_idx = token_idx;
            cur_meta.begin_split_idx = split_idx;
            int64_t remain_payload = payload;
            while (req_idx < batch_size) {
                auto remain_cost = seq_cost(seqlens_kv[req_idx] - token_idx);
                if (remain_payload >= remain_cost) {
                    cum_num_splits += split_idx + 1;
                    meta->num_splits[req_idx + 1] = cum_num_splits;
                    remain_payload -= remain_cost + kernel_meta.fixed_overhead_num_blocks * block_cost;
                    ++req_idx;
                    token_idx = 0;
                    split_idx = 0;
                } else {
                    if (allow_split && remain_payload >= block_cost) {
                        token_idx += remain_payload / block_cost * kernel_meta.kv_block_size;
                        ++split_idx;
                        remain_payload = 0;
                    }
                    break;
                }
            }
            if (token_idx == 0 && req_idx == 0) {
                return false;
            }
            cur_meta.end_req_idx = token_idx > 0 ? req_idx : req_idx - 1;
            cur_meta.end_token_idx = token_idx > 0 ? token_idx : seqlens_kv[req_idx - 1];
        }
        return req_idx == batch_size;
    };

    int64_t payload_min = ceil_div(sum_seqlen_cost, (int64_t)meta->num_thread_parts) - 1;
    int64_t payload_max = ceil_div(total_num_blocks, (int64_t)meta->num_thread_parts) * block_cost + split_cost;

    while (payload_max - payload_min > 1) {
        int64_t payload = payload_min + (payload_max - payload_min) / 2;
        if (check(payload - split_cost, 1) || check(payload, 0)) {
            payload_max = payload;
        } else {
            payload_min = payload;
        }
    }

    if (!check(payload_max - split_cost, 1)) {
        if (!check(payload_max, 0)) {
            FLASH_MLA_CHECK(false, payload_max);
        }
    }
}

} // namespace flash_mla

} // namespace kutacc