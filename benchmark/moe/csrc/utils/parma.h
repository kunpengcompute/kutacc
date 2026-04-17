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

#include <cstdint>
#include <string>
#include "tensor.h"
struct Params_latency {
    int64_t dim = 7168;
    int64_t inter_dim = 18432;
    int64_t moe_inter_dim = 2048;
    int64_t n_layers = 61;
    int64_t n_dense_layers = 3;
    int64_t n_heads = 128;
    int64_t n_tokens = 128;
    int64_t n_routed_experts = 256;
    int64_t n_shared_experts = 1;
    int64_t n_activated_experts = 8;
    int64_t n_expert_groups = 8;
    int64_t n_limited_groups = 4;
    int64_t ppn = 16;

    int64_t world_size;
    int64_t local_size;
    int64_t world_rank;
    int64_t local_rank;
    int64_t max_token;
    int64_t moe_ep;
    int64_t moe_dp;
    int64_t moe_tp;
    int64_t num_max_dispatch_tokens_per_rank;
    int64_t max_seq_len;

    // low_latency
    utils::Tensor parallel_sizes;
    utils::Tensor dispatch_send_buf;
    utils::Tensor topk_idx;
    utils::Tensor combine_send_buf;
    utils::Tensor recv_token_ids_buf;
    utils::Tensor recv_token_ids;
    utils::Tensor recv_experts_offset;
};

extern Params_latency params;