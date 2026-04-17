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
#include <mpi.h>
#include <arm_bf16.h>

#include "kutacc.h"
#include "../utils/tensor.h"

namespace low_latency {

extern utils::Tensor packed_recv_x;
extern utils::Tensor recv_src_info;
extern utils::Tensor combined_x;

inline MPI_Comm intra_comm;
inline int16_t *src_info;
inline int my_rank;
inline int num_ranks;
inline bfloat16_t *tmpx_for_sum;
inline int64_t num_local_experts;
inline int64_t recv_src_info_size;
inline int64_t recv_src_info_conut;
inline int64_t recv_src_info_ep_bias;
inline int64_t packed_recv_x_size;
inline int64_t disbuf_size;
inline void *disbuf_baseptr;
inline kutacc::kurmcl_conn_info_h ds_conn_info;

utils::Tensor create_dispatch_send_buf(void* disp_baseptr, utils::ScalarType dtype, const std::vector<int64_t> &sizes);
utils::Tensor create_combine_send_buf(void* comb_baseptr, void* baseptr_stay,
    utils::ScalarType dtype, const std::vector<int64_t> &sizes);

// Initialize dispatch resources, only call once before inference
void dispatch_init(const utils::Tensor &x, int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank,
    int64_t hidden);

/*
dispatch_send params
- x: int8 [num_tokens, hidden_size]
    x[i][:] = Token i to be sent by the current process.
- topk_idx: int16 [num_tokens, num_topk]
    topk_idx[i][j] = The token i of the current process needs to be sent to the expert j.
- num_max_dispatch_tokens_per_rank: int64
    Maximum number of tokens sent by each process.
- num_experts: int64
    Total number of experts.
- parallel_sizes: int16 [3](EP/DP/E-TP)
    Degree of parallelism.
*/
void dispatch_send(const utils::Tensor &x, const utils::Tensor &topk_idx, int64_t num_max_dispatch_tokens_per_rank,
    int64_t num_experts, const utils::Tensor &parallel_sizes, int64_t batch_id);

/*
dispatch_recv params
- num_experts: int64
    Total number of experts.
- num_max_dispatch_tokens_per_rank: int64
    Maximum number of tokens sent by each process.

- packed_recv_x: int8 [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden_size]
    packed_recv_x[i][j,k][:] = Current expert i receives tokens, which from rank j and token k.
- recv_src_info: int16 [num_local_experts, num_ranks * (num_max_dispatch_tokens_per_rank + 1)]
    recv_src_info[i][j,0] = Number of tokens received by expert i form rank j.
    recv_src_info[i][j,k] = The token received by expert i is the kth token of process j.
*/
void dispatch_recv(int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t batch_id);

// Release dispatch resources, only call once after inference
void dispatch_finalize();

// Initialize combine resources, only call once before inference
void combine_init(const utils::Tensor &new_packed_recv_x, int64_t num_tokens, int64_t num_experts, int64_t num_topk,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden);

/*
combine_send params
- x: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden_size]
    x[i][j,k][:] = After dispatch and compute, current expert i receives tokens, which from rank j and token k.
- src_info: int16 [num_local_experts, num_ranks * (num_max_dispatch_tokens_per_rank + 1)] = dispatch's output
    src_info[i][j,0] = Number of tokens received by expert i form rank j.
    src_info[i][j,k] = The token received by expert i is the kth token of process j.
- num_tokens: int64
    Total number of tokens.
- num_max_dispatch_tokens_per_rank: int64
    Maximum number of tokens sent by each process.
- num_experts: int64
    Total number of experts.
- parallel_sizes: int16 [3](EP/DP/E-TP)
    Degree of parallelism.
*/
void combine_send(const utils::Tensor &x, const utils::Tensor &src_info, int64_t num_tokens,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, const utils::Tensor &parallel_sizes,
    int64_t batch_id);

/*
combine_recv params
- topk_idx: int16 [num_tokens, num_topk]
    topk_idx[i][j] = The token i of the current process needs to be sent to the expert j.
- topk_weights: fp32 [num_tokens, num_topk]
    topk_weights[i][j] = Expert weights corresponding to topk_idx[i][j].
- src_info: int16 [num_local_experts, num_ranks * (num_max_dispatch_tokens_per_rank + 1)] = dispatch's output
    src_info[i][j,0] = Number of tokens received by expert i form rank j.
    src_info[i][j,k] = The token received by expert i is the kth token of process j.
- num_max_dispatch_tokens_per_rank: int64
    Maximum number of tokens sent by each process.
- num_experts: int64
    Total number of experts.
- parallel_sizes: int16 [3](EP/DP/E-TP)
    Degree of parallelism.
- hidden: int64
    Hidden size.

- combined_x: bf16 [num_tokens, hidden]
    combined_x[i][:] = The final result after combine about token i.
*/
void combine_recv(utils::Tensor &topk_idx, utils::Tensor &topk_weights, utils::Tensor &src_info,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, utils::Tensor &parallel_sizes, int64_t hidden,
    int64_t batch_id);

// Release combine resources, only call once after inference
void combine_finalize();

void low_latency_barrier();

}  // namespace low_latency
