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

typedef struct moe_comm {
    kutacc::kurmcl_conn_info_h global_ds_conn_info;
    kutacc::kurmcl_conn_info_h local_ds_conn_info;
    int64_t num_max_dispatch_tokens_per_rank;
} *moe_comm_h, moe_comm_t;

extern utils::Tensor packed_recv_x;
extern utils::Tensor recv_src_info;
extern utils::Tensor recv_src_info_bak;
extern utils::Tensor combined_x;

inline MPI_Comm intra_comm;
inline int16_t* src_info;
inline int my_rank;
inline int num_ranks;
inline bfloat16_t* tmpx_for_sum;
inline int64_t num_local_experts;
inline int64_t recv_src_info_size;
inline int64_t recv_src_info_conut;
inline int64_t recv_src_info_ep_bias;
inline int64_t packed_recv_x_size;
inline int64_t disbuf_size;
inline void* disbuf_baseptr;
inline kupl_shm_win_h kupl_win;
inline kutacc::kurmcl_conn_info_h ds_conn_info;

utils::Tensor create_dispatch_send_buf(void *disp_baseptr, void *disp_recvptr, utils::ScalarType dtype,
    const std::vector<int64_t> &sizes);
utils::Tensor create_combine_send_buf(void *comb_baseptr, void *comb_recvptr, utils::ScalarType dtype,
    const std::vector<int64_t> &sizes);

// Initialize dispatch\combine resources, only call once before inference
void moe_comm_create(const utils::Tensor& dispatch_send_buf, const utils::Tensor& combine_send_buf,
    int64_t redundant_n_routed_experts, int64_t n_activated_experts, int64_t num_max_dispatch_tokens_per_rank,
    const utils::Tensor& parallel_policy, MPI_Comm sub_comm, moe_comm_h ds_moe_comm);

/*
dispatch_send params
- x: int8 [num_tokens, hidden_size]
    x[i][:] = Token i to be sent by the current process.
- topk_idx: int16 [num_tokens, num_topk]
    topk_idx[i][j] = The token i of the current process needs to be sent to the expert j.
- num_experts: int64
    Total number of experts.
- parallel_policy: int16 [3](EP/DP/E-TP)
    Degree of parallelism.
- batch_id: int16
    reserve for double batch.
- ds_moe_comm: struct
    local moe communicator handler.
*/
void dispatch_send(const utils::Tensor& x, const utils::Tensor& topk_idx, int64_t num_experts,
    const utils::Tensor& parallel_policy, int64_t batch_id, moe_comm_h ds_moe_comm);

/*
dispatch_recv params
- batch_id: int16
    reserve for double batch.
- ds_moe_comm: struct
    local moe communicator handler.
*/
void dispatch_recv(int64_t batch_id, moe_comm_h ds_moe_comm);

/*
combine_send params
- x: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden_size]
    x[i][j,k][:] = After dispatch and compute, current expert i receives tokens, which from rank j and token k.
- src_info: int16 [num_local_experts, num_ranks * (num_max_dispatch_tokens_per_rank + 1)] = dispatch's output
    src_info[i][j,0] = Number of tokens received by expert i form rank j.
    src_info[i][j,k] = The token received by expert i is the kth token of process j.
- num_experts: int64
    Total number of experts.
- parallel_policy: int16 [3](EP/DP/E-TP)
    Degree of parallelism.
- batch_id: int16
    reserve for double batch.
- ds_moe_comm: struct
    local moe communicator handler.
*/
void combine_send(const utils::Tensor& x, const utils::Tensor& src_info, int64_t num_experts,
    const utils::Tensor& parallel_policy, int64_t batch_id, moe_comm_h ds_moe_comm,
    utils::Tensor& topk_idx, utils::Tensor& topk_weights);

/*
combine_recv params
- topk_idx: int16 [num_tokens, num_topk]
    topk_idx[i][j] = The token i of the current process needs to be sent to the expert j.
- topk_weights: fp32 [num_tokens, num_topk]
    topk_weights[i][j] = Expert weights corresponding to topk_idx[i][j].
- parallel_policy: int16 [3](EP/DP/E-TP)
    Degree of parallelism.
- hidden: int64
    Hidden size.
- batch_id: int16
    reserve for double batch.
- ds_moe_comm: struct
    local moe communicator handler.
*/
void combine_recv(utils::Tensor& topk_idx, utils::Tensor& topk_weights, utils::Tensor& parallel_policy, int64_t hidden,
    int64_t batch_id, moe_comm_h ds_moe_comm);

// Release dispatch\combine resources, only call once after inference
void moe_comm_destroy();

void pp_init(utils::Tensor &x, moe_comm_h ds_moe_comm);

void pp_put(int64_t dest_rank, int64_t send_offset, int64_t size, moe_comm_h ds_moe_comm);

void pp_recv(int64_t src_rank, moe_comm_h ds_moe_comm);

void low_latency_barrier(moe_comm_h ds_moe_comm);

}  // namespace low_latency
