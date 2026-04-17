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
#include <arm_bf16.h>
#include <arm_fp16.h>
#include <mpi.h>

#include "kutacc.h"
#include "../utils/tensor.h"
#include "../utils/memory_pool.h"
#include "low_latency.h"

namespace low_latency {

utils::Tensor packed_recv_x;
utils::Tensor recv_src_info;
utils::Tensor combined_x;

static int oob_allgather_callback(const void *sendbuf, void *recvbuf, int size, int group,
                                  kutacc::kurmcl_datatype_t datatype)
{
    switch (datatype) {
        case kutacc::KURMCL_DATATYPE_CHAR:
            return MPI_Allgather(sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_INT:
            return MPI_Allgather(sendbuf, size, MPI_INT, recvbuf, size, MPI_INT, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_LONG:
            return MPI_Allgather(sendbuf, size, MPI_LONG, recvbuf, size, MPI_LONG, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_FLOAT:
            return MPI_Allgather(sendbuf, size, MPI_FLOAT, recvbuf, size, MPI_FLOAT, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_DOUBLE:
            return MPI_Allgather(sendbuf, size, MPI_DOUBLE, recvbuf, size, MPI_DOUBLE, (MPI_Comm)group);
        default:
            printf("not support datatype");
            return -1;
    }
}
static int oob_alltoall_callback(const void *sendbuf, int sendcount, kutacc::kurmcl_datatype_t sendtype,
                                 void *recvbuf, int recvcount, kutacc::kurmcl_datatype_t recvtype, int group)
{
    switch (sendtype) {
        case kutacc::KURMCL_DATATYPE_CHAR:
            return MPI_Alltoall(sendbuf, sendcount, MPI_CHAR, recvbuf, recvcount, MPI_CHAR, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_INT:
            return MPI_Alltoall(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_LONG:
            return MPI_Alltoall(sendbuf, sendcount, MPI_LONG, recvbuf, recvcount, MPI_LONG, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_FLOAT:
            return MPI_Alltoall(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, (MPI_Comm)group);
        case kutacc::KURMCL_DATATYPE_DOUBLE:
            return MPI_Alltoall(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, (MPI_Comm)group);
        default:
            printf("not support datatype");
            return -1;
    }
}
static int oob_barrier_callback(int group)
{
    return MPI_Barrier((MPI_Comm)group);
}

utils::Tensor create_dispatch_send_buf(void *disp_baseptr, utils::ScalarType dtype, const std::vector<int64_t> &sizes)
{
    int64_t numel = 1;
    for (int64_t i = 0; i < sizes.size(); i++) {
        numel *= sizes[i];
    }
    int64_t size = numel * utils::elementSize(dtype);
    memset(disp_baseptr, 0, size);
    return utils::Tensor::from_blob(dtype, sizes, disp_baseptr);
}

utils::Tensor create_combine_send_buf(void *comb_baseptr, void *baseptr_stay, utils::ScalarType dtype,
                                      const std::vector<int64_t> &sizes)
{
    int64_t numel = 1;
    for (int64_t i = 0; i < sizes.size(); i++) {
        numel *= sizes[i];
    }
    int64_t size = numel * utils::elementSize(dtype);
    memset(comb_baseptr, 0, size);
    disbuf_baseptr = (char *)baseptr_stay + size / 2;
    tmpx_for_sum = (bfloat16_t *)baseptr_stay;
    return utils::Tensor::from_blob(dtype, sizes, comb_baseptr);
}

void dispatch_init(const utils::Tensor &x, int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank,
                   int64_t hidden)
{
    kutacc::kurmcl_oob_cb_t oob_cbs;
    kutacc::kurmcl_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;
    oob_cbs_h->oob_alltoall = oob_alltoall_callback;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    kutacc::kurmcl_comm_create(num_ranks, my_rank, oob_cbs_h, MPI_COMM_WORLD, &ds_conn_info);

    int64_t num_tokens = static_cast<int>(x.size(0));
    uint8_t *x_data = x.data_ptr<uint8_t>();
    num_local_experts = num_experts / num_ranks == 0 ? 1 : num_experts / num_ranks;
    packed_recv_x_size =
        num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden * sizeof(char);
    recv_src_info_conut =
        num_local_experts * num_ranks * (num_max_dispatch_tokens_per_rank + 1);
    recv_src_info_size = recv_src_info_conut * sizeof(int16_t);
    recv_src_info_ep_bias = num_ranks * (num_max_dispatch_tokens_per_rank + 1);
    disbuf_size = packed_recv_x_size + recv_src_info_size;
    packed_recv_x = utils::Tensor::from_blob(utils::ScalarType::UInt8,
        {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
        static_cast<uint8_t *>(disbuf_baseptr));
    recv_src_info = utils::Tensor::from_blob(utils::ScalarType::Int16,
        {num_local_experts, num_ranks * (num_max_dispatch_tokens_per_rank + 1)},
        static_cast<uint8_t *>(disbuf_baseptr) + packed_recv_x_size);
    src_info = recv_src_info.data_ptr<int16_t>() + recv_src_info_conut;
    memset(packed_recv_x.ptr, 0, packed_recv_x.numel() * utils::elementSize(packed_recv_x.dtype()));
    uint8_t *packed_recv_x_data = packed_recv_x.data_ptr<uint8_t>();
    int16_t *recv_src_info_data = recv_src_info.data_ptr<int16_t>();
    kutacc::moe_dispatch_init(
        x_data, packed_recv_x_data, recv_src_info_data, num_experts,
        num_max_dispatch_tokens_per_rank, hidden, num_tokens, num_ranks,
        my_rank, src_info, disbuf_baseptr, ds_conn_info);
}

void dispatch_send(const utils::Tensor &x, const utils::Tensor &topk_idx, int64_t num_max_dispatch_tokens_per_rank,
                   int64_t num_experts, const utils::Tensor &parallel_sizes, int64_t batch_id)
{
    uint8_t *x_data = x.data_ptr<uint8_t>();
    int16_t *topk_idx_data = topk_idx.data_ptr<int16_t>();
    int16_t *parallel_sizes_data = parallel_sizes.data_ptr<int16_t>();
    uint8_t *packed_recv_x_data = packed_recv_x.data_ptr<uint8_t>();
    int16_t *recv_src_info_data = recv_src_info.data_ptr<int16_t>();
    int64_t num_tokens = static_cast<int>(x.size(0));
    int64_t hidden = static_cast<int>(x.size(1));
    int64_t num_topk = static_cast<int>(topk_idx.size(1));
    kutacc::moe_dispatch_send(x_data, topk_idx_data, num_tokens, num_topk, num_experts,
        num_max_dispatch_tokens_per_rank, hidden, parallel_sizes_data, packed_recv_x_data, recv_src_info_data,
        batch_id);
}

void dispatch_recv(int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t batch_id)
{
    int16_t *recv_src_info_data = recv_src_info.data_ptr<int16_t>();
    kutacc::moe_dispatch_recv(recv_src_info_data, num_experts, num_max_dispatch_tokens_per_rank, batch_id);
}

void dispatch_finalize()
{
    kutacc::moe_dispatch_finalize();
}

void combine_init(const utils::Tensor &new_packed_recv_x, int64_t num_tokens, int64_t num_experts, int64_t num_topk,
                  int64_t num_max_dispatch_tokens_per_rank, int64_t hidden)
{
    int node_nums = num_ranks / 16;
    int intra_size = 16;
    bfloat16_t *base_ptr = (bfloat16_t *)utils::Tensor::alloc_from(utils::ScalarType::BFloat16,
        {num_tokens / node_nums * hidden}, utils::shm_available).ptr;
    uint8_t *meta_ptr =
        (uint8_t *)utils::Tensor::alloc_from(utils::ScalarType::UInt8, {16}, utils::shm_available).ptr;
    int mynode_id = utils::global_rank / 16;
    MPI_Comm_split(MPI_COMM_WORLD, mynode_id, utils::global_rank, &intra_comm);
    int local_size = intra_size;
    int local_rank = utils::group_rank;
    std::vector<bfloat16_t *> group_ptr;
    std::vector<uint8_t *> meta_group_ptr;
    group_ptr.resize(local_size, nullptr);
    meta_group_ptr.resize(local_size, nullptr);
    for (int i = 0; i < local_size; ++i) {
        if (i != local_rank) {
            utils::get_peer_shm_baseptr(i, base_ptr, (void **)&group_ptr[i]);
            utils::get_peer_shm_baseptr(i, meta_ptr, (void **)&meta_group_ptr[i]);
        } else {
            group_ptr[i] = base_ptr;
            meta_group_ptr[i] = meta_ptr;
        }
    }
    bfloat16_t *new_packed_recv_x_data = new_packed_recv_x.data_ptr<bfloat16_t>();
    kutacc::moe_combine_init(new_packed_recv_x_data, num_tokens, num_experts, num_topk,
        num_max_dispatch_tokens_per_rank, hidden, std::move(group_ptr), std::move(meta_group_ptr),
        local_size, local_rank, tmpx_for_sum);
    combined_x = utils::Tensor::from_blob(utils::ScalarType::BFloat16, {num_tokens / node_nums, hidden},
        base_ptr);
}

void combine_send(const utils::Tensor &x, const utils::Tensor &src_info, int64_t num_tokens,
                  int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, const utils::Tensor &parallel_sizes,
                  int64_t batch_id)
{
    bfloat16_t *x_data = x.data_ptr<bfloat16_t>();
    int16_t *src_info_data = src_info.data_ptr<int16_t>();
    int16_t *parallel_sizes_data = parallel_sizes.data_ptr<int16_t>();
    int64_t hidden = static_cast<int>(x.size(2));
    kutacc::moe_combine_send(x_data, src_info_data, num_tokens, num_max_dispatch_tokens_per_rank, num_experts,
        hidden, parallel_sizes_data, batch_id);
}

void combine_recv(utils::Tensor &topk_idx, utils::Tensor &topk_weights, utils::Tensor &src_info,
                  int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, utils::Tensor &parallel_sizes,
                  int64_t hidden, int64_t batch_id)
{
    int16_t *topk_idx_data = topk_idx.data_ptr<int16_t>();
    float *topk_weights_data = topk_weights.data_ptr<float>();
    int64_t num_tokens = static_cast<int>(topk_idx.size(0));
    int64_t num_topk = static_cast<int>(topk_idx.size(1));
    bfloat16_t *combined_x_data = combined_x.data_ptr<bfloat16_t>();
    kutacc::moe_combine_recv(combined_x_data, topk_idx_data, topk_weights_data, num_tokens, num_experts,
        num_max_dispatch_tokens_per_rank, num_topk, hidden, batch_id, utils::kupl_win);
}

void combine_finalize()
{
    kutacc::moe_combine_finalize();
}

void low_latency_barrier()
{
    kutacc::kurmcl_barrier(ds_conn_info, num_ranks, my_rank);
}

}