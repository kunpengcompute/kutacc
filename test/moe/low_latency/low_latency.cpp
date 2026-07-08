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
#include "../utils/parma.h"
#include "../utils/memory_pool.h"
#include "low_latency.h"

namespace low_latency {

utils::Tensor packed_recv_x;
utils::Tensor recv_src_info;
utils::Tensor recv_src_info_bak;
utils::Tensor combined_x;
int call_times = 0;

static int oob_allgather_callback(
    const void *sendbuf, void *recvbuf, int size, void *group, kutacc::kurmcl_datatype_t datatype)
{
    int *group_ = (int *)group;
    if (sendbuf == nullptr) {
        sendbuf = (void *)(-1);
    }
    switch (datatype) {
        case kutacc::KURMCL_DATATYPE_CHAR:
            return MPI_Allgather(sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_INT:
            return MPI_Allgather(sendbuf, size, MPI_INT, recvbuf, size, MPI_INT, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_LONG:
            return MPI_Allgather(sendbuf, size, MPI_LONG, recvbuf, size, MPI_LONG, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_FLOAT:
            return MPI_Allgather(sendbuf, size, MPI_FLOAT, recvbuf, size, MPI_FLOAT, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_DOUBLE:
            return MPI_Allgather(sendbuf, size, MPI_DOUBLE, recvbuf, size, MPI_DOUBLE, (MPI_Comm)(*group_));
        default:
            printf("not support datatype");
            return KUPL_ERROR;
    }
}

static int oob_alltoall_callback(const void* sendbuf, int sendcount, kutacc::kurmcl_datatype_t sendtype, void* recvbuf,
    int recvcount, [[maybe_unused]] kutacc::kurmcl_datatype_t recvtype, void *group)
{
    int *group_ = (int *)group;
    if (sendbuf == nullptr) {
        sendbuf = (void *)(-1);
    }
    switch (sendtype) {
        case kutacc::KURMCL_DATATYPE_CHAR:
            return MPI_Alltoall(sendbuf, sendcount, MPI_CHAR, recvbuf, recvcount, MPI_CHAR, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_INT:
            return MPI_Alltoall(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_LONG:
            return MPI_Alltoall(sendbuf, sendcount, MPI_LONG, recvbuf, recvcount, MPI_LONG, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_FLOAT:
            return MPI_Alltoall(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, (MPI_Comm)(*group_));
        case kutacc::KURMCL_DATATYPE_DOUBLE:
            return MPI_Alltoall(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, (MPI_Comm)(*group_));
        default:
            printf("not support datatype");
            return -1;
    }
}
static int oob_barrier_callback(void *group)
{
    int *group_ = (int *)group;
    return MPI_Barrier((MPI_Comm)(*group_));
}

utils::Tensor create_dispatch_send_buf(void *disp_baseptr, void *disp_recvptr, utils::ScalarType dtype,
    const std::vector<int64_t> &sizes)
{
    int64_t numel = 1;
    for (size_t i = 0; i < sizes.size(); i++) {
        numel *= sizes[i];
    }
    int64_t size = numel * utils::elementSize(dtype);
    memset(disp_baseptr, 0, size);
    disbuf_baseptr = (char *)disp_recvptr;
    return utils::Tensor::from_blob(dtype, sizes, disp_baseptr);
}

utils::Tensor create_combine_send_buf(void *comb_baseptr, void *comb_recvptr, utils::ScalarType dtype,
    const std::vector<int64_t> &sizes)
{
    int64_t numel = 1;
    for (size_t i = 0; i < sizes.size(); i++) {
        numel *= sizes[i];
    }
    int64_t size = numel * utils::elementSize(dtype);
    memset(comb_baseptr, 0, size);
    tmpx_for_sum = (bfloat16_t *)comb_recvptr;
    return utils::Tensor::from_blob(dtype, sizes, comb_baseptr);
}

void moe_comm_create(const utils::Tensor& dispatch_send_buf, const utils::Tensor& combine_send_buf,
    int64_t redundant_n_routed_experts, int64_t n_activated_experts, int64_t num_max_dispatch_tokens_per_rank,
    const utils::Tensor& parallel_policy, MPI_Comm sub_comm, moe_comm_h ds_moe_comm)
{
    kutacc::kurmcl_oob_cb_t oob_cbs;
    kutacc::kurmcl_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;
    oob_cbs_h->oob_alltoall = oob_alltoall_callback;
    int world_size;
    int world_rank;
    static MPI_Comm world_comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    kutacc::kurmcl_comm_create(
        world_size, world_rank, oob_cbs_h, (void *)(&world_comm), &ds_moe_comm->global_ds_conn_info);
    MPI_Comm_size(sub_comm, &num_ranks);
    MPI_Comm_rank(sub_comm, &my_rank);
    kutacc::kurmcl_comm_create(num_ranks, my_rank, oob_cbs_h, (void *)(&sub_comm), &ds_moe_comm->local_ds_conn_info);
    ds_moe_comm->num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;

    int dtp = parallel_policy.data_ptr<int16_t>()[0] / parallel_policy.data_ptr<int16_t>()[1];
    if (dtp == 8) {
        kupl_win = utils::kupl_win_intra_socket;
    } else {
        kupl_win = utils::kupl_win;
    }
    int16_t* parallel_policy_data = parallel_policy.data_ptr<int16_t>();
    int64_t num_tokens = static_cast<int>(dispatch_send_buf.size(0));
    int64_t dispatch_hidden = static_cast<int>(dispatch_send_buf.size(1));
    uint8_t* dispatch_send_buf_data = dispatch_send_buf.data_ptr<uint8_t>();
    int multiple = 2;
    num_local_experts = redundant_n_routed_experts / num_ranks == 0 ? 1 : redundant_n_routed_experts / num_ranks;
    if (params.is_prefill) {
        packed_recv_x_size = num_local_experts * 2 * num_tokens * dispatch_hidden * sizeof(char);
        packed_recv_x = utils::Tensor::from_blob(utils::ScalarType::UInt8,
            {num_local_experts, 2 * num_tokens, dispatch_hidden},
            static_cast<uint8_t*>(disbuf_baseptr));
    } else {
        packed_recv_x_size =
            num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * dispatch_hidden * sizeof(char);
        packed_recv_x = utils::Tensor::from_blob(utils::ScalarType::UInt8,
            {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, dispatch_hidden},
            static_cast<uint8_t*>(disbuf_baseptr));
    }
    recv_src_info_conut = num_local_experts * num_ranks * (num_max_dispatch_tokens_per_rank * 2 + 1);
    recv_src_info_size = recv_src_info_conut * sizeof(int16_t);
    recv_src_info_ep_bias = num_ranks * (num_max_dispatch_tokens_per_rank * 2 + 1);
    disbuf_size = packed_recv_x_size + recv_src_info_size;
    recv_src_info = utils::Tensor::from_blob(utils::ScalarType::Int16,
        {num_local_experts, num_ranks * (num_max_dispatch_tokens_per_rank * 2 + 1)},
        static_cast<uint8_t*>(disbuf_baseptr) + packed_recv_x_size);
    recv_src_info_bak = utils::Tensor::from_blob(utils::ScalarType::Int16,
        {num_local_experts, num_ranks * (num_max_dispatch_tokens_per_rank * 2 + 1)},
        static_cast<uint8_t*>(disbuf_baseptr) + disbuf_size);
    src_info = recv_src_info_bak.data_ptr<int16_t>() + recv_src_info_conut;
    memset(packed_recv_x.ptr, 0, packed_recv_x.numel() * utils::elementSize(packed_recv_x.dtype()));
    uint8_t* packed_recv_x_data = packed_recv_x.data_ptr<uint8_t>();
    int16_t* recv_src_info_data = recv_src_info.data_ptr<int16_t>();
    int16_t* recv_src_info_bak_data = recv_src_info_bak.data_ptr<int16_t>();
    kutacc::moe_dispatch_init(dispatch_send_buf_data, recv_src_info_data, recv_src_info_bak_data,
        redundant_n_routed_experts, multiple, num_max_dispatch_tokens_per_rank, dispatch_hidden, num_tokens, dtp,
        src_info, disbuf_baseptr, ds_moe_comm->local_ds_conn_info);
    int node_nums = world_size / 16;
    int mynode_id = world_rank / 16;
    int color = my_rank / dtp;
    int64_t combine_hidden = static_cast<int>(combine_send_buf.size(2));
    bfloat16_t* base_ptr = (bfloat16_t*)utils::Tensor::alloc_from(utils::ScalarType::BFloat16,
        {num_tokens * params.dim}, utils::shm_available).ptr;
    combined_x =
        utils::Tensor::from_blob(utils::ScalarType::BFloat16, {num_tokens, params.dim}, base_ptr);

    MPI_Comm_split(sub_comm, color, my_rank, &intra_comm);
    int local_rank;
    MPI_Comm_rank(intra_comm, &local_rank);
    std::vector<bfloat16_t*> group_ptr;
    group_ptr.resize(dtp, nullptr);
    for (int i = 0; i < dtp; ++i) {
        if (i != local_rank) {
            utils::get_peer_shm_baseptr(i, base_ptr, (void**)&group_ptr[i]);
        } else {
            group_ptr[i] = base_ptr;
        }
    }
    if (dtp == 8 && world_rank % 16 >= 8) {
        for (int i = 0; i < dtp; ++i) {
            utils::get_peer_shm_baseptr(i + 8, base_ptr, (void**)&group_ptr[i]);
        }
    }
    for (int k = 0; k < dtp; ++k) {
        int group_start = k + my_rank / dtp * dtp + 1;
        for (int i = 0; i < 8; ++i) {
            int row_idx = k * 8 + i;
            for (int j = 0; j < n_activated_experts; ++j) {
                params.topk_idx.data_ptr<int16_t>()[row_idx * 8 + j] =
                    (group_start + j) > 255 ? (group_start + j) - 256 : (group_start + j);
            }
        }
    }
    bfloat16_t *combine_send_buf_data = combine_send_buf.data_ptr<bfloat16_t>();
    std::vector<bfloat16_t*> comb_recv_group_ptr;
    comb_recv_group_ptr.resize(dtp, nullptr);
    if (params.use_static_route) {
        for (int i = 0; i < dtp; ++i) {
            if (i != local_rank) {
                utils::get_peer_shm_baseptr(i, tmpx_for_sum, (void**)&comb_recv_group_ptr[i]);
            } else {
                comb_recv_group_ptr[i] = tmpx_for_sum;
            }
        }
        if (dtp == 8 && world_rank % 16 >= 8) {
            for (int i = 0; i < dtp; ++i) {
                utils::get_peer_shm_baseptr(i + 8, tmpx_for_sum, (void**)&comb_recv_group_ptr[i]);
            }
        }
    } else {
        comb_recv_group_ptr[local_rank] = tmpx_for_sum;
    }
    kutacc::moe_combine_init(combine_send_buf_data, num_tokens, redundant_n_routed_experts,
        num_max_dispatch_tokens_per_rank, n_activated_experts, params.dim, std::move(group_ptr), local_rank,
        std::move(comb_recv_group_ptr), ds_moe_comm->local_ds_conn_info);
}

void dispatch_send(const utils::Tensor& x, const utils::Tensor& topk_idx,  [[maybe_unused]] int64_t num_experts,
    const utils::Tensor& parallel_policy, int64_t batch_id, moe_comm_h ds_moe_comm)
{
    call_times++;
    uint8_t* x_data = x.data_ptr<uint8_t>();
    int16_t* topk_idx_data = topk_idx.data_ptr<int16_t>();
    int16_t* parallel_policy_data = parallel_policy.data_ptr<int16_t>();
    int64_t num_tokens = params.n_tokens;
    int64_t hidden = static_cast<int>(x.size(1));
    int64_t num_topk = static_cast<int>(topk_idx.size(1)) / 2;
    kutacc::moe_dispatch_send(x_data, topk_idx_data, num_tokens, num_topk,
        ds_moe_comm->num_max_dispatch_tokens_per_rank, hidden, parallel_policy_data, batch_id,
        ds_moe_comm->local_ds_conn_info);
}

void dispatch_recv(int64_t batch_id, moe_comm_h ds_moe_comm)
{
    kutacc::moe_dispatch_recv(batch_id, ds_moe_comm->local_ds_conn_info);
}

void combine_send(const utils::Tensor& x, const utils::Tensor& src_info, int64_t num_experts,
    const utils::Tensor& parallel_policy, int64_t batch_id, moe_comm_h ds_moe_comm,
    utils::Tensor& topk_idx, utils::Tensor& topk_weights)
{
    bfloat16_t* x_data = x.data_ptr<bfloat16_t>();
    int16_t* src_info_data = src_info.data_ptr<int16_t>();
    int16_t* parallel_policy_data = parallel_policy.data_ptr<int16_t>();
    int64_t hidden = static_cast<int>(x.size(2));
    bfloat16_t* combined_x_data = combined_x.data_ptr<bfloat16_t>();
    int16_t* topk_idx_data = topk_idx.data_ptr<int16_t>();
    float* topk_weights_data = topk_weights.data_ptr<float>();
    int64_t num_tokens = params.n_tokens;
    int64_t num_topk = static_cast<int>(topk_idx.size(1)) / 2;
    kutacc::moe_combine_send(x_data, src_info_data, ds_moe_comm->num_max_dispatch_tokens_per_rank, num_experts, hidden,
        parallel_policy_data, batch_id, ds_moe_comm->local_ds_conn_info, combined_x_data, topk_idx_data,
        topk_weights_data, num_tokens, num_topk);
}

void combine_recv(utils::Tensor& topk_idx, utils::Tensor& topk_weights, [[maybe_unused]] utils::Tensor& parallel_policy,
    int64_t hidden, int64_t batch_id, moe_comm_h ds_moe_comm)
{
    int16_t* topk_idx_data = topk_idx.data_ptr<int16_t>();
    float* topk_weights_data = topk_weights.data_ptr<float>();
    int64_t num_tokens = params.n_tokens;
    int64_t num_topk = static_cast<int>(topk_idx.size(1)) / 2;
    bfloat16_t* combined_x_data = combined_x.data_ptr<bfloat16_t>();
    kutacc::moe_combine_recv(combined_x_data, topk_idx_data, topk_weights_data, num_tokens,
        ds_moe_comm->num_max_dispatch_tokens_per_rank, num_topk, hidden, batch_id, kupl_win,
        ds_moe_comm->local_ds_conn_info);
}

void moe_comm_destroy()
{
    kutacc::moe_dispatch_finalize();
    kutacc::moe_combine_finalize();
}

void pp_init(utils::Tensor &x, moe_comm_h ds_moe_comm)
{
    uint8_t *x_data = x.data_ptr<uint8_t>();
    int64_t size = x.numel() * x.element_size();
    kutacc::pp_init(x_data, size, ds_moe_comm->global_ds_conn_info);
}

void pp_put(int64_t dest_rank, int64_t send_offset, int64_t size, moe_comm_h ds_moe_comm)
{
    kutacc::pp_put(dest_rank, send_offset, size, ds_moe_comm->global_ds_conn_info);
}

void pp_recv(int64_t src_rank, moe_comm_h ds_moe_comm)
{
    kutacc::pp_recv(src_rank, ds_moe_comm->global_ds_conn_info);
}

void low_latency_barrier(moe_comm_h ds_moe_comm)
{
    kutacc::kurmcl_barrier(ds_moe_comm->local_ds_conn_info);
}

}
