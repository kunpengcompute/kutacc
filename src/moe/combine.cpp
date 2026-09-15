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
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <arm_bf16.h>
#include <arm_neon.h>
#include <arm_sve.h>
#include <omp.h>

#include "internal.h"
#include "../utils/memory.h"
#include "../utils/timer.h"
#include "comm/comm.h"
#include "kutacc.h"

namespace kutacc {

buf_mr_info_t comb_token_buf_send, comb_token_buf_recv, allg_token_buf_send, allg_token_buf_recv;
std::vector<buf_mr_info_t> comb_recv_buf_group;
std::vector<bfloat16_t *> recv_buff_group;
buf_mr_info_t *comb_remote_buf;  // 封装rkey用于交换
buf_mr_info_t *allg_remote_buf;  // 封装rkey用于交换
std::vector<int> iovcont_per_rank_for_comb;
std::vector<std::vector<int>> comb_iov_put_idxs(thread_max_num, std::vector<int>(2, 0));
std::vector<bool> comb_target_rank_indices;
kurmcl_iov_t *iovlist_for_comb = NULL;
kurmcl_iov_t *iovlist_for_allgather = NULL;
bool enable_allgather;
bfloat16_t* sum_temp[MAX_THREAD_NUMS][MAX_NUM_TOPK];

std::atomic<int> counter(0);
std::atomic<int> token_sum_fence(0);
int mynode_id = 0;
int node_nums = 0;
int16_t *true_recv_src_info;

typedef struct kuasync_combine_args {
    int64_t num_tokens;
    int16_t hidden;
    int64_t num_experts;
    int64_t num_topk;
    int16_t num_max_dispatch_tokens_per_rank;
    bfloat16_t *combined_x_data;
    int16_t *topk_idx_data;
    float *topk_weights_data;
    int16_t *true_recv_src_info;
    kurmcl_conn_info_h ds_conn_info;
} kuasync_combine_args_t;
kuasync_combine_args_t comb_args;
std::atomic<int> com_asy_fence1(0);
std::atomic<int> com_asy_fence2(0);
std::atomic<int> com_asy_fence3(0);
std::atomic<int> com_asy_fence4(0);
std::atomic<int> com_asy_fence5(0);

void moe_combine_init(bfloat16_t *new_packed_recv_x_data, int64_t max_num_tokens, int16_t num_experts,
    int16_t num_max_dispatch_tokens_per_rank, int16_t n_activated_experts, int16_t hidden,
    std::vector<bfloat16_t *>&& group_ptr, int local_rank_, std::vector<bfloat16_t *>&& recv_group,
    kurmcl_conn_info_h ds_conn_info)
{
    int world_size = ds_conn_info->world_size;
    int world_rank = ds_conn_info->world_rank;
    int my_rank = ds_conn_info->my_rank;
    int num_ranks = ds_conn_info->comm_size;
    int64_t num_local_experts = num_experts / num_ranks;
    int64_t tmpx_for_sum_size;
    if (cfg.use_static_route) {
        tmpx_for_sum_size = max_num_tokens * n_activated_experts * hidden;
    } else {
        tmpx_for_sum_size = max_num_tokens / dtp * n_activated_experts * hidden;
    }
    recv_buff_group = std::move(recv_group);
    tmpx_for_sum = recv_buff_group[local_rank_];
    mynode_id = world_rank / 16;
    node_nums = world_size / 16;
    comb_target_rank_indices.resize(num_ranks, false);
    int res;
    combinedx_group_ptr = std::move(group_ptr);
    local_rank = local_rank_;
    combinedx_base_ptr = combinedx_group_ptr[local_rank];
    memset(combinedx_base_ptr, 0, max_num_tokens * hidden * sizeof(bfloat16_t));
    iovcont_per_rank_for_comb.resize(num_ranks);
    kurmcl_reg_mr(&comb_token_buf_send, new_packed_recv_x_data,
        num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden * sizeof(int16_t),
        ds_conn_info);  // 注册发方token mr
    kurmcl_reg_mr(&comb_token_buf_recv, tmpx_for_sum, tmpx_for_sum_size * sizeof(int16_t),
        ds_conn_info);  // 注册发方meta mr
    comb_remote_buf = (buf_mr_info_t *)malloc(num_ranks * sizeof(buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &comb_token_buf_recv, comb_remote_buf);  // allgather交换，获取所有其他进程的接收buffer
    iovlist_for_comb =
        (kurmcl_iov_t *)malloc(num_ranks * num_local_experts * max_num_tokens * sizeof(kurmcl_iov_t));  // 申请iovlist
    kurmcl_reg_mr(&allg_token_buf_send, combinedx_base_ptr,
        max_num_tokens * hidden * sizeof(bfloat16_t), ds_conn_info);
    kurmcl_reg_mr(&allg_token_buf_recv, combinedx_base_ptr,
        max_num_tokens * hidden * sizeof(bfloat16_t), ds_conn_info);
    allg_remote_buf = (buf_mr_info_t*)malloc(num_ranks*sizeof (buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &allg_token_buf_recv, allg_remote_buf); // allgather交换，获取所有其他进程的接收buffer
    iovlist_for_allgather = (kurmcl_iov_t *)malloc(16 * sizeof(kurmcl_iov_t));
    for (int i = 0; i < num_ranks; ++i) {
        iovcont_per_rank_for_comb[i] = 0;
    }
}


void combine_async_static(void *args)
{
kuasync_combine_args_t *comb_args = (kuasync_combine_args_t*)args;
    int num_threads = async_tg_get_thread_num();  // 注意num_threads
    int tid = async_tg_get_thread_id();
    int start;
    int end;
    volatile int x;
    int64_t num_tokens = comb_args->num_tokens;
    int64_t hidden = comb_args->hidden;
    int64_t num_topk = comb_args->num_topk;
    int64_t num_experts = comb_args->num_experts;
    kurmcl_conn_info_h ds_conn_info = comb_args->ds_conn_info;
    int my_rank = ds_conn_info->my_rank;
    int num_ranks = ds_conn_info->comm_size;
    int local_rank = my_rank % dtp;
    int16_t num_max_dispatch_tokens_per_rank = comb_args->num_max_dispatch_tokens_per_rank;
    int16_t *true_recv_src_info = comb_args->true_recv_src_info;
    int64_t num_local_experts = num_experts / num_ranks;
    bfloat16_t *combined_x_data = comb_args->combined_x_data;
    int16_t *topk_idx_data = comb_args->topk_idx_data;
    float *topk_weights_data = comb_args->topk_weights_data;
    int16_t send_tokens_per_rank = num_tokens / dtp;
    int16_t start_token_idx = (my_rank % dtp) * send_tokens_per_rank;
    int16_t end_token_idx = (my_rank % dtp + 1) * send_tokens_per_rank;
    int64_t num_threads_per_rank = num_ranks / num_threads;
    int64_t myleader_rank = local_rank < 8 ? 0 : 8;
    int64_t peerleader_rank = local_rank < 8 ? 8 : 0;
    /* comb_send */
    if (tid == 0) {
        int send_bias = 0;
        for (int64_t i = 0; i < num_local_experts; ++i) {
            for (int64_t j = 0; j < num_ranks; ++j) {
                // src_info_data: during dispatch, what i receive
                int recv_nums;
                int count_bias = i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1);
                recv_nums = true_recv_src_info[count_bias];
                if (recv_nums >= 1) {
                    for (int k = 1; k <= recv_nums; ++k) {
                        int remote_node_leader = j / dtp * dtp;
                        int remote_peer = remote_node_leader + (my_rank % dtp);
                        int peer_rank_iov_cnt = iovcont_per_rank_for_comb[remote_node_leader];
                        int64_t iov_bias = remote_node_leader * num_local_experts * num_tokens + peer_rank_iov_cnt;
                        int64_t remote_bias =
                            (j % dtp) * (max_num_tokens / dtp * num_topk * hidden * sizeof(bfloat16_t));
                        iovlist_for_comb[iov_bias].len = hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].local_buffer =
                            comb_token_buf_send.buffer + (send_bias * hidden) * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].lkey = comb_token_buf_send.lkey;
                        iovlist_for_comb[iov_bias].remote_buffer = comb_remote_buf[remote_peer].buffer +
                            remote_bias + true_recv_src_info[count_bias + k * 2] * hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].rkey = comb_remote_buf[remote_peer].rkey;
                        iovcont_per_rank_for_comb[remote_node_leader]++;
                        send_bias++;
                    }
                }
            }
        }
        for (int i = 0; i < num_ranks; i += dtp) {
            if (iovcont_per_rank_for_comb[i] == 0) {
                int64_t iov_bias = i * num_local_experts * num_tokens;
                iovlist_for_comb[iov_bias].len = 0;
                iovlist_for_comb[iov_bias].local_buffer = comb_token_buf_send.buffer;
                iovlist_for_comb[iov_bias].lkey = comb_token_buf_send.lkey;
                iovlist_for_comb[iov_bias].remote_buffer = comb_remote_buf[i].buffer;
                iovlist_for_comb[iov_bias].rkey = comb_remote_buf[i].rkey;
                iovcont_per_rank_for_comb[i] = 1;
            }
        }
    }
    com_asy_fence1++;
    do {
        kutacc_memory_cpu_load_fence();
        x = com_asy_fence1;
    } while (x != num_threads);
    int total_task_nums = num_ranks / dtp;
    int used_thread_nums = std::min(num_threads, total_task_nums);
    int task_nums = total_task_nums / used_thread_nums;
    int start_rank = (my_rank + tid * (num_ranks / used_thread_nums)) % num_ranks;
    for (int i = 0; i < task_nums && tid < used_thread_nums; ++i) {
        int peer_rank = (start_rank + i * dtp) % num_ranks;
        int remote_leader = peer_rank / dtp * dtp;
        kurmcl_put(&(iovlist_for_comb[remote_leader * num_local_experts * num_tokens]),
            iovcont_per_rank_for_comb[remote_leader], peer_rank, 1, ds_conn_info);
    }
    /* comb_recv */
    for (int i = 0; i < task_nums && tid < used_thread_nums; ++i) {
        int peer_rank = (start_rank + i * dtp) % num_ranks;
        int remote_leader = peer_rank / dtp * dtp;
        iovcont_per_rank_for_comb[remote_leader] = 0;
        kurmcl_recv_imm_cnt(1, peer_rank, ds_conn_info);
    }
    com_asy_fence3++;
    do {
        kutacc_memory_cpu_load_fence();
        x = com_asy_fence3;
    } while (x != num_threads);
    if (tid == 0) {
        kurmcl_flush(num_ranks / dtp, ds_conn_info);
        std::fill(comb_target_rank_indices.begin(), comb_target_rank_indices.end(), false);
        dis_peer_nums = 0;
    }
}

void combine_async_dynamic(void *args)
{
    kuasync_combine_args_t *comb_args = (kuasync_combine_args_t*)args;
    int num_threads = async_tg_get_thread_num();  // 注意num_threads
    int tid = async_tg_get_thread_id();
    int start;
    int end;
    volatile int x;
    int64_t num_tokens = comb_args->num_tokens;
    int64_t hidden = comb_args->hidden;
    int64_t num_topk = comb_args->num_topk;
    int64_t num_experts = comb_args->num_experts;
    kurmcl_conn_info_h ds_conn_info = comb_args->ds_conn_info;
    int my_rank = ds_conn_info->my_rank;
    int num_ranks = ds_conn_info->comm_size;
    int local_rank = my_rank % dtp;
    int16_t num_max_dispatch_tokens_per_rank = comb_args->num_max_dispatch_tokens_per_rank;
    int16_t *true_recv_src_info = comb_args->true_recv_src_info;
    int64_t num_local_experts = num_experts / num_ranks;
    bfloat16_t *combined_x_data = comb_args->combined_x_data;
    int16_t *topk_idx_data = comb_args->topk_idx_data;
    float *topk_weights_data = comb_args->topk_weights_data;
    int start_token_idx;
    int end_token_idx;
    cal_task_range(local_rank, dtp, num_tokens, &start_token_idx, &end_token_idx);
    int64_t send_tokens_per_rank = end_token_idx - start_token_idx;
    int64_t num_threads_per_rank = num_ranks / num_threads;
    int64_t myleader_rank = local_rank < 8 ? 0 : 8;
    int64_t peerleader_rank = local_rank < 8 ? 8 : 0;
    /* comb_send */
    if (tid == 0) {
        int send_bias = 0;
        for (int64_t i = 0; i < num_local_experts; ++i) {
            for (int64_t j = 0; j < num_ranks; ++j) {
                // src_info_data: during dispatch, what i receive
                int recv_nums;
                int count_bias = i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1);
                recv_nums = true_recv_src_info[count_bias];
                if (recv_nums != -1) {
                    for (int k = 1; k <= recv_nums; ++k) {
                        int64_t peer_rank_iov_cnt = iovcont_per_rank_for_comb[j];
                        int64_t iov_bias = j * num_local_experts * num_max_dispatch_tokens_per_rank + peer_rank_iov_cnt;
                        iovlist_for_comb[iov_bias].len = hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].local_buffer =
                            comb_token_buf_send.buffer + (send_bias * hidden) * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].lkey = comb_token_buf_send.lkey;
                        iovlist_for_comb[iov_bias].remote_buffer = comb_remote_buf[j].buffer +
                            true_recv_src_info[count_bias + k * 2] * hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].rkey = comb_remote_buf[j].rkey;
                        iovcont_per_rank_for_comb[j]++;
                        comb_target_rank_indices[j] = true;
                        send_bias++;
                    }
                }
            }
        }
        task_partition(comb_target_rank_indices, comb_iov_put_idxs, num_threads);
    }
    com_asy_fence1++;
    do {
        kutacc_memory_cpu_load_fence();
        x = com_asy_fence1;
    } while (x != num_threads);
    for (int i = comb_iov_put_idxs[tid][0]; i < comb_iov_put_idxs[tid][1]; ++i) {
        if (iovcont_per_rank_for_comb[i] > 0) {
            kurmcl_put(&(iovlist_for_comb[i * num_local_experts * num_max_dispatch_tokens_per_rank]),
                iovcont_per_rank_for_comb[i], i, 1, ds_conn_info);
        }
    }
    com_asy_fence2++;
    do {
        kutacc_memory_cpu_load_fence();
        x = com_asy_fence2;
    } while (x != num_threads);
    /* comb_recv */
    cal_task_range(tid, num_threads, dis_peer_nums, &start, &end);
    for (int i = start; i < end; ++i) {
        kurmcl_recv_imm_cnt(1, will_recv_from_rank[i], ds_conn_info);
    }
    cal_task_range(tid, num_threads, num_ranks, &start, &end);
    for (int i = start; i < end; ++i) {
        iovcont_per_rank_for_comb[i] = 0;
    }
    com_asy_fence3++;
    do {
        kutacc_memory_cpu_load_fence();
        x = com_asy_fence3;
    } while (x != num_threads);
    if (tid == 0) {
        std::fill(comb_target_rank_indices.begin(), comb_target_rank_indices.end(), false);
        dis_peer_nums = 0;
    }
    cal_task_range(tid, num_threads, send_tokens_per_rank, &start, &end);
    for (int t = start; t < end; ++t) {
        int token_idx = t + start_token_idx;
        volatile int x;
        int64_t private_token_idx = token_idx >= end_token_idx ? token_idx - 8 : token_idx;
        if (token_idx < end_token_idx) {
            bfloat16_t *token_sum = combined_x_data + token_idx * hidden;
            if (dtp == 8) {
                float *cur_topk_weight_data = topk_weights_data + token_idx * num_topk;
                bfloat16_t *cur_tmpx_for_sum = tmpx_for_sum + t * num_topk * hidden;
                auto ptrue32 = svptrue_b16();
                auto sve_zero = svdup_bf16(0);
                for (int64_t i = 0; i < hidden; i += 32) {
                    auto pred = svwhilelt_b16(i, hidden);
                    auto sve0 = svdup_f32(0);
                    auto sve1 = svdup_f32(0);
                    for (int64_t k = 0; k < num_topk; ++k) {
                        auto sve_ = svldnt1(pred, cur_tmpx_for_sum + k * hidden + i);
                        sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)),
                            cur_topk_weight_data[k]));
                        sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)),
                            cur_topk_weight_data[k]));
                    }
                    auto sve = svuzp1(svcvt_bf16_x(ptrue32, sve0), svcvt_bf16_x(ptrue32, sve1));
                    // Allgather
                    if (enable_allgather) {
                        for (int peer = 0; peer < 8; ++peer) {
                            auto ptr = combinedx_group_ptr[peer] + private_token_idx * hidden;
                            svstnt1(pred, ptr + i, sve);
                        }
                    } else {
                        auto ptr = combinedx_group_ptr[local_rank] + private_token_idx * hidden;
                        svstnt1(pred, ptr + i, sve);
                    }
                }
            } else if (dtp == 16) {
                std::vector<std::pair<float, bfloat16_t *>> g;
                for (int i = 0; i < num_topk; ++i) {
                    g.push_back(
                        {topk_weights_data[token_idx * num_topk + i], tmpx_for_sum + (t * num_topk + i) * hidden});
                }
                int i = 0;
                svbool_t pred = svptrue_b16();
                svbfloat16_t sve_zero = svdup_bf16(0);
                for (; i + 32 <= hidden; i += 32) {
                    svbfloat16_t sve_ = sve_zero;  // svld1(pred, token_sum + i);
                    svfloat32_t sve0 = svreinterpret_f32(svzip1(sve_zero, sve_));
                    svfloat32_t sve1 = svreinterpret_f32(svzip2(sve_zero, sve_));
                    for (const auto &[v, p] : g) {
                        sve_ = svld1(pred, p + i);
                        sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)), v));
                        sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)), v));
                    }
                    svfloat32_t sve2 = svuzp1_f32(sve0, sve1);
                    svfloat32_t sve3 = svuzp2_f32(sve0, sve1);
                    svbfloat16_t sve = svcvt_bf16_f32_x(pred, sve2);
                    sve = svcvtnt_bf16_f32_x(sve, pred, sve3);
                    svst1(pred, token_sum + i, sve);
                }
                if (__builtin_expect((i < hidden), 0)) {
                    for (int j = i; j < hidden; ++j) {
                        token_sum[j] = 0;
                        for (const auto &[v, p] : g) {
                            token_sum[j] += p[j] * v;
                        }
                    }
                }
            }
        }
    }
    if (send_tokens_per_rank % num_threads) {
        ++token_sum_fence;
        do {
            kutacc_memory_cpu_load_fence();
            x = token_sum_fence;
        } while (x != num_threads);
    }
    if (dtp == 16 && enable_allgather) {
        cal_task_range(tid, num_threads, 8, &start, &end);  // 8个对端进程
        for (int t = start; t < end; ++t) {
            int i = 0;
            int peer = mynode_id * 16 + t + peerleader_rank;
            int64_t part_length = send_tokens_per_rank * hidden / 8;
            bfloat16_t *token_sum = combined_x_data + (local_rank * send_tokens_per_rank * hidden + t * part_length);

            iovlist_for_allgather[t].len = part_length * sizeof(bfloat16_t);
            iovlist_for_allgather[t].local_buffer = allg_token_buf_send.buffer +
                ((local_rank * send_tokens_per_rank * hidden) + t * part_length) * sizeof(bfloat16_t);
            iovlist_for_allgather[t].lkey = allg_token_buf_send.lkey;
            iovlist_for_allgather[t].remote_buffer = allg_remote_buf[peer].buffer +
                ((local_rank * send_tokens_per_rank * hidden) + t * part_length) * sizeof(bfloat16_t);
            iovlist_for_allgather[t].rkey = allg_remote_buf[peer].rkey;
            kurmcl_put(&(iovlist_for_allgather[t]), 1, peer, 1, ds_conn_info);

            for (i = 0; i < part_length; i += 32) {
                svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)part_length);
                auto sve = svld1(pred, token_sum + i);
                for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                    if (peer != local_rank) {
                        svst1(pred,
                            combinedx_group_ptr[peer] +
                                ((local_rank * send_tokens_per_rank * hidden) + t * part_length) + i,
                            sve);
                    }
                }
            }
        }
        for (int t = start; t < end; ++t) {
            int peer = mynode_id * 16 + t + peerleader_rank;
            int64_t part_length = send_tokens_per_rank * hidden / 8;
            kurmcl_recv_imm_cnt(1, mynode_id * 16 + peerleader_rank + t, ds_conn_info);
            int64_t bias = ((peerleader_rank + t) * 8 + local_rank - myleader_rank);
            bias = bias * part_length;

            for (int i = 0; i < part_length; i += 32) {
                svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)part_length);
                auto sve = svld1(pred, &(combined_x_data[bias]) + i);
                for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                    if (peer != local_rank) {
                        svst1(pred, combinedx_group_ptr[peer] + bias + i, sve);
                    }
                }
            }
        }
    }
    ++com_asy_fence5; //
    do {
        kutacc_memory_cpu_load_fence();
        x = com_asy_fence5;
    } while (x != num_threads);
    cal_task_range(tid, num_threads, num_ranks, &start, &end);
    for (int j = start; j < end; ++j) {
        for (int i = 0; i < num_local_experts; ++i) {
            true_recv_src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank  * 2 + 1)] = -1;
            src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = 0;
        }
    }
}

void combine_async(void *args)
{
    if (cfg.use_static_route) {
        combine_async_static(args);
    } else {
        combine_async_dynamic(args);
    }
}

void moe_combine_send_static(bfloat16_t *x_data, int16_t *src_info_data,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden,
    int16_t *parallel_policy_data, int64_t batch_id, kurmcl_conn_info_h ds_conn_info,
    bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data, int64_t num_tokens, int64_t num_topk)
{
    comb_times++;
    if (comb_times % 2 == 1) {
        true_recv_src_info = catch_recv_src_info;
    } else {
        true_recv_src_info = catch_recv_src_info_bak;
    }
    if (cfg.use_async_comb) {
        comb_args.num_tokens = num_tokens;
        comb_args.num_topk = num_topk;
        comb_args.hidden = hidden;
        comb_args.num_experts = num_experts;
        comb_args.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;
        comb_args.combined_x_data = combined_x_data;
        comb_args.topk_idx_data = topk_idx_data;
        comb_args.topk_weights_data = topk_weights_data;
        comb_args.true_recv_src_info = true_recv_src_info;
        comb_args.ds_conn_info = ds_conn_info;
        async_tg_task_submit(comm_thread_exec, combine_async, &comb_args);
    } else {
        int my_rank = ds_conn_info->my_rank;
        int num_ranks = ds_conn_info->comm_size;
        int64_t num_local_experts = num_experts / num_ranks;
        int tid;
        int local_rank = my_rank % dtp;
        // 接收连续输入
        int send_bias = 0;
        for (int64_t i = 0; i < num_local_experts; ++i) {
            for (int64_t j = 0; j < num_ranks; ++j) {
                // src_info_data: during dispatch, what i receive
                int recv_nums;
                int count_bias = i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1);
                recv_nums = true_recv_src_info[count_bias];
                if (recv_nums >= 1) {
                    for (int k = 1; k <= recv_nums; ++k) {
                        int remote_node_leader = j / dtp * dtp;
                        int remote_peer = remote_node_leader + (my_rank % dtp);
                        int peer_rank_iov_cnt = iovcont_per_rank_for_comb[remote_node_leader];
                        int64_t iov_bias = remote_node_leader * num_local_experts * num_tokens + peer_rank_iov_cnt;
                        int64_t remote_bias =
                            (j % dtp) * (max_num_tokens / dtp * num_topk * hidden * sizeof(bfloat16_t));
                        iovlist_for_comb[iov_bias].len = hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].local_buffer =
                            comb_token_buf_send.buffer + (send_bias * hidden) * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].lkey = comb_token_buf_send.lkey;
                        iovlist_for_comb[iov_bias].remote_buffer = comb_remote_buf[remote_peer].buffer +
                            remote_bias + true_recv_src_info[count_bias + k * 2] * hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].rkey = comb_remote_buf[remote_peer].rkey;
                        iovcont_per_rank_for_comb[remote_node_leader]++;
                        send_bias++;
                    }
                }
            }
        }
        for (int i = 0; i < num_ranks; i += dtp) {
            if (iovcont_per_rank_for_comb[i] == 0) {
                int64_t iov_bias = i * num_local_experts * num_tokens;
                iovlist_for_comb[iov_bias].len = 0;
                iovlist_for_comb[iov_bias].local_buffer = comb_token_buf_send.buffer;
                iovlist_for_comb[iov_bias].lkey = comb_token_buf_send.lkey;
                iovlist_for_comb[iov_bias].remote_buffer = comb_remote_buf[i].buffer;
                iovlist_for_comb[iov_bias].rkey = comb_remote_buf[i].rkey;
                iovcont_per_rank_for_comb[i] = 1;
            }
        }
        kutacc::parallel_for(0, num_ranks / dtp, 1, [&](int64_t start, int64_t end) {
            int64_t local_rank = my_rank % dtp;
            for (int i = (start * dtp) + local_rank; i < (end * dtp) + local_rank; i += dtp) {
                int remote_leader = i / dtp * dtp;
                kurmcl_put(&(iovlist_for_comb[remote_leader * num_local_experts * num_tokens]),
                    iovcont_per_rank_for_comb[remote_leader], i, 1, ds_conn_info);
            }
        });
    }
}

void moe_combine_send_dynamic(bfloat16_t *x_data, int16_t *src_info_data,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden,
    int16_t *parallel_policy_data, int64_t batch_id, kurmcl_conn_info_h ds_conn_info,
    bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data, int64_t num_tokens, int64_t num_topk)
{
    comb_times++;
    if (comb_times % 2 == 1) {
        true_recv_src_info = catch_recv_src_info;
    } else {
        true_recv_src_info = catch_recv_src_info_bak;
    }
    if (cfg.use_async_comb) {
        comb_args.num_tokens = num_tokens;
        comb_args.num_topk = num_topk;
        comb_args.hidden = hidden;
        comb_args.num_experts = num_experts;
        comb_args.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;
        comb_args.combined_x_data = combined_x_data;
        comb_args.topk_idx_data = topk_idx_data;
        comb_args.topk_weights_data = topk_weights_data;
        comb_args.true_recv_src_info = true_recv_src_info;
        comb_args.ds_conn_info = ds_conn_info;
        async_tg_task_submit(comm_thread_exec, combine_async, &comb_args);
    } else {
        int my_rank = ds_conn_info->my_rank;
        int num_ranks = ds_conn_info->comm_size;
        int64_t num_local_experts = num_experts / num_ranks;
        int tid;
        int local_rank = my_rank % dtp;
        // 接收连续输入
        int send_bias = 0;
        for (int64_t i = 0; i < num_local_experts; ++i) {
            for (int64_t j = 0; j < num_ranks; ++j) {
                // src_info_data: during dispatch, what i receive
                int recv_nums;
                int count_bias = i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1);
                recv_nums = true_recv_src_info[count_bias];
                if (recv_nums >= 1) {
                    for (int k = 1; k <= recv_nums; ++k) {
                        int64_t peer_rank_iov_cnt = iovcont_per_rank_for_comb[j];
                        int64_t iov_bias = j * num_local_experts * num_max_dispatch_tokens_per_rank + peer_rank_iov_cnt;
                        iovlist_for_comb[iov_bias].len = hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].local_buffer =
                            comb_token_buf_send.buffer + (send_bias * hidden) * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].lkey = comb_token_buf_send.lkey;
                        iovlist_for_comb[iov_bias].remote_buffer = comb_remote_buf[j].buffer +
                            true_recv_src_info[count_bias + k * 2] * hidden * sizeof(bfloat16_t);
                        iovlist_for_comb[iov_bias].rkey = comb_remote_buf[j].rkey;
                        iovcont_per_rank_for_comb[j]++;
                        comb_target_rank_indices[j] = true;
                        send_bias++;
                    }
                }
            }
        }
        int thread_num = 16;
        task_partition(comb_target_rank_indices, comb_iov_put_idxs, thread_num);
        kutacc::parallel_for(0, thread_num, 1, [&](int64_t start, int64_t end) {
            int tid = start;
            for (int i = comb_iov_put_idxs[tid][0]; i < comb_iov_put_idxs[tid][1]; ++i) {
                if (iovcont_per_rank_for_comb[i] > 0) {
                    kurmcl_put(&(iovlist_for_comb[i * num_local_experts * num_max_dispatch_tokens_per_rank]),
                        iovcont_per_rank_for_comb[i], i, 1, ds_conn_info);
                }
            }
        });
    }
}

void moe_combine_send(bfloat16_t *x_data, int16_t *src_info_data, int64_t num_max_dispatch_tokens_per_rank,
    int64_t num_experts, int64_t hidden, int16_t *parallel_policy_data, int64_t batch_id,
    kurmcl_conn_info_h ds_conn_info, bfloat16_t *combined_x_data, int16_t *topk_idx_data,
    float *topk_weights_data, int64_t num_tokens, int64_t num_topk, bool enable_allgather_)
{
    enable_allgather = enable_allgather_;
    if (cfg.use_static_route) {
        moe_combine_send_static(x_data, src_info, num_max_dispatch_tokens_per_rank, num_experts, hidden,
            parallel_policy_data, batch_id, ds_conn_info, combined_x_data, topk_idx_data, topk_weights_data,
            num_tokens, num_topk);
    } else {
        moe_combine_send_dynamic(x_data, src_info, num_max_dispatch_tokens_per_rank, num_experts, hidden,
            parallel_policy_data, batch_id, ds_conn_info, combined_x_data, topk_idx_data, topk_weights_data,
            num_tokens, num_topk);
    }
}

void moe_combine_recv_static(bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data,
    int64_t num_tokens, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden, int64_t batch_id,
    kupl_shm_win_h win, kurmcl_conn_info_h ds_conn_info)
{
    int my_rank = ds_conn_info->my_rank;
    int num_ranks = ds_conn_info->comm_size;
    int send_tokens_per_rank = num_tokens / dtp;
    int start_token_idx = (my_rank % dtp) * send_tokens_per_rank;
    int end_token_idx = (my_rank % dtp + 1) * send_tokens_per_rank;
    int64_t tid;
    int64_t num_threads = 8;
    int64_t num_threads_per_rank = num_ranks / num_threads;
    int64_t myleader_rank = local_rank < 8 ? 0 : 8;
    int64_t peerleader_rank = local_rank < 8 ? 8 : 0;
    int64_t local_recv_buff_bias = (my_rank % dtp) * (max_num_tokens / dtp * num_topk * hidden);

    if (cfg.use_async_comb) {
        async_tg_wait(comm_thread_exec);
        com_asy_fence1 = 0;
        com_asy_fence2 = 0;
        com_asy_fence3 = 0;
        com_asy_fence4 = 0;
        com_asy_fence5 = 0;
    } else {
        kutacc::parallel_for(0, num_ranks / dtp, 1, [&](int64_t start, int64_t end) {
            for (int i = start; i < end; ++i) {
                kurmcl_recv_imm_cnt(1, my_rank % dtp + i * dtp, ds_conn_info);
            }
        });
        kurmcl_flush(num_ranks / dtp, ds_conn_info);
        dis_peer_nums = 0;
        for (int i = 0; i < num_ranks; ++i) {
            iovcont_per_rank_for_comb[i] = 0;
        }
    }
    kupl_shm_fence(win);
    const int used_thread_nums = 8;  // 暂不支持非8
    int send_tokens_per_thread;
    kutacc::parallel_for(0, used_thread_nums, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int thread_start_token_idx;
        int thread_end_token_idx;
        cal_task_range(tid, used_thread_nums, send_tokens_per_rank, &thread_start_token_idx, &thread_end_token_idx);
        send_tokens_per_thread = thread_end_token_idx - thread_start_token_idx;
        for (int token_idx = thread_start_token_idx + start_token_idx;
            token_idx < thread_end_token_idx + start_token_idx; ++token_idx) {
            int local_idx = token_idx - start_token_idx;
            bfloat16_t *token_sum = combined_x_data + token_idx * hidden;
            int64_t reduce_bias = local_recv_buff_bias + local_idx * num_topk * hidden;
            for (int i = 0; i < num_topk; ++i) {
                // sum_temp[i] = recv_buff_group[topk_idx_data[token_idx * num_topk + i] / num_local_experts % dtp] + reduce_bias + i * hidden;
                sum_temp[tid][i] =
                    recv_buff_group[topk_idx_data[token_idx * num_topk * 2 + i * 2] % dtp] +reduce_bias + i * hidden;
            }
            if (dtp == 8) {
                float *cur_topk_weight_data = topk_weights_data + token_idx * num_topk;
                auto ptrue32 = svptrue_b16();
                auto sve_zero = svdup_bf16(0);
                for (int64_t i = 0; i < hidden; i += 32) {
                    auto pred = svwhilelt_b16(i, hidden);
                    auto sve0 = svdup_f32(0);
                    auto sve1 = svdup_f32(0);
                    for (int64_t k = 0; k < num_topk; ++k) {
#ifdef ENABLE_EXTRA_PREFETCH
                        constexpr int prf_stride = 3 * 1024 / 8 & ~63;
                        svprfh(svwhilelt_b16(i + prf_stride / 2, hidden),
                            sum_temp[tid][k] + i + prf_stride / 2, SV_PLDL2STRM);
#endif
                        auto sve_ = svldnt1(pred, sum_temp[tid][k] + i);
                        sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)),
                            cur_topk_weight_data[k]));
                        sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)),
                            cur_topk_weight_data[k]));
                    }
                    auto sve = svuzp1(svcvt_bf16_x(ptrue32, sve0), svcvt_bf16_x(ptrue32, sve1));
                    if (enable_allgather) {
                        for (int peer = 0; peer < 8; ++peer) {
                            auto ptr = combinedx_group_ptr[peer] + token_idx * hidden;
#ifdef ENABLE_EXTRA_PREFETCH
                            constexpr int prf_stride = 4 * 1024 / 8 & ~63;
                            svprfh(svwhilelt_b16(i + prf_stride / 2, hidden), ptr + i + prf_stride / 2, SV_PSTL2STRM);
#endif
                            svstnt1(pred, ptr + i, sve);
                        }
                    } else {
                        auto ptr = combinedx_group_ptr[local_rank] + token_idx * hidden;
                        svstnt1(pred, ptr + i, sve);
                    }
                }
            } else if (dtp == 16) {
                std::vector<std::pair<float, bfloat16_t *>> g;
                for (int i = 0; i < num_topk; ++i) {
                    int src_rank = topk_idx_data[token_idx * num_topk * 2 + i * 2];
                    int src_rank_local = src_rank % dtp;
                    g.push_back({topk_weights_data[token_idx * num_topk + i],
                        recv_buff_group[src_rank_local] + local_recv_buff_bias + (local_idx * num_topk + i) * hidden});
                }
                int i = 0;
                svbool_t pred = svptrue_b16();
                svbfloat16_t sve_zero = svdup_bf16(0);
                for (; i + 32 <= hidden; i += 32) {
                    svbfloat16_t sve_ = sve_zero;  // svld1(pred, token_sum + i);
                    svfloat32_t sve0 = svreinterpret_f32(svzip1(sve_zero, sve_));
                    svfloat32_t sve1 = svreinterpret_f32(svzip2(sve_zero, sve_));
                    for (const auto &[v, p] : g) {
                        sve_ = svld1(pred, p + i);
                        sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)), v));
                        sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)), v));
                    }
                    svfloat32_t sve2 = svuzp1_f32(sve0, sve1);
                    svfloat32_t sve3 = svuzp2_f32(sve0, sve1);
                    svbfloat16_t sve = svcvt_bf16_f32_x(pred, sve2);
                    sve = svcvtnt_bf16_f32_x(sve, pred, sve3);
                    svst1(pred, token_sum + i, sve);
                }
                if (__builtin_expect((i < hidden), 0)) {
                    for (int j = i; j < hidden; ++j) {
                        token_sum[j] = 0;
                        for (const auto &[v, p] : g) {
                            token_sum[j] += p[j] * v;
                        }
                    }
                }
            }
        }
        // dtp16 Allgather
        if (dtp == 16 && enable_allgather) {
            if (send_tokens_per_rank % used_thread_nums) {
                volatile int x;
                ++token_sum_fence;
                do {
                    kutacc_memory_cpu_load_fence();
                    x = token_sum_fence;
                } while (x != used_thread_nums);
            }
            int peer = mynode_id * 16 + tid + peerleader_rank;
            int64_t part_length = send_tokens_per_rank * hidden / 8;
            int data_bias = local_rank * send_tokens_per_rank * hidden + tid * part_length;
            bfloat16_t *token_sum = combined_x_data + data_bias;

            iovlist_for_allgather[tid].len = part_length * sizeof(bfloat16_t);
            iovlist_for_allgather[tid].local_buffer = allg_token_buf_send.buffer
                + data_bias * sizeof(bfloat16_t);
            iovlist_for_allgather[tid].lkey = allg_token_buf_send.lkey;
            iovlist_for_allgather[tid].remote_buffer = allg_remote_buf[peer].buffer
                + data_bias * sizeof(bfloat16_t);
            iovlist_for_allgather[tid].rkey = allg_remote_buf[peer].rkey;
            kurmcl_put(&(iovlist_for_allgather[tid]), 1, peer, 1, ds_conn_info);
            for (int i = 0; i < part_length; i += 32) {
                svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)part_length);
                auto sve = svld1(pred, token_sum + i);
                for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                    if (peer != local_rank) {
                        svst1(pred, combinedx_group_ptr[peer] + data_bias + i, sve);
                    }
                }
            }
            kurmcl_recv_imm_cnt(1, mynode_id * 16 + peerleader_rank + tid, ds_conn_info);
            int64_t bias = ((peerleader_rank + tid) * 8 + local_rank - myleader_rank);
            bias = bias * part_length;
            for (int i = 0; i < part_length; i += 32) {
                svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)part_length);
                auto sve = svld1(pred, &(combined_x_data[bias]) + i);
                for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                    if (peer != local_rank) {
                        svst1(pred, combinedx_group_ptr[peer] + bias + i, sve);
                    }
                }
            }
        }
        for (int j = tid * num_threads_per_rank; j < (tid + 1) * num_threads_per_rank; ++j) {
            for (int i = 0; i < num_local_experts; ++i) {
                true_recv_src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = -1;
                src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = 0;
            }
        }
    });
    token_sum_fence = 0;
    kurmcl_flush(8, ds_conn_info);
    counter = 0;
    kupl_shm_fence(win);
}

void moe_combine_recv_dynamic(bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data,
    int64_t num_tokens, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden, int64_t batch_id,
    kupl_shm_win_h win, kurmcl_conn_info_h ds_conn_info)
{
    int my_rank = ds_conn_info->my_rank;
    int num_ranks = ds_conn_info->comm_size;
    int local_rank = my_rank % dtp;
    int start_token_idx;
    int end_token_idx;
    cal_task_range(local_rank, dtp, num_tokens, &start_token_idx, &end_token_idx);
    int64_t send_tokens_per_rank = end_token_idx - start_token_idx;
    int64_t tid;
    int64_t num_threads = 8;
    int64_t num_threads_per_rank = num_ranks / num_threads;
    int64_t myleader_rank = local_rank < 8 ? 0 : 8;
    int64_t peerleader_rank = local_rank < 8 ? 8 : 0;
    int64_t local_recv_buff_bias = (my_rank % dtp) * (num_tokens / dtp * num_topk * hidden);

    if (cfg.use_async_comb) {
        async_tg_wait(comm_thread_exec);
        com_asy_fence1 = 0;
        com_asy_fence2 = 0;
        com_asy_fence3 = 0;
        com_asy_fence4 = 0;
        com_asy_fence5 = 0;
        token_sum_fence = 0;
    } else {
        kutacc::parallel_for(0, dis_peer_nums, 1, [&](int64_t start, int64_t end) {
            for (int i = start; i < end; ++i) {
                kurmcl_recv_imm_cnt(1, will_recv_from_rank[i], ds_conn_info);
            }
        });
        kurmcl_flush(1, ds_conn_info);
        std::fill(comb_target_rank_indices.begin(), comb_target_rank_indices.end(), false);
        dis_peer_nums = 0;
        for (int i = 0; i < num_ranks; ++i) {
            iovcont_per_rank_for_comb[i] = 0;
        }
        const int used_thread_nums = 8;  // 暂不支持非8
        kutacc::parallel_for(0, used_thread_nums, 1, [&](int64_t start, int64_t end) {
            int tid = start;
            int thread_start;
            int thread_end;
            cal_task_range(tid, used_thread_nums, send_tokens_per_rank, &thread_start, &thread_end);
            for (int token_idx = start_token_idx + thread_start; token_idx < start_token_idx + thread_end;
                token_idx++) {
                int local_idx = token_idx - start_token_idx;
                bfloat16_t *token_sum = combined_x_data + token_idx * hidden;
                std::vector<std::pair<float, bfloat16_t *>> g;
                for (int i = 0; i < num_topk; ++i) {
                    g.push_back({topk_weights_data[token_idx * num_topk + i],
                        tmpx_for_sum + (local_idx * num_topk + i) * hidden});
                }
                if (dtp == 8) {
                    auto ptrue32 = svptrue_b16();
                    auto sve_zero = svdup_bf16(0);
                    for (int64_t i = 0; i < hidden; i += 32) {
                        auto pred = svwhilelt_b16(i, hidden);
                        auto sve0 = svdup_f32(0);
                        auto sve1 = svdup_f32(0);
                        for (const auto [v, p] : g) {
#ifdef ENABLE_EXTRA_PREFETCH
                            constexpr int prf_stride = 3 * 1024 / 8 & ~63;
                            svprfh(svwhilelt_b16(i + prf_stride / 2, hidden), p + i + prf_stride / 2, SV_PLDL2STRM);
#endif
                            auto sve_ = svldnt1(pred, p + i);
                            sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)), v));
                            sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)), v));
                        }
                        auto sve = svuzp1(svcvt_bf16_x(ptrue32, sve0), svcvt_bf16_x(ptrue32, sve1));
                        int allg_range = enable_allgather ? 8 : 1;
                        for (int offset = 0; offset < allg_range; ++offset) {
                            auto ptr = combinedx_group_ptr[(local_rank+offset)%8] + token_idx * hidden;
#ifdef ENABLE_EXTRA_PREFETCH
                            constexpr int prf_stride = 4 * 1024 / 8 & ~63;
                            svprfh(svwhilelt_b16(i + prf_stride / 2, hidden), ptr + i + prf_stride / 2, SV_PSTL2STRM);
#endif
                            svstnt1(pred, ptr + i, sve);
                        }
                    }
                } else if (dtp == 16) {
                    int i = 0;
                    svbool_t pred = svptrue_b16();
                    svbfloat16_t sve_zero = svdup_bf16(0);
                    for (; i + 32 <= hidden; i += 32) {
                        svbfloat16_t sve_ = sve_zero;  // svld1(pred, token_sum + i);
                        svfloat32_t sve0 = svreinterpret_f32(svzip1(sve_zero, sve_));
                        svfloat32_t sve1 = svreinterpret_f32(svzip2(sve_zero, sve_));
                        for (const auto &[v, p] : g) {
                            sve_ = svld1(pred, p + i);
                            sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)), v));
                            sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)), v));
                        }
                        svfloat32_t sve2 = svuzp1_f32(sve0, sve1);
                        svfloat32_t sve3 = svuzp2_f32(sve0, sve1);
                        svbfloat16_t sve = svcvt_bf16_f32_x(pred, sve2);
                        sve = svcvtnt_bf16_f32_x(sve, pred, sve3);
                        svst1(pred, token_sum + i, sve);
                    }
                    if (__builtin_expect((i < hidden), 0)) {
                        for (int j = i; j < hidden; ++j) {
                            token_sum[j] = 0;
                            for (const auto &[v, p] : g) {
                                token_sum[j] += p[j] * v;
                            }
                        }
                    }
                }
            }
            volatile int x;
            // dtp16 Allgather
            if (dtp == 16 && enable_allgather) {
                if (send_tokens_per_rank % used_thread_nums) {
                    ++token_sum_fence;
                    do {
                        kutacc_memory_cpu_load_fence();
                        x = token_sum_fence;
                    } while (x != used_thread_nums);
                }
                int peer = mynode_id * 16 + tid + peerleader_rank;
                int64_t part_length = send_tokens_per_rank * hidden / used_thread_nums;
                int data_bias = start_token_idx * hidden + tid * part_length;
                bfloat16_t *token_sum = combined_x_data + data_bias;
                if (num_tokens % dtp) {
                    for (int peer = 0; peer < dtp; ++peer) {
                        if (peer != local_rank) {
                            memcpy(combinedx_group_ptr[peer] + data_bias,
                                combinedx_base_ptr + data_bias, part_length * sizeof(bfloat16_t));
                        }
                    }
                } else {
                    iovlist_for_allgather[tid].len = part_length * sizeof(bfloat16_t);
                    iovlist_for_allgather[tid].local_buffer = allg_token_buf_send.buffer
                        + data_bias * sizeof(bfloat16_t);
                    iovlist_for_allgather[tid].lkey = allg_token_buf_send.lkey;
                    iovlist_for_allgather[tid].remote_buffer = allg_remote_buf[peer].buffer
                        + data_bias * sizeof(bfloat16_t);
                    iovlist_for_allgather[tid].rkey = allg_remote_buf[peer].rkey;
                    kurmcl_put(&(iovlist_for_allgather[tid]), 1, peer, 1, ds_conn_info);
                    for (int i = 0; i < part_length; i += 32) {
                        svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)part_length);
                        auto sve = svld1(pred, token_sum + i);
                        for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                            if (peer != local_rank) {
                                svst1(pred, combinedx_group_ptr[peer] + data_bias + i, sve);
                            }
                        }
                    }
                    kurmcl_recv_imm_cnt(1, mynode_id * 16 + peerleader_rank + tid, ds_conn_info);
                    int64_t bias = ((peerleader_rank + tid) * 8 + local_rank - myleader_rank);
                    bias = bias * part_length;
                    for (int i = 0; i < part_length; i += 32) {
                        svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)part_length);
                        auto sve = svld1(pred, &(combined_x_data[bias]) + i);
                        for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                            if (peer != local_rank) {
                                svst1(pred, combinedx_group_ptr[peer] + bias + i, sve);
                            }
                        }
                    }
                }
            }
            ++counter;
            do {
                kutacc_memory_cpu_load_fence();
                x = counter;
            } while (x != used_thread_nums);
            for (int j = tid * num_threads_per_rank; j < (tid + 1) * num_threads_per_rank; ++j) {
                for (int i = 0; i < num_local_experts; ++i) {
                    true_recv_src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = -1;
                    src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = 0;
                }
            }
        });
        kurmcl_flush(8, ds_conn_info);
        counter = 0;
        token_sum_fence = 0;
    }
    kupl_shm_fence(win);
}

void moe_combine_recv(bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data, int64_t num_tokens,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden, int64_t batch_id,
    kupl_shm_win_h win, kurmcl_conn_info_h ds_conn_info)
{
    if (cfg.use_static_route) {
        moe_combine_recv_static(combined_x_data, topk_idx_data, topk_weights_data, num_tokens,
            num_max_dispatch_tokens_per_rank, num_topk, hidden, batch_id, win, ds_conn_info);
    } else {
        moe_combine_recv_dynamic(combined_x_data, topk_idx_data, topk_weights_data, num_tokens,
            num_max_dispatch_tokens_per_rank, num_topk, hidden, batch_id, win, ds_conn_info);
    }
}

void moe_combine_finalize()
{
    kurmcl_dreg_mr(&comb_token_buf_send);
    kurmcl_dreg_mr(&comb_token_buf_recv);
    kurmcl_finalize(ds_conn_info);
    free(iovlist_for_comb);
    if (cfg.use_async_disp || cfg.use_async_comb) {
        async_tg_destroy(comm_thread_exec);
    }
}

}  // namespace kutacc
