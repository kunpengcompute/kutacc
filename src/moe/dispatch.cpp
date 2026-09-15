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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include "../utils/timer.h"
#include "../utils/memory.h"
#include "internal.h"
#include "kutacc.h"

namespace kutacc {

buf_mr_info_t disp_token_buf_send, disp_meta_buf_send, disp_buf_recv;
buf_mr_info_t *disp_remote_buf;  // 封装rkey用于交换
std::vector<int> iovcont_per_rank_for_disp;
kurmcl_iov_t *iovlist_for_disp = NULL;
std::vector<std::vector<int>>disp_iov_put_idxs(thread_max_num, std::vector<int>(2, 0)); // {start, end}
std::vector<bool> disp_target_rank_indices;
std::atomic<int> put_fence(0);
// node based barrier abbreviated as nbb
buf_mr_info_t disp_nbb_send, disp_nbb_recv, *disp_nbb_remote;
int nbb_ppn = 16;
int nbb_num_node;
kurmcl_iov_t nbb_iov;
// end nnb
// for prefill metedata exchange
kurmcl_iov_t *iovlist_for_a2a = NULL;
std::vector<int> token_offset, token_offset_static;
buf_mr_info_t disp_send_count, disp_recv_count, *disp_recv_count_remote;
int send_count_ary_size;
// end prefill metedata exchange

typedef struct kuasync_dispatch_args {
    int64_t num_max_dispatch_tokens_per_rank;
    int64_t num_topk;
    int16_t *topk_idx_data;
    int16_t hidden;
    int64_t num_tokens;
    int64_t target_recv_src_info_bias;
    kurmcl_conn_info_h ds_conn_info;
} kuasync_dispatch_args_t;
kuasync_dispatch_args_t dispatch_args;
kurmcl_iov_t local_nbb_iov[ASY_CORE_USED_CNT_DEFAULT];
std::atomic<int> dis_asy_fence1(0);
std::atomic<int> dis_asy_fence2(0);

void moe_dispatch_init(uint8_t *x_data, int16_t *recv_src_info_data,
    int16_t *recv_src_info_bak_data, int64_t num_experts, int64_t multiple, int64_t num_max_dispatch_tokens_per_rank,
    int64_t hidden, int64_t max_num_tokens_, int64_t dtp_, int16_t *src_info_, void* disbuf_baseptr,
    kurmcl_conn_info_h ds_conn_info)
{
    moe_config_init(is_prefill_default, use_async_disp_default, use_async_comb_default, use_static_route_default);
    int num_ranks = ds_conn_info->comm_size;
    max_num_tokens = max_num_tokens_;
    num_local_experts = num_experts / num_ranks == 0 ? 1 : num_experts / num_ranks;
    packed_recv_x_size = cfg.is_prefill ? num_local_experts * multiple * max_num_tokens * hidden * sizeof(char) :
        num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden * sizeof(char);
    recv_src_info_conut =
        num_local_experts * num_ranks * (num_max_dispatch_tokens_per_rank * 2 + 1);
    recv_src_info_size = recv_src_info_conut * sizeof(int16_t);
    recv_src_info_ep_bias = num_ranks * (num_max_dispatch_tokens_per_rank * 2 + 1);
    disbuf_size = packed_recv_x_size + recv_src_info_size * 2;
    disp_target_rank_indices.resize(num_ranks, false);
    int res;
    catch_recv_src_info = recv_src_info_data;
    catch_recv_src_info_bak = recv_src_info_bak_data;
    src_info = src_info_;
    if (src_info == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    dtp = dtp_;
    kurmcl_reg_mr(&disp_token_buf_send, x_data, max_num_tokens * hidden, ds_conn_info);  // 注册发方token mr
    kurmcl_reg_mr(&disp_meta_buf_send, src_info, recv_src_info_conut * sizeof(int16_t),
        ds_conn_info);                                                   // 注册发方meta mr
    kurmcl_reg_mr(&disp_buf_recv, disbuf_baseptr, disbuf_size, ds_conn_info);  // 注册收方mr
    disp_remote_buf = (buf_mr_info_t *)malloc(num_ranks * sizeof(buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &disp_buf_recv, disp_remote_buf);
    iovlist_for_disp =
        (kurmcl_iov_t *)malloc(num_ranks * num_local_experts * max_num_tokens * 2 * sizeof(kurmcl_iov_t));  // 申请iovlist
    iovcont_per_rank_for_disp.resize(num_ranks);
    will_recv_from_rank.resize(num_ranks);
    for (int i = 0; i < num_ranks; ++i) {
        iovcont_per_rank_for_disp[i] = 0;
    }
    // node based barrier
    int *nbb_send;
    int *nbb_recv;
    nbb_num_node = num_ranks / dtp;
    if (num_ranks % dtp != 0) {
        printf("Error: feature node_based need a unified ppn %d node_cnt %d\n", dtp, nbb_num_node);
    }
    if (dtp > 128) {
        printf("Error: feature node_based need ppn < 128 %d\n", dtp);
    }
    int nbb_byte_cnt =
        nbb_num_node * (dtp + 1) * sizeof(int);  // should send to every node and one extra Bype for count
    nbb_send = (int *)malloc(nbb_byte_cnt);
    nbb_recv = (int *)malloc(nbb_byte_cnt);
    kurmcl_reg_mr(&disp_nbb_send, nbb_send, nbb_byte_cnt, ds_conn_info);
    kurmcl_reg_mr(&disp_nbb_recv, nbb_recv, nbb_byte_cnt, ds_conn_info);
    disp_nbb_remote = (buf_mr_info_t *)malloc(num_ranks * sizeof(buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &disp_nbb_recv, disp_nbb_remote);
    // end node based barrier
    // for prefill metedata exchange
    iovlist_for_a2a =
        (kurmcl_iov_t *)malloc(num_ranks * sizeof(kurmcl_iov_t));  // 申请iovlist
    token_offset.resize(num_local_experts * num_ranks);
    token_offset_static.resize(num_local_experts * num_ranks);
    for (int i = 0; i < num_local_experts * num_ranks; ++i) {
        token_offset[i] = 0;
        token_offset_static[i] = 0;
    }
    send_count_ary_size = num_local_experts * num_ranks * sizeof(int);
    int *send_count_ary = (int *)malloc(send_count_ary_size);
    int *recv_count_ary = (int *)malloc(send_count_ary_size * num_ranks);
    kurmcl_reg_mr(&disp_send_count, send_count_ary, send_count_ary_size, ds_conn_info);
    kurmcl_reg_mr(&disp_recv_count, recv_count_ary, send_count_ary_size * num_ranks, ds_conn_info);
    disp_recv_count_remote = (buf_mr_info_t *)malloc(num_ranks * sizeof(buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &disp_recv_count, disp_recv_count_remote);
    // end prefill metedata exchange
    if (cfg.use_async_disp || cfg.use_async_comb) {
        async_tg_create(comm_thread_exec, asy_thread_num, asy_core_offset_in_numa);
        sleep(1);
    }
    for (int i = 0; i < recv_src_info_conut; ++i) {
        catch_recv_src_info[i] = -1;
        catch_recv_src_info_bak[i] = -1;
        src_info[i] = 0;
    }
    for (int i = 0; i < num_local_experts; ++i) {
        for (int j = 0; j < num_ranks; ++j) {
            recv_src_info_data[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = -1;
            recv_src_info_bak_data[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = -1;
            src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1)] = 0;
        }
    }
    kutacc_time_init();
}

void dispatch_async_dynamic(void *args)
{
    int num_threads = async_tg_get_thread_num();
    int tid = async_tg_get_thread_id();
    kuasync_dispatch_args_t *disp_args = (kuasync_dispatch_args_t *)args;
    int start;
    int end;
    kurmcl_conn_info_h ds_conn_info = disp_args->ds_conn_info;
    int num_ranks = ds_conn_info->comm_size;
    int my_rank = ds_conn_info->my_rank;
    int64_t num_max_dispatch_tokens_per_rank = disp_args->num_max_dispatch_tokens_per_rank;
    int64_t num_topk = disp_args->num_topk;
    int16_t *topk_idx_data = disp_args->topk_idx_data;
    int16_t hidden = disp_args->hidden;
    int64_t num_tokens = disp_args->num_tokens;
    int64_t target_recv_src_info_bias = disp_args->target_recv_src_info_bias;
    int start_token_idx;
    int end_token_idx;
    cal_task_range(local_rank, dtp, num_tokens, &start_token_idx, &end_token_idx);
    int64_t send_tokens_per_rank = end_token_idx - start_token_idx;
    int64_t recv_x_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t recv_x_rank_bias = num_max_dispatch_tokens_per_rank * hidden;

    int last_peer = -1;
    int64_t nbb_per_rank  = num_tokens / dtp;
    int nbb_start_token_idx = 0;
    int nbb_end_token_idx = num_tokens;
    int nbb_peer_rank;
    int nbb_dst_exp;
    int nbb_peer_node;
    int *nnb_sendbuf = (int *)disp_nbb_send.buffer;

    // iov_make
    if (tid == 0) {
        for (int i = 0; i < nbb_num_node; i++) {
            for (int j = 0; j < dtp; j++) {
                nnb_sendbuf[i * dtp + j] = 0;  // initialize bit
            }
        }
        for (int64_t i = nbb_start_token_idx; i < nbb_end_token_idx; i++) {
            int rem = num_tokens % dtp;
            int first_end = rem * (nbb_per_rank + 1);
            int src_rank = (i < first_end) ? (i / (nbb_per_rank + 1)) : (rem + (i - first_end) / nbb_per_rank);
            for (int64_t j = 0; j < num_topk; ++j) {
                nbb_peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
                if ((nbb_peer_rank % dtp) == (my_rank % dtp)) {   // local id inside node is the same
                    nbb_peer_node = nbb_peer_rank / dtp;              // node id of the peer rank
                    nnb_sendbuf[nbb_peer_node * dtp + src_rank] = 1;  // store the src rank id should be globalid
                }
            }
        }
        int64_t send_nums = 0;
        for (int64_t i = start_token_idx; i < end_token_idx; ++i) {
            for (int64_t j = 0; j < num_topk; ++j) {
                int64_t peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
                int64_t local_dst_exp = topk_idx_data[i * num_topk * 2 + j * 2 + 1];
                int64_t count_bias =
                    local_dst_exp * recv_src_info_ep_bias + peer_rank * (num_max_dispatch_tokens_per_rank * 2 + 1);
                int64_t count = src_info[count_bias];
                int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[peer_rank];
                int64_t iov_bias =
                    peer_rank * num_local_experts * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
                iovlist_for_disp[iov_bias].len = hidden;
                iovlist_for_disp[iov_bias].local_buffer = disp_token_buf_send.buffer + i * hidden;
                iovlist_for_disp[iov_bias].lkey = disp_token_buf_send.lkey;
                iovlist_for_disp[iov_bias].remote_buffer = disp_remote_buf[peer_rank].buffer +
                                                        local_dst_exp * recv_x_ep_bias + my_rank * recv_x_rank_bias +
                                                        count * hidden;
                iovlist_for_disp[iov_bias].rkey = disp_remote_buf[peer_rank].rkey;

                src_info[count_bias]++;
                src_info[count_bias + count * 2 + 1] = i;
                src_info[count_bias + count * 2 + 2] = send_nums;
                iovcont_per_rank_for_disp[peer_rank]++;
                send_nums++;
                disp_target_rank_indices[peer_rank] = true;
            }
        }
        for (int64_t j = 0; j < num_ranks; ++j) {
            for (int64_t i = 0; i < num_local_experts; ++i) {
                int64_t send_start_addr = i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1);
                int64_t meta_nums = src_info[send_start_addr];
                if (meta_nums > 0) {
                    int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[j];
                    int64_t iov_bias = j * num_local_experts * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
                    iovlist_for_disp[iov_bias].len = (meta_nums * 2 + 1) * 2;
                    iovlist_for_disp[iov_bias].local_buffer =
                        disp_meta_buf_send.buffer + send_start_addr * 2;
                    iovlist_for_disp[iov_bias].lkey = disp_meta_buf_send.lkey;
                    iovlist_for_disp[iov_bias].remote_buffer = disp_remote_buf[j].buffer + target_recv_src_info_bias +
                        (i * recv_src_info_ep_bias + my_rank * (num_max_dispatch_tokens_per_rank * 2 + 1)) * 2;
                    iovlist_for_disp[iov_bias].rkey = disp_remote_buf[j].rkey;
                    iovcont_per_rank_for_disp[j]++;
                    disp_target_rank_indices[j] = true;
                    if (last_peer != j) {
                        will_recv_from_rank[dis_peer_nums] = j;
                        dis_peer_nums++;
                        last_peer = j;
                    }
                }
            }
        }
        task_partition(disp_target_rank_indices, disp_iov_put_idxs, num_threads);
    }
    volatile int x;
    dis_asy_fence1++;
    do {
        kutacc_memory_cpu_load_fence();
        x = dis_asy_fence1;
    } while (x != num_threads);
    // nbb_put
    cal_task_range(tid, num_threads, nbb_num_node, &start, &end);
    int src_node_id = my_rank / dtp;
    for (int i = start; i < end; i++) {
        int peer_rank = my_rank % dtp;
        peer_rank = peer_rank + i * dtp;  // golbal id
        int offset_per_rank = dtp * sizeof(int);
        local_nbb_iov[tid].len = offset_per_rank;
        local_nbb_iov[tid].local_buffer = disp_nbb_send.buffer + i * offset_per_rank;
        local_nbb_iov[tid].lkey = disp_nbb_send.lkey;
        local_nbb_iov[tid].remote_buffer = disp_nbb_remote[peer_rank].buffer + src_node_id * offset_per_rank;
        local_nbb_iov[tid].rkey = disp_nbb_remote[peer_rank].rkey;
        kurmcl_put(&local_nbb_iov[tid], 1, peer_rank, 1, ds_conn_info);
    }
    ++dis_asy_fence2;
    do {
        kutacc_memory_cpu_load_fence();
        x = dis_asy_fence2;
    } while (x != num_threads);
    // iov_put and polling
    for (int i = disp_iov_put_idxs[tid][0]; i < disp_iov_put_idxs[tid][1]; ++i) {
        if (iovcont_per_rank_for_disp[i] > 0) {
            kurmcl_put(&(iovlist_for_disp[i * num_local_experts * num_max_dispatch_tokens_per_rank * 2]),
                iovcont_per_rank_for_disp[i], i, 1, ds_conn_info);
        }
    }
    for (int i = disp_iov_put_idxs[tid][0]; i < disp_iov_put_idxs[tid][1]; ++i) {
        iovcont_per_rank_for_disp[i] = 0;
    }
    if (tid == 0) {
        std::fill(disp_target_rank_indices.begin(), disp_target_rank_indices.end(), false);
    }
    int *nnb_recvbuf_tmp = (int *)disp_nbb_recv.buffer;
    cal_task_range(tid, num_threads, nbb_num_node, &start, &end);
    for (int i = start; i < end; ++i) {
        int src_rank = my_rank % dtp;
        src_rank = i * dtp + src_rank;
        kurmcl_recv_imm_cnt(1, src_rank, ds_conn_info);
        for (int j = 0; j < dtp; j++) {
            if (nnb_recvbuf_tmp[i * dtp + j] != 0) {
                src_rank = i * dtp + j;
                kurmcl_recv_imm_cnt(1, src_rank, ds_conn_info);
            }
        }
    }
}

void dispatch_async_static(void *args)
{
    int num_threads = async_tg_get_thread_num();
    int tid = async_tg_get_thread_id();
    kuasync_dispatch_args_t *disp_args = (kuasync_dispatch_args_t *)args;
    int start;
    int end;
    kurmcl_conn_info_h ds_conn_info = disp_args->ds_conn_info;
    int num_ranks = ds_conn_info->comm_size;
    int my_rank = ds_conn_info->my_rank;
    int64_t num_max_dispatch_tokens_per_rank = disp_args->num_max_dispatch_tokens_per_rank;
    int64_t num_topk = disp_args->num_topk;
    int16_t *topk_idx_data = disp_args->topk_idx_data;
    int16_t hidden = disp_args->hidden;
    int64_t num_tokens = disp_args->num_tokens;
    int64_t target_recv_src_info_bias = disp_args->target_recv_src_info_bias;
    int64_t send_tokens_per_rank = num_tokens / dtp;
    int16_t start_token_idx = (my_rank % dtp) * send_tokens_per_rank;
    int16_t end_token_idx = (my_rank % dtp + 1) * send_tokens_per_rank;
    int64_t recv_x_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t recv_x_rank_bias = num_max_dispatch_tokens_per_rank * hidden;

    int64_t node_id = my_rank / dtp;
    int64_t local_rank = my_rank % dtp;

    // iov_make
    if (tid == 0) {
        for (int64_t i = 0; i < num_tokens; ++i) {
            for (int64_t j = 0; j < num_topk; ++j) {
                int64_t peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
                if ((peer_rank % dtp) == (my_rank % dtp)) {
                    int64_t old_src_rank = node_id * dtp + (i / send_tokens_per_rank);  // global
                    int64_t local_old_src_rank = i / send_tokens_per_rank;              // local
                    int64_t local_leader_peer = peer_rank / dtp * dtp;
                    int64_t local_dst_exp = topk_idx_data[i * num_topk * 2 + j * 2 + 1];
                    int64_t index = local_dst_exp * recv_src_info_ep_bias +
                        (local_leader_peer + local_old_src_rank) * (num_max_dispatch_tokens_per_rank * 2 + 1);
                    int64_t cnt = src_info[index];
                    int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[peer_rank];
                    int64_t iov_bias = peer_rank * num_local_experts * num_tokens * 2 + peer_rank_iov_cnt;
                    iovlist_for_disp[iov_bias].len = hidden;
                    iovlist_for_disp[iov_bias].local_buffer = disp_token_buf_send.buffer + i * hidden;
                    iovlist_for_disp[iov_bias].lkey = disp_token_buf_send.lkey;
                    iovlist_for_disp[iov_bias].remote_buffer = disp_remote_buf[peer_rank].buffer +
                        local_dst_exp * recv_x_ep_bias + old_src_rank * recv_x_rank_bias + cnt * hidden;
                    iovlist_for_disp[iov_bias].rkey = disp_remote_buf[peer_rank].rkey;
                    iovcont_per_rank_for_disp[peer_rank]++;

                    src_info[index + cnt * 2 + 1] = i;
                    src_info[index + cnt * 2 + 2] = (i % send_tokens_per_rank) * num_topk + j;
                    src_info[index]++;
                }
            }
        }
        for (int64_t j = local_rank; j < num_ranks; j += dtp) {
            for (int64_t i = 0; i < num_local_experts; ++i) {
                int64_t local_leader_peer = j / dtp * dtp;
                int64_t send_start_addr =
                    i * recv_src_info_ep_bias + local_leader_peer * (num_max_dispatch_tokens_per_rank * 2 + 1);
                int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[j];
                int64_t iov_bias = j * num_local_experts * num_tokens * 2 + peer_rank_iov_cnt;
                iovlist_for_disp[iov_bias].len = (num_max_dispatch_tokens_per_rank * 2 + 1) * dtp * 2;
                iovlist_for_disp[iov_bias].local_buffer =
                    disp_meta_buf_send.buffer + (send_start_addr) * 2;
                iovlist_for_disp[iov_bias].lkey = disp_meta_buf_send.lkey;
                // 代表[ node_id*dtp , node_id*(dtp+1) )
                iovlist_for_disp[iov_bias].remote_buffer =
                    disp_remote_buf[j].buffer + target_recv_src_info_bias +
                    (i * recv_src_info_ep_bias + (node_id * dtp) * (num_max_dispatch_tokens_per_rank * 2 + 1)) * 2;
                iovlist_for_disp[iov_bias].rkey = disp_remote_buf[j].rkey;
                iovcont_per_rank_for_disp[j]++;
            }
        }
    }
    volatile int x;
    dis_asy_fence1++;
    do {
        kutacc_memory_cpu_load_fence();
        x = dis_asy_fence1;
    } while (x != num_threads);
    // partition
    cal_task_range(tid, num_threads, num_ranks / dtp, &start, &end);
    // iov_put and polling
    int total_task_nums = num_ranks / dtp;
    int used_thread_nums = std::min(num_threads, total_task_nums);
    int task_nums = total_task_nums / used_thread_nums;
    int start_rank = (my_rank + tid * (num_ranks / used_thread_nums)) % num_ranks;
    for (int i = 0; i < task_nums && tid < used_thread_nums; ++i) {
        int peer_rank = (start_rank + i * dtp) % num_ranks;
        if (iovcont_per_rank_for_disp[peer_rank] > 0) {
            kurmcl_put(&(iovlist_for_disp[peer_rank * num_local_experts * num_tokens * 2]),
                iovcont_per_rank_for_disp[peer_rank], peer_rank, 1, ds_conn_info);
        }
    }
    for (int i = 0; i < task_nums && tid < used_thread_nums; ++i) {
        int peer_rank = (start_rank + i * dtp) % num_ranks;
        iovcont_per_rank_for_disp[peer_rank] = 0;
        kurmcl_recv_imm_cnt(1, peer_rank, ds_conn_info);
    }
}

void dispatch_async(void *args)
{
    if (cfg.use_static_route) {
        dispatch_async_static(args);
    } else {
        dispatch_async_dynamic(args);
    }
}

void moe_dispatch_send_dynamic(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    dis_times++;
    int num_ranks = ds_conn_info->comm_size;
    int my_rank = ds_conn_info->my_rank;
    int64_t recv_x_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t recv_x_rank_bias = num_max_dispatch_tokens_per_rank * hidden;
    int64_t target_recv_src_info_bias;
    int local_rank = my_rank % dtp;
    int start_token_idx;
    int end_token_idx;
    cal_task_range(local_rank, dtp, num_tokens, &start_token_idx, &end_token_idx);
    int64_t send_tokens_per_rank = end_token_idx - start_token_idx;
    int tid;
    int last_peer = -1;
    if (dis_times % 2 == 1) {
        target_recv_src_info_bias = packed_recv_x_size;
    } else {
        target_recv_src_info_bias = packed_recv_x_size + recv_src_info_size;
    }
    if (cfg.use_async_disp) {
        dispatch_args.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;
        dispatch_args.num_topk = num_topk;
        dispatch_args.topk_idx_data = topk_idx_data;
        dispatch_args.hidden = hidden;
        dispatch_args.num_tokens = num_tokens;
        dispatch_args.target_recv_src_info_bias = target_recv_src_info_bias;
        dispatch_args.ds_conn_info = ds_conn_info;
        async_tg_task_submit(comm_thread_exec, dispatch_async, &dispatch_args);
    } else {
        // node based barrier
        int64_t nbb_per_rank = num_tokens / dtp;
        int nbb_start_token_idx = 0;
        int nbb_end_token_idx = num_tokens;
        int nbb_peer_rank;
        int nbb_dst_exp;
        int nbb_peer_node;
        int *nnb_sendbuf = (int *)disp_nbb_send.buffer;
        for (int i = 0; i < nbb_num_node; i++) {
            for (int j = 0; j < dtp; j++) {
                nnb_sendbuf[i * dtp + j] = 0;  // initialize bit
            }
        }
        for (int64_t i = nbb_start_token_idx; i < nbb_end_token_idx; i++) {
            int rem = num_tokens % dtp;
            int first_end = rem * (nbb_per_rank + 1);
            int src_rank = (i < first_end) ? (i / (nbb_per_rank + 1)) : (rem + (i - first_end) / nbb_per_rank);
            for (int64_t j = 0; j < num_topk; ++j) {
                nbb_peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
                if ((nbb_peer_rank % dtp) == (my_rank % dtp)) {   // local id inside node is the same
                    nbb_peer_node = nbb_peer_rank / dtp;              // node id of the peer rank
                    nnb_sendbuf[nbb_peer_node * dtp + src_rank] = 1;  // store the src rank id should be globalid
                }
            }
        }
        // end node based barrier
        int64_t send_nums = 0;
        for (int64_t i = start_token_idx; i < end_token_idx; ++i) {
            for (int64_t j = 0; j < num_topk; ++j) {
                int64_t peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
                int64_t local_dst_exp = topk_idx_data[i * num_topk * 2 + j * 2 + 1];
                int64_t count_bias =
                    local_dst_exp * recv_src_info_ep_bias + peer_rank * (num_max_dispatch_tokens_per_rank * 2 + 1);
                int64_t count = src_info[count_bias];
                int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[peer_rank];
                int64_t iov_bias =
                    peer_rank * num_local_experts * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
                iovlist_for_disp[iov_bias].len = hidden;
                iovlist_for_disp[iov_bias].local_buffer = disp_token_buf_send.buffer + i * hidden;
                iovlist_for_disp[iov_bias].lkey = disp_token_buf_send.lkey;
                iovlist_for_disp[iov_bias].remote_buffer = disp_remote_buf[peer_rank].buffer +
                                                        local_dst_exp * recv_x_ep_bias + my_rank * recv_x_rank_bias +
                                                        count * hidden;
                iovlist_for_disp[iov_bias].rkey = disp_remote_buf[peer_rank].rkey;

                src_info[count_bias]++;
                src_info[count_bias + count * 2 + 1] = i;
                src_info[count_bias + count * 2 + 2] = send_nums;
                iovcont_per_rank_for_disp[peer_rank]++;
                send_nums++;
                disp_target_rank_indices[peer_rank] = true;
            }
        }
        for (int64_t j = 0; j < num_ranks; ++j) {
            for (int64_t i = 0; i < num_local_experts; ++i) {
                int64_t send_start_addr = i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1);
                int64_t meta_nums = src_info[send_start_addr];
                if (meta_nums > 0) {
                    int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[j];
                    int64_t iov_bias = j * num_local_experts * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
                    iovlist_for_disp[iov_bias].len = (meta_nums * 2 + 1) * 2;
                    iovlist_for_disp[iov_bias].local_buffer =
                        disp_meta_buf_send.buffer + (send_start_addr) * 2;
                    iovlist_for_disp[iov_bias].lkey = disp_meta_buf_send.lkey;
                    iovlist_for_disp[iov_bias].remote_buffer =
                        disp_remote_buf[j].buffer + target_recv_src_info_bias +
                        (i * recv_src_info_ep_bias + my_rank * (num_max_dispatch_tokens_per_rank * 2 + 1)) * 2;
                    iovlist_for_disp[iov_bias].rkey = disp_remote_buf[j].rkey;
                    iovcont_per_rank_for_disp[j]++;
                    disp_target_rank_indices[j] = true;
                    if (last_peer != j) {
                        will_recv_from_rank[dis_peer_nums] = j;
                        dis_peer_nums++;
                        last_peer = j;
                    }
                }
            }
        }
        int thread_num = 16;
        int src_node_id = my_rank / dtp;
        task_partition(disp_target_rank_indices, disp_iov_put_idxs, thread_num);
        kutacc::parallel_for(0, thread_num, 1, [&](int64_t start, int64_t end) {
            int tid = start;
            kurmcl_iov_t local_nbb_iov;
            int start_idx = 0;
            int end_idx = 0;
            if (nbb_num_node >= thread_num) {
                start_idx = (nbb_num_node / thread_num) * tid;
                end_idx = (nbb_num_node / thread_num) * (tid + 1);
            } else {
                if (tid < nbb_num_node) {
                    start_idx = tid;
                    end_idx = tid+1;
                }
            }
            for (int i = start_idx; i < end_idx; ++i) {
                int peer_rank = my_rank % dtp;
                peer_rank = peer_rank + i * dtp;  // golbal id
                int offset_per_rank = dtp * sizeof(int);
                local_nbb_iov.len = offset_per_rank;
                local_nbb_iov.local_buffer = disp_nbb_send.buffer + i * offset_per_rank;
                local_nbb_iov.lkey = disp_nbb_send.lkey;
                local_nbb_iov.remote_buffer = disp_nbb_remote[peer_rank].buffer + src_node_id * offset_per_rank;
                local_nbb_iov.rkey = disp_nbb_remote[peer_rank].rkey;
                kurmcl_put(&local_nbb_iov, 1, peer_rank, 1, ds_conn_info);
            }
            volatile int x;
            ++put_fence;
            do {
                kutacc_memory_cpu_load_fence();
                x = put_fence;
            } while (x != thread_num);
            for (int i = disp_iov_put_idxs[tid][0]; i < disp_iov_put_idxs[tid][1]; ++i) {
                if (iovcont_per_rank_for_disp[i] > 0) {
                    kurmcl_put(&(iovlist_for_disp[i * num_local_experts * num_max_dispatch_tokens_per_rank * 2]),
                        iovcont_per_rank_for_disp[i], i, 1, ds_conn_info);
                }
            }
        });
        put_fence = 0;
    }
}

void moe_dispatch_send_static(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    dis_times++;
    int num_ranks = ds_conn_info->comm_size;
    int my_rank = ds_conn_info->my_rank;
    int64_t recv_x_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t recv_x_rank_bias = num_max_dispatch_tokens_per_rank * hidden;
    int64_t send_tokens_per_rank = num_tokens / dtp;
    int64_t target_recv_src_info_bias;
    int16_t start_token_idx = (my_rank % dtp) * send_tokens_per_rank;
    int16_t end_token_idx = (my_rank % dtp + 1) * send_tokens_per_rank;
    int tid;
    int last_peer = -1;
    if (dis_times % 2 == 1) {
        target_recv_src_info_bias = packed_recv_x_size;
    } else {
        target_recv_src_info_bias = packed_recv_x_size + recv_src_info_size;
    }
    if (cfg.use_async_disp) {
        dispatch_args.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;
        dispatch_args.num_topk = num_topk;
        dispatch_args.topk_idx_data = topk_idx_data;
        dispatch_args.hidden = hidden;
        dispatch_args.num_tokens = num_tokens;
        dispatch_args.target_recv_src_info_bias = target_recv_src_info_bias;
        dispatch_args.ds_conn_info = ds_conn_info;
        async_tg_task_submit(comm_thread_exec, dispatch_async, &dispatch_args);
    } else {
        int64_t node_id = my_rank / dtp;
        int64_t local_rank = my_rank % dtp;
        for (int64_t i = 0; i < num_tokens; ++i) {
            for (int64_t j = 0; j < num_topk; ++j) {
                int64_t peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
                if ((peer_rank % dtp) == (my_rank % dtp)) {
                    int64_t old_src_rank = node_id * dtp + (i / send_tokens_per_rank);  // global
                    int64_t local_old_src_rank = i / send_tokens_per_rank;              // local
                    int64_t local_leader_peer = peer_rank / dtp * dtp;
                    int64_t local_dst_exp = topk_idx_data[i * num_topk * 2 + j * 2 + 1];
                    int64_t index = local_dst_exp * recv_src_info_ep_bias +
                        (local_leader_peer + local_old_src_rank) * (num_max_dispatch_tokens_per_rank * 2 + 1);
                    int64_t cnt = src_info[index];
                    int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[peer_rank];
                    int64_t iov_bias = peer_rank * num_local_experts * num_tokens * 2 + peer_rank_iov_cnt;
                    iovlist_for_disp[iov_bias].len = hidden;
                    iovlist_for_disp[iov_bias].local_buffer = disp_token_buf_send.buffer + i * hidden;
                    iovlist_for_disp[iov_bias].lkey = disp_token_buf_send.lkey;
                    iovlist_for_disp[iov_bias].remote_buffer = disp_remote_buf[peer_rank].buffer +
                        local_dst_exp * recv_x_ep_bias + old_src_rank * recv_x_rank_bias + cnt * hidden;
                    iovlist_for_disp[iov_bias].rkey = disp_remote_buf[peer_rank].rkey;
                    iovcont_per_rank_for_disp[peer_rank]++;

                    src_info[index + cnt * 2 + 1] = i;
                    src_info[index + cnt * 2 + 2] = (i % send_tokens_per_rank) * num_topk + j;
                    src_info[index]++;
                }
            }
        }
        for (int64_t j = local_rank; j < num_ranks; j += dtp) {
            for (int64_t i = 0; i < num_local_experts; ++i) {
                int64_t local_leader_peer = j / dtp * dtp;
                int64_t send_start_addr = i * recv_src_info_ep_bias +
                    local_leader_peer * (num_max_dispatch_tokens_per_rank * 2 + 1);
                int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[j];
                int64_t iov_bias = j * num_local_experts * num_tokens * 2 + peer_rank_iov_cnt;
                iovlist_for_disp[iov_bias].len = (num_max_dispatch_tokens_per_rank * 2 + 1) * dtp * 2;
                iovlist_for_disp[iov_bias].local_buffer =
                    disp_meta_buf_send.buffer + (send_start_addr) * 2;
                iovlist_for_disp[iov_bias].lkey = disp_meta_buf_send.lkey;
                // 代表[ node_id*dtp , node_id*(dtp+1) )
                iovlist_for_disp[iov_bias].remote_buffer =
                    disp_remote_buf[j].buffer + target_recv_src_info_bias +
                    (i * recv_src_info_ep_bias + (node_id * dtp) * (num_max_dispatch_tokens_per_rank * 2 + 1)) * 2;
                iovlist_for_disp[iov_bias].rkey = disp_remote_buf[j].rkey;
                iovcont_per_rank_for_disp[j]++;
            }
        }
        kutacc::parallel_for(0, num_ranks / dtp, 1, [&](int64_t start, int64_t end) {
            for (int i = (start * dtp) + local_rank; i < (end * dtp) + local_rank; i += dtp) {
            kurmcl_put(&(iovlist_for_disp[i * num_local_experts * num_tokens * 2]), iovcont_per_rank_for_disp[i], i, 1,
                ds_conn_info);
            }
        });
    }
}

void moe_dispatch_send_prefill(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    dis_times++;
    int num_ranks = ds_conn_info->comm_size;
    int my_rank = ds_conn_info->my_rank;
    int64_t recv_x_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t recv_x_rank_bias = num_max_dispatch_tokens_per_rank * hidden;
    int64_t target_recv_src_info_bias;
    int start_token_idx;
    int end_token_idx;
    cal_task_range(local_rank, dtp, num_tokens, &start_token_idx, &end_token_idx);
    int64_t send_tokens_per_rank = end_token_idx - start_token_idx;
    int tid;
    int last_peer = -1;
    if (dis_times % 2 == 1) {
        target_recv_src_info_bias = packed_recv_x_size;
    } else {
        target_recv_src_info_bias = packed_recv_x_size + recv_src_info_size;
    }

    int *nnb_sendbuf = (int *)disp_send_count.buffer;
    int64_t send_nums = 0;
    for (int64_t i = start_token_idx; i < end_token_idx; ++i) {
        for (int64_t j = 0; j < num_topk; ++j) {
            int64_t peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
            int64_t local_dst_exp = topk_idx_data[i * num_topk * 2 + j * 2 + 1];
            int64_t count_bias =
                local_dst_exp * recv_src_info_ep_bias + peer_rank * (num_max_dispatch_tokens_per_rank * 2 + 1);
            int64_t count = src_info[count_bias];
            src_info[count_bias]++;
            src_info[count_bias + count * 2 + 1] = i;
            src_info[count_bias + count * 2 + 2] = send_nums;
            send_nums++;
        }
    }
    for (int64_t j = 0; j < num_ranks; ++j) {
        for (int64_t i = 0; i < num_local_experts; ++i) {
            int64_t send_start_addr = i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank * 2 + 1);
            int64_t meta_nums = src_info[send_start_addr];
            nnb_sendbuf[i * num_ranks + j] = meta_nums;
            if (meta_nums > 0) {
                int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[j];
                int64_t iov_bias = j * num_local_experts * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
                iovlist_for_disp[iov_bias].len = (meta_nums * 2 + 1) * 2;
                iovlist_for_disp[iov_bias].local_buffer =
                    disp_meta_buf_send.buffer + (send_start_addr) * 2;
                iovlist_for_disp[iov_bias].lkey = disp_meta_buf_send.lkey;
                iovlist_for_disp[iov_bias].remote_buffer =
                    disp_remote_buf[j].buffer + target_recv_src_info_bias +
                    (i * recv_src_info_ep_bias + my_rank * (num_max_dispatch_tokens_per_rank * 2 + 1)) * 2;
                iovlist_for_disp[iov_bias].rkey = disp_remote_buf[j].rkey;
                iovcont_per_rank_for_disp[j]++;
                disp_target_rank_indices[j] = true;
                if (last_peer != j) {
                    will_recv_from_rank[dis_peer_nums] = j;
                    dis_peer_nums++;
                    last_peer = j;
                }
            }
        }
    }
    kutacc::parallel_for(0, num_ranks, 1, [&](int64_t start, int64_t end) {
        for (int i = start; i < end; ++i) {
            iovlist_for_a2a[i].len = send_count_ary_size;
            iovlist_for_a2a[i].local_buffer = disp_send_count.buffer;
            iovlist_for_a2a[i].lkey = disp_send_count.lkey;
            iovlist_for_a2a[i].remote_buffer = disp_recv_count_remote[i].buffer + my_rank * send_count_ary_size;
            iovlist_for_a2a[i].rkey = disp_recv_count_remote[i].rkey;
            kurmcl_put(&iovlist_for_a2a[i], 1, i, 1, ds_conn_info);
        }
    });
    kutacc::parallel_for(0, num_ranks, 1, [&](int64_t start, int64_t end) {
        for (int i = start; i < end; ++i) {
            kurmcl_recv_imm_cnt(1, i, ds_conn_info);
        }
    });
    kurmcl_flush(1, ds_conn_info);
    int *nnb_recvbuf_tmp = (int *)disp_recv_count.buffer;
    kutacc::parallel_for(0, num_ranks, 1, [&](int64_t start, int64_t end) {
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < num_local_experts; ++j) {
                for (int k = 0; k < my_rank; ++k) {
                    token_offset[j * num_ranks + i] +=
                        nnb_recvbuf_tmp[k * num_local_experts * num_ranks + j * num_ranks + i];
                    token_offset_static[j * num_ranks + i] +=
                        nnb_recvbuf_tmp[k * num_local_experts * num_ranks + j * num_ranks + i];
                }
            }
        }
    });
    for (int64_t i = start_token_idx; i < end_token_idx; ++i) {
        for (int64_t j = 0; j < num_topk; ++j) {
            int64_t peer_rank = topk_idx_data[i * num_topk * 2 + j * 2];
            int64_t local_dst_exp = topk_idx_data[i * num_topk * 2 + j * 2 + 1];
            int64_t count_bias =
                local_dst_exp * recv_src_info_ep_bias + peer_rank * (num_max_dispatch_tokens_per_rank * 2 + 1);
            int64_t count = src_info[count_bias];
            int64_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[peer_rank];
            int64_t iov_bias = peer_rank * num_local_experts * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
            iovlist_for_disp[iov_bias].len = hidden;
            iovlist_for_disp[iov_bias].local_buffer = disp_token_buf_send.buffer + i * hidden;
            iovlist_for_disp[iov_bias].lkey = disp_token_buf_send.lkey;
            iovlist_for_disp[iov_bias].remote_buffer = disp_remote_buf[peer_rank].buffer +
                local_dst_exp * 2 * max_num_tokens * hidden +
                token_offset[local_dst_exp * num_ranks + peer_rank] * hidden;
            iovlist_for_disp[iov_bias].rkey = disp_remote_buf[peer_rank].rkey;
            iovcont_per_rank_for_disp[peer_rank]++;
            disp_target_rank_indices[peer_rank] = true;
            token_offset[local_dst_exp * num_ranks + peer_rank]++;
        }
    }
    int thread_num = 32;
    task_partition(disp_target_rank_indices, disp_iov_put_idxs, thread_num);
    kutacc::parallel_for(0, thread_num, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        for (int i = disp_iov_put_idxs[tid][0]; i < disp_iov_put_idxs[tid][1]; ++i) {
            if (iovcont_per_rank_for_disp[i] > 0) {
                kurmcl_put(&(iovlist_for_disp[i * num_local_experts * num_max_dispatch_tokens_per_rank * 2]),
                    iovcont_per_rank_for_disp[i], i, 1, ds_conn_info);
            }
        }
    });
}

void moe_dispatch_send(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    KUTACC_CHECK(num_tokens % dtp == 0, "BatchSize must be divisible by dtp");
    if (cfg.is_prefill) {
        moe_dispatch_send_prefill(x_data, topk_idx_data, num_tokens, num_topk, num_max_dispatch_tokens_per_rank, hidden,
                                parallel_policy_data, batch_id, ds_conn_info);
    } else if (cfg.use_static_route) {
        moe_dispatch_send_static(x_data, topk_idx_data, num_tokens, num_topk, num_max_dispatch_tokens_per_rank, hidden,
                                parallel_policy_data, batch_id, ds_conn_info);
    } else {
        moe_dispatch_send_dynamic(x_data, topk_idx_data, num_tokens, num_topk, num_max_dispatch_tokens_per_rank, hidden,
                                parallel_policy_data, batch_id, ds_conn_info);
    }
}

void moe_dispatch_recv_dynamic(int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    if (cfg.use_async_disp) {
        async_tg_wait(comm_thread_exec);
        kurmcl_flush(dis_peer_nums + nbb_num_node, ds_conn_info);
        dis_asy_fence1 = 0;
        dis_asy_fence2 = 0;
    } else {
        int num_ranks = ds_conn_info->comm_size;
        int my_rank = ds_conn_info->my_rank;
        for (int i = 0; i < num_ranks; ++i) {
            iovcont_per_rank_for_disp[i] = 0;
        }
        std::fill(disp_target_rank_indices.begin(), disp_target_rank_indices.end(), false);
        int *nnb_recvbuf_tmp = (int *)disp_nbb_recv.buffer;
        kutacc::parallel_for(0, nbb_num_node, 1, [&](int64_t start, int64_t end) {
            for (int i = start; i < end; ++i) {
                int src_rank = my_rank % dtp;                 // local id in node
                src_rank = i * dtp + src_rank;                // global id
                kurmcl_recv_imm_cnt(1, src_rank, ds_conn_info);  // node info recveived, recv token info
                for (int j = 0; j < dtp; j++) {
                    if (nnb_recvbuf_tmp[i * dtp + j] != 0) {
                        src_rank = i * dtp + j;
                        kurmcl_recv_imm_cnt(1, src_rank, ds_conn_info);
                    }
                }
            }
        });
        kurmcl_flush(1, ds_conn_info);
    }
}

void moe_dispatch_recv_static(int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    if (cfg.use_async_disp) {
        async_tg_wait(comm_thread_exec);
        if (cfg.use_static_route) {
            kurmcl_flush(num_tokens / dtp, ds_conn_info);
        } else {
            kurmcl_flush(dis_peer_nums + nbb_num_node, ds_conn_info);
        }
        dis_asy_fence1 = 0;
        dis_asy_fence2 = 0;
    } else {
        int num_ranks = ds_conn_info->comm_size;
        int my_rank = ds_conn_info->my_rank;
        for (int i = 0; i < num_ranks; ++i) {
            iovcont_per_rank_for_disp[i] = 0;
        }
        kutacc::parallel_for(0, num_ranks / dtp, 1, [&](int64_t start, int64_t end) {
            for (int i = (start * dtp) + local_rank; i < (end * dtp) + local_rank; i += dtp) {
                kurmcl_recv_imm_cnt(1, i, ds_conn_info);
            }
        });
        kurmcl_flush(1, ds_conn_info);
    }
}

void moe_dispatch_recv_prefill(int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    int num_ranks = ds_conn_info->comm_size;
    int my_rank = ds_conn_info->my_rank;
    for (int i = 0; i < num_ranks; ++i) {
        iovcont_per_rank_for_disp[i] = 0;
    }
    int *nnb_recvbuf_tmp = (int *)disp_recv_count.buffer;
    for (int i = 0; i < num_ranks; ++i) {
        int recv_count = 0;
        for (int j = 0; j < num_local_experts; ++j) {
            recv_count += nnb_recvbuf_tmp[i * num_local_experts * num_ranks + j * num_ranks + my_rank];
        }
        if (recv_count > 0) {
            kurmcl_recv_imm_cnt(1, i, ds_conn_info);
        }
    }
    kurmcl_flush(1, ds_conn_info);
    for (int i = 0; i < num_local_experts; ++i) {
        for (int j = 0; j < num_ranks; ++j) {
            token_offset[i * num_ranks + j] = 0;
            token_offset_static[i * num_ranks + j] = 0;
        }
    }
}

void moe_dispatch_recv(int64_t batch_id, kurmcl_conn_info_h ds_conn_info)
{
    if (cfg.is_prefill) {
        moe_dispatch_recv_prefill(batch_id, ds_conn_info);
    } else if (cfg.use_static_route) {
        moe_dispatch_recv_static(batch_id, ds_conn_info);
    } else {
        moe_dispatch_recv_dynamic(batch_id, ds_conn_info);
    }
}

void moe_dispatch_finalize()
{
    kurmcl_dreg_mr(&disp_token_buf_send);
    kurmcl_dreg_mr(&disp_meta_buf_send);
    free(iovlist_for_disp);
}

}  // namespace kutacc
