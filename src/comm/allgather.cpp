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
#include "comm.h"
#include "kutacc.h"
#include "utils/collapse.h"
#include "utils/check.h"
#include "utils/timer.h"
#include "core/kurmcl/kurmcl_impl.h"

#include <cstdint>
#include <atomic>
#include <functional>
#include <kupl.h>
#include <iostream>
#include <cstring>
#include <arm_sve.h>


namespace kutacc {

namespace {

buf_mr_info_t allg_buf_send, allg_buf_recv;
buf_mr_info_t *allgather_remote_buf;
kurmcl_iov_t *iovlist_for_allgather;
std::vector<void *> allgather_group_ptr;
int64_t max_buf_size;

template <typename scalar_t>
struct SVEParams;

template <>
struct SVEParams<uint8_t> {
    static constexpr int vector_elements = 64;
    static inline svbool_t create_predicate(uint64_t i, uint64_t length)
    {
        return svwhilelt_b8(i, length);
    }
};

template <>
struct SVEParams<bfloat16_t> {
    static constexpr int vector_elements = 32;
    static inline svbool_t create_predicate(uint64_t i, uint64_t length)
    {
        return svwhilelt_b16(i, length);
    }
};

template <>
struct SVEParams<float> {
    static constexpr int vector_elements = 16;
    static inline svbool_t create_predicate(uint64_t i, uint64_t length)
    {
        return svwhilelt_b32(i, length);
    }
};

template <typename scalar_t>
void copy_buffer(scalar_t *dst, const scalar_t *src, int64_t numel, int64_t grain_size = 8192)
{
    int64_t real_grain_size = grain_size / sizeof(scalar_t);
    if (real_grain_size == 0) {
        real_grain_size = 1;
    }
    kutacc::parallel_for(0, numel, real_grain_size,
        [&](int64_t start, int64_t end) { memcpy(dst + start, src + start, (end - start) * sizeof(scalar_t)); });
}

}

template <typename scalar_t>
void shm_batch_allgather_comm8(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                               int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
                               const std::function<void()> &barrier)
{
    KUTACC_CHECK(sendbuf == (scalar_t *)buffers[rank], "no implement");
    const int64_t socket_size = 8;
    KUTACC_CHECK(recvcount == sendcount * socket_size, "allgather");
    const int64_t thread_nums = kutacc::get_thread_num();
    const int64_t total_batch = batch * socket_size;
    const int64_t total_thread_batch = (total_batch + thread_nums - 1) / thread_nums;
    const int64_t start_rank = (rank / socket_size) * socket_size;
    barrier();
    // step 1: copy out
    kutacc::parallel_for(0, thread_nums, 1, [&](int64_t start, int64_t end) {
        const int64_t thread_id = start;
        const int64_t bs_start = thread_id * total_thread_batch;
        const int64_t bs_end = std::min(total_batch, bs_start + total_thread_batch);
        const int64_t bs_num = bs_end - bs_start;
        if (bs_num > 0) {
            int64_t recv_bs_offset = bs_start % batch;
            int64_t recv_w_offset = bs_start / batch * sendcount;
            int64_t cur_rank = bs_start / batch + start_rank;
            for (int64_t i = 0; i < bs_num; i++) {
                memcpy(recvbuf + recv_bs_offset * (sendcount * socket_size) + recv_w_offset,
                    ((scalar_t *)buffers[cur_rank]) + recv_bs_offset * sendcount, sendcount * sizeof(scalar_t));
                recv_bs_offset++;
                if (recv_bs_offset == batch) {
                    recv_w_offset += sendcount;
                    recv_bs_offset = 0;
                    cur_rank++;
                }
            }
        }
    });
    barrier();
}

typedef struct shm_allgather_request {
    int rank;
    int comm_size;
    int global_rank;
    int global_size;
    kupl_shm_win_h shm_win_intra_die;
    kupl_shm_win_h shm_win_intra_socket;
    kupl_shm_win_h shm_win_intra_node;
} *shm_allgather_request_h;

void shm_allgather_request_create(int rank, int comm_size, shm_allgather_request_h &request)
{
    int num_threads = kutacc::get_thread_num();
    KUTACC_CHECK(comm_size == 16 || comm_size == 8, "");
    request = (shm_allgather_request *)malloc(sizeof(shm_allgather_request));

    request->rank = rank;
    request->comm_size = comm_size;
}

void shm_allgather_request_init(kupl_shm_win_h shm_win_intra_die,
                                kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node,
                                const shm_allgather_request_h &request)
{
    request->shm_win_intra_die = shm_win_intra_die;
    request->shm_win_intra_socket = shm_win_intra_socket;
    request->shm_win_intra_node = shm_win_intra_node;
    kupl_shm_fence(shm_win_intra_node);
}

void shm_allgather_request_destroy(const shm_allgather_request_h &request)
{
    free(request);
}

template <typename scalar_t>
void shm_batch_allgather_comm8_optimize(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
    int64_t recvcount, uint8_t** remote_send, uint8_t** remote_recv, const shm_allgather_request_h &request)
{
    const int64_t socket_size = 8;
    KUTACC_CHECK(recvcount == sendcount * socket_size, "allgather");
    const int64_t thread_nums = kutacc::get_thread_num();
    int64_t rank = request->rank;
    int64_t my_dieleader = rank < 4 ? 0 : 4;
    int64_t bs_per_thread;
    int64_t remaining_bs;
    int64_t actual_threads;
    if (batch <= thread_nums) {
        actual_threads = batch;
        bs_per_thread = 1;
        remaining_bs = 0;
    } else {
        actual_threads = thread_nums;
        bs_per_thread = batch / thread_nums;
        remaining_bs = batch % thread_nums;
    }
    int64_t read_from_rank[8] = {4, 5, 6, 7, 0, 1, 2, 3};
    kupl_shm_fence(request->shm_win_intra_socket);
    kutacc::parallel_for(0, actual_threads, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int64_t bs_start;
        int64_t bs_end;
        if (tid == actual_threads - 1) {
            bs_start = tid * bs_per_thread;
            bs_end = bs_start + bs_per_thread + remaining_bs;
        } else {
            bs_start = tid * bs_per_thread;
            bs_end = bs_start + bs_per_thread;
        }
        int64_t i = 0;
        int64_t part_length = (bs_end - bs_start) * sendcount;
        for (int bs_num = bs_start; bs_num < bs_end; ++bs_num) {
            scalar_t *sendbuf_part = (scalar_t *)sendbuf + bs_num * sendcount;
            scalar_t *read_from = (scalar_t *)remote_send[read_from_rank[rank]] + bs_num * sendcount;
            for (i = 0; i < sendcount; i += SVEParams<scalar_t>::vector_elements) {
#ifdef ENABLE_EXTRA_PREFETCH
                constexpr int prf_stride = 3 * 1024 / 8 & ~63;
                svprfh(svwhilelt_b16(i + prf_stride / 2, sendcount), sendbuf_part + i + prf_stride / 2, SV_PLDL2STRM);
                svprfh(svwhilelt_b16(i + prf_stride / 2, sendcount), read_from + i + prf_stride / 2, SV_PLDL2STRM);
#endif
                svbool_t pred = SVEParams<scalar_t>::create_predicate(i, sendcount);
                auto sve1 = svld1(pred, sendbuf_part + i);
                auto sve2 = svld1(pred, read_from + i);
                for (int peer = my_dieleader; peer < my_dieleader + 4; ++peer) {
                    svst1(pred, (scalar_t *)remote_recv[peer] + bs_num * recvcount + rank * sendcount + i, sve1);
                    svst1(pred,
                        (scalar_t *)remote_recv[peer] + bs_num * recvcount + read_from_rank[rank] * sendcount + i,
                        sve2);
                }
            }
        }
    });
    kupl_shm_fence(request->shm_win_intra_socket);
}

template <typename scalar_t>
void shm_batch_allgather_comm8_hierarchical(int64_t batch, const scalar_t *sendbuf, int64_t sendcount,
    scalar_t *recvbuf, int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
    const std::function<void()> &barrier)
{
    KUTACC_CHECK(sendbuf == (scalar_t *)buffers[rank], "no implement");
    int64_t half_size = buffer_size / 2 / sizeof(scalar_t);
    int64_t socket_size = 8;
    int64_t die_size = 4;
    KUTACC_CHECK(sendcount * socket_size == recvcount, sendcount, " ", recvcount);
    int64_t socket_root = rank / socket_size * socket_size;
    int64_t die_root = rank / die_size * die_size;
    int64_t part_numel = batch * sendcount;
    // step 1: inter-die gather
    {
        int64_t remote_rank = rank ^ die_size;
        auto src = (scalar_t *)buffers[rank];
        auto dst = (scalar_t *)buffers[remote_rank] + half_size;
        kutacc::parallel_for(0, part_numel, 8192 / sizeof(scalar_t),
            [&](int64_t start, int64_t end) {
                memcpy(dst + start, src + start, (end - start) * sizeof(scalar_t));
            });
        barrier();
    }
    // step 2: intra-die gather
    {
        kutacc::parallel_for(0, socket_size * batch, 1, [&](int64_t start, int64_t end) {
            kutacc::collapse_for(start, end, socket_size, batch, [&](int64_t remote_rank_off, int64_t bi) {
                int64_t remote_rank = socket_root + remote_rank_off;
                bool same_die = (rank / die_size == remote_rank / die_size);
                auto src = (scalar_t *)buffers[die_root + remote_rank % die_size] + (same_die ? 0 : half_size) +
                        bi * sendcount;
                auto dst = recvbuf + bi * recvcount + remote_rank_off * sendcount;
                memcpy(dst, src, sendcount * sizeof(scalar_t));
            });
        });
        barrier();
    }
}

template <typename scalar_t>
void shm_batch_allgather_comm16(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                                int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
                                const std::function<void()> &barrier)
{
    constexpr int world_size = 16;
    KUTACC_CHECK(sendbuf == (scalar_t *)buffers[rank], "no implement");
    KUTACC_CHECK(recvcount == sendcount * world_size, "allgather recvcount");
    const int64_t thread_nums = kutacc::get_thread_num();
    const int64_t total_batch = batch * world_size;
    const int64_t total_thread_batch = (total_batch + thread_nums - 1) / thread_nums;
    barrier();
    kutacc::parallel_for(0, thread_nums, 1, [&](int64_t start, int64_t end) {
        const int64_t thread_id = start;
        const int64_t bs_start = thread_id * total_thread_batch;
        const int64_t bs_end = std::min(total_batch, bs_start + total_thread_batch);
        const int64_t bs_num = bs_end - bs_start;
        int64_t cur_rank = bs_start / batch;
        int64_t recv_bs_offset = bs_start % batch;
        int64_t recv_w_offset = cur_rank * sendcount;
        for (int64_t i = 0; i < bs_num; i++) {
            memcpy(recvbuf + recv_bs_offset * (sendcount * world_size) + recv_w_offset,
                ((scalar_t *)buffers[cur_rank]) + recv_bs_offset * sendcount, sendcount * sizeof(scalar_t));
            recv_bs_offset++;
            if (recv_bs_offset == batch) {
                recv_w_offset += sendcount;
                recv_bs_offset = 0;
                cur_rank++;
            }
        }
    });
    barrier();
}

template <typename scalar_t>
void shm_batch_allgather_comm16_optimize(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
    int64_t recvcount, uint8_t** remote_send, uint8_t** remote_recv, const shm_allgather_request_h &request)
{
    const int64_t world_size = 16;
    KUTACC_CHECK(recvcount == sendcount * world_size, "allgather");
    const int64_t thread_nums = kutacc::get_thread_num();
    int64_t rank = request->rank;
    int64_t my_socketleader = rank < 8 ? 0 : 8;
    int64_t bs_per_thread;
    int64_t remaining_bs;
    int64_t actual_threads;
    if (batch <= thread_nums) {
        actual_threads = batch;
        bs_per_thread = 1;
        remaining_bs = 0;
    } else {
        actual_threads = thread_nums;
        bs_per_thread = batch / thread_nums;
        remaining_bs = batch % thread_nums;
    }
    int64_t read_from_rank[16] = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};
    kupl_shm_fence(request->shm_win_intra_node);
    kutacc::parallel_for(0, actual_threads, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int64_t bs_start;
        int64_t bs_end;
        if (tid == actual_threads - 1) {
            bs_start = tid * bs_per_thread;
            bs_end = bs_start + bs_per_thread + remaining_bs;
        } else {
            bs_start = tid * bs_per_thread;
            bs_end = bs_start + bs_per_thread;
        }
        int i = 0;
        int64_t part_length = (bs_end - bs_start) * sendcount;
        for (int bs_num = bs_start; bs_num < bs_end; ++bs_num) {
            scalar_t *sendbuf_part = (scalar_t *)sendbuf + bs_num * sendcount;
            scalar_t *read_from = (scalar_t *)remote_send[read_from_rank[rank]] + bs_num * sendcount;
            for (i = 0; i < sendcount; i += SVEParams<scalar_t>::vector_elements) {
                svbool_t pred = SVEParams<scalar_t>::create_predicate(i, sendcount);
                auto sve1 = svld1(pred, sendbuf_part + i);
                auto sve2 = svld1(pred, read_from + i);
                for (int peer = my_socketleader; peer < my_socketleader + 8; ++peer) {
                    svst1(pred, (scalar_t *)remote_recv[peer] + bs_num * recvcount + rank * sendcount + i, sve1);
                    svst1(pred,
                        (scalar_t *)remote_recv[peer] + bs_num * recvcount + read_from_rank[rank] * sendcount + i,
                        sve2);
                }
            }
        }
    });
    kupl_shm_fence(request->shm_win_intra_node);
}

template <typename scalar_t>
void shm_batch_allgather_comm16_hierarchical(int64_t batch, const scalar_t *sendbuf, int64_t sendcount,
                                             scalar_t *recvbuf, int64_t recvcount, int64_t rank, uint8_t **buffers,
                                             int64_t buffer_size, const std::function<void()> &barrier)
{
    constexpr int world_size = 16;
    int64_t quarter_size = buffer_size / 4 / sizeof(scalar_t);
    int64_t count = sendcount;
    KUTACC_CHECK(sendbuf != (scalar_t *)buffers[rank], "no implement");
    KUTACC_CHECK(quarter_size >= count, "allgather sendcount too large");
    KUTACC_CHECK(recvcount == sendcount * world_size, "allgather recvcount");
    int64_t socket_size = world_size / 2;
    int64_t die_size = socket_size / 2;
    int64_t die_root = rank / die_size * die_size;
    int64_t batch_step = quarter_size / count;
    for (int64_t batch_start = 0; batch_start < batch; batch_start += batch_step) {
        int64_t batch_end = std::min(batch, batch_start + batch_step);
        int64_t chunk_numel = (batch_end - batch_start) * count;
        const scalar_t *send_ptr = sendbuf + batch_start * count;
        scalar_t *recv_ptr = recvbuf + batch_start * recvcount;
        // step 1 & 2: copy in & inter-socket gather
        {
            int64_t rank_offset = (rank / die_size) * quarter_size;
            int64_t remote_rank = (rank + socket_size) % world_size;
            kutacc::parallel_for(0, chunk_numel, 1, [&](int64_t start, int64_t end) {
                memcpy((scalar_t *)buffers[rank] + rank_offset + start, send_ptr + start,
                    (end - start) * sizeof(scalar_t));
                memcpy((scalar_t *)buffers[remote_rank] + rank_offset + start, send_ptr + start,
                    (end - start) * sizeof(scalar_t));
            });
            barrier();
        }
        // step 3: inter-die gather
        {
            int64_t local_rank = (rank + die_size) % socket_size + (rank >= socket_size ? socket_size : 0);
            int64_t local_offset = (1 - rank % socket_size / die_size) * quarter_size;
            copy_buffer((scalar_t *)buffers[rank] + local_offset, (scalar_t *)buffers[local_rank] + local_offset,
                chunk_numel, sizeof(scalar_t));
            copy_buffer((scalar_t *)buffers[rank] + local_offset + quarter_size * 2,
                (scalar_t *)buffers[local_rank] + local_offset + quarter_size * 2, chunk_numel, sizeof(scalar_t));
            barrier();
        }
        // step 4: copy out
        {
            if (count > 128)
                kutacc::parallel_for(0, world_size * batch_step, 1, [&](int64_t start, int64_t end) {
                    kutacc::collapse_for(start, end, world_size, batch_step, [&](int64_t remote_rank, int64_t bi) {
                        if (batch_start + bi < batch) {
                            memcpy(recv_ptr + bi * world_size * count + remote_rank * count,
                                (scalar_t *)buffers[die_root + remote_rank % die_size] +
                                    (remote_rank / die_size) * quarter_size + bi * count,
                                count * sizeof(scalar_t));
                        }
                    });
                });
            else
                kutacc::parallel_for(batch_start, batch_end, 1, [&](int64_t start, int64_t end) {
                    for (int64_t bi = start; bi < end; ++bi)
                        for (int64_t i = 0; i < world_size; ++i) {
                            int64_t now_rank = (rank + i) % world_size;
                            memcpy(recv_ptr + bi * world_size * count + now_rank * count,
                                (scalar_t *)buffers[die_root + now_rank % die_size] +
                                    (now_rank / die_size) * quarter_size + bi * count,
                                count * sizeof(scalar_t));
                        }
                });
            barrier();
        }
    }
}

template <typename scalar_t>
void shm_batch_allgather_base(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
    int64_t recvcount, uint8_t **buffers, int64_t buffer_size, const shm_allgather_request_h &request,
    bool is_hierarchical)
{
    int64_t rank = request->rank;    // 0 ~ 16
    int64_t world_size = request->comm_size;
    if (world_size == 8) {
        const std::function<void()> barrier = [&]() {kupl_shm_fence(request->shm_win_intra_socket);};
        if (is_hierarchical) {
            shm_batch_allgather_comm8_hierarchical(
                batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        } else {
            shm_batch_allgather_comm8(
                batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        }
    } else if (world_size == 16) {
        const std::function<void()> barrier = [&]() {kupl_shm_fence(request->shm_win_intra_node);};
        if (is_hierarchical) {
            shm_batch_allgather_comm16_hierarchical(
                batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        } else {
            shm_batch_allgather_comm16(
                batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        }
    } else {
        KUTACC_CHECK(false, "no implement");
    }
}

template <typename scalar_t>
void shm_batch_allgather_optimize(int64_t batch, scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
    int64_t recvcount, uint8_t **remote_sendbuf, uint8_t **remote_recvbuf, const shm_allgather_request_h &request)
{
    if (request->comm_size == 16) {
        shm_batch_allgather_comm16_optimize(
            batch, sendbuf, sendcount, recvbuf, recvcount, remote_sendbuf, remote_recvbuf, request);
    } else if (request->comm_size == 8) {
        shm_batch_allgather_comm8_optimize(
            batch, sendbuf, sendcount, recvbuf, recvcount, remote_sendbuf, remote_recvbuf, request);
    } else {
        KUTACC_CHECK(false, "no implement for comm_size = ", request->comm_size);
    }
}

void shm_batch_allgather(int64_t batch, void *sendbuf, int64_t sendcount, void *recvbuf,
    int64_t recvcount, shm_datatype_t datatype, uint8_t **remote_sendbuf, uint8_t **remote_recvbuf,
    int64_t buffer_size, const shm_allgather_request_h &request,
    bool is_hierarchical, bool use_base)
{
    KUTACC_CHECK(datatype == SHM_DATATYPE_UINT8 || datatype == SHM_DATATYPE_BFLOAT16, "");
    if (use_base) {
        if (datatype == SHM_DATATYPE_UINT8) {
            shm_batch_allgather_base(batch, (uint8_t *)sendbuf, sendcount, (uint8_t *)recvbuf, recvcount,
                remote_recvbuf, buffer_size, request, is_hierarchical);
        } else {
            shm_batch_allgather_base(batch, (bfloat16_t *)sendbuf, sendcount, (bfloat16_t *)recvbuf, recvcount,
                remote_recvbuf, buffer_size, request, is_hierarchical);
        }
    } else {
        if (datatype == SHM_DATATYPE_UINT8) {
            shm_batch_allgather_optimize(batch, (uint8_t *)sendbuf, sendcount, (uint8_t *)recvbuf, recvcount,
                remote_sendbuf, remote_recvbuf, request);
        } else {
            shm_batch_allgather_optimize(batch, (bfloat16_t *)sendbuf, sendcount, (bfloat16_t *)recvbuf, recvcount,
                remote_sendbuf, remote_recvbuf, request);
        }
    }
}

void shm_allgather_init(void *allgather_sendbuf, void *allgather_recvbuf, int64_t max_buf_size_,
                        kurmcl_conn_info_h ds_conn_info)
{
    max_buf_size = max_buf_size_;
    int64_t group_rank = ds_conn_info->my_rank % 16;
    kurmcl_reg_mr(&allg_buf_send, allgather_sendbuf, max_buf_size, ds_conn_info);
    kurmcl_reg_mr(&allg_buf_recv, allgather_recvbuf, max_buf_size, ds_conn_info);
    allgather_remote_buf = (buf_mr_info_t*)malloc(ds_conn_info->comm_size * sizeof (buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &allg_buf_recv, allgather_remote_buf);
    iovlist_for_allgather = (kurmcl_iov_t *)malloc(16 * sizeof(kurmcl_iov_t));
}

template <typename scalar_t>
void shm_allgather_comm16(const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf, int64_t recvcount,
    int64_t rank, uint8_t **buffers, kurmcl_conn_info_h ds_conn_info, const std::function<void()> &barrier)
{
    KUTACC_CHECK(sendcount <= max_buf_size,
        "input sendcount must less than max_buf_size you applied in shm_allgather_standard_init");
    int part_conut = sendcount / 8;
    int64_t myleader_rank = rank < 8 ? 0 : 8;
    int64_t peerleader_rank = rank < 8 ? 8 : 0;
    int64_t group_id = ds_conn_info->my_rank / 16;
    int64_t group_rank = ds_conn_info->my_rank % 16;
    kutacc::parallel_for(0, 8, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int peer = tid + peerleader_rank;
        iovlist_for_allgather[tid].len = part_conut * sizeof(scalar_t);
        iovlist_for_allgather[tid].local_buffer = allg_buf_send.buffer +
            (group_rank * sendcount + tid * part_conut) * sizeof(scalar_t);
        iovlist_for_allgather[tid].lkey = allg_buf_send.lkey;
        iovlist_for_allgather[tid].remote_buffer = allgather_remote_buf[peer].buffer +
            (group_rank * sendcount + tid * part_conut) * sizeof(scalar_t);
        iovlist_for_allgather[tid].rkey = allgather_remote_buf[peer].rkey;
        kurmcl_put(&(iovlist_for_allgather[tid]), 1, peer, 1, ds_conn_info);
        int i = 0;
        scalar_t *sendbuf_part = (scalar_t *)sendbuf + tid * part_conut;
        for (i = 0; i < part_conut; i += SVEParams<scalar_t>::vector_elements) {
            svbool_t pred = SVEParams<scalar_t>::create_predicate(i, part_conut);
            auto sve = svld1(pred, sendbuf_part + i);
            for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                svst1(pred, (scalar_t *)buffers[peer] + ((rank * sendcount) + tid * part_conut) + i, sve);
            }
        }
        kurmcl_recv_imm_cnt(1, peerleader_rank + tid, ds_conn_info);
        scalar_t *readbuf_part =
            (scalar_t *)recvbuf + ((peerleader_rank + tid) * 8 + group_rank - myleader_rank) * part_conut;
        for (i = 0; i < part_conut; i += SVEParams<scalar_t>::vector_elements) {
            svbool_t pred = SVEParams<scalar_t>::create_predicate(i, part_conut);
            auto sve = svld1(pred, readbuf_part + i);
            for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                if (peer != rank) {
                    svst1(pred,
                        (scalar_t *)buffers[peer] + ((peerleader_rank + tid) * 8 + group_rank - myleader_rank) *
                            part_conut + i,
                        sve);
                }
            }
        }
    });
    barrier();
}

template <typename scalar_t>
void shm_allgather_comm8(const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf, int64_t recvcount, int64_t rank,
                         uint8_t **buffers, const std::function<void()> &barrier)
{
    KUTACC_CHECK(sendcount <= max_buf_size,
        "input sendcount must less than max_buf_size you applied in shm_allgather_standard_init");
    const int64_t thread_nums = kutacc::get_thread_num();
    int64_t part_length = sendcount / thread_nums;
    int64_t my_dieleader = rank < 4 ? 0 : 4;
    int64_t read_from_rank[8] = {4, 5, 6, 7, 0, 1, 2, 3};
    barrier();
    kutacc::parallel_for(0, thread_nums, 1, [&](int64_t start, int64_t end) {
        for (int tid = start; tid < end; ++tid) {
            int i = 0;
            scalar_t *sendbuf_part = (scalar_t *)sendbuf + tid * part_length;
            scalar_t *read_from =
                (scalar_t *)buffers[read_from_rank[rank]] + ((read_from_rank[rank] * sendcount) + tid * part_length);
            for (i = 0; i < part_length; i += SVEParams<scalar_t>::vector_elements) {
                svbool_t pred = SVEParams<scalar_t>::create_predicate(i, part_length);
                auto sve1 = svld1(pred, sendbuf_part + i);
                auto sve2 = svld1(pred, read_from + i);
                for (int peer = my_dieleader; peer < my_dieleader + 4; ++peer) {
                    if (peer != rank) {
                        svst1(pred, (scalar_t *)buffers[peer] + ((rank * sendcount) + tid * part_length) + i, sve1);
                    }
                    svst1(pred,
                        (scalar_t *)buffers[peer] + ((read_from_rank[rank] * sendcount) + tid * part_length) + i,
                        sve2);
                }
            }
        }
    });
    barrier();
}

template <typename scalar_t, int world_size>
void shm_allgather(const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                   int64_t recvcount, int64_t rank, uint8_t **buffers,
                   kurmcl_conn_info_h ds_conn_info, const std::function<void()> &barrier)
{
    if constexpr(world_size == 8) {
        if (rank < 8) {
            shm_allgather_comm8(sendbuf, sendcount, recvbuf, recvcount, rank, buffers, barrier);
        }
    } else if constexpr(world_size == 16) {
        shm_allgather_comm16(sendbuf, sendcount, recvbuf, recvcount, rank, buffers, ds_conn_info, barrier);
    } else {
        KUTACC_CHECK(false, "no implement");
    }
}

template void shm_batch_allgather_base<uint8_t>(int64_t, const uint8_t *, int64_t, uint8_t *, int64_t,
    uint8_t **, int64_t, const shm_allgather_request_h &, bool);
template void shm_batch_allgather_base<bfloat16_t>(int64_t, const bfloat16_t *, int64_t, bfloat16_t *, int64_t,
    uint8_t **, int64_t, const shm_allgather_request_h &, bool);
template void shm_batch_allgather_optimize<uint8_t>(int64_t, uint8_t *, int64_t, uint8_t *, int64_t,
    uint8_t **, uint8_t **, const shm_allgather_request_h &);
template void shm_batch_allgather_optimize<bfloat16_t>(int64_t, bfloat16_t *, int64_t, bfloat16_t *, int64_t,
    uint8_t **, uint8_t **, const shm_allgather_request_h &);

template void shm_allgather<uint8_t, 16>(const uint8_t *, int64_t, uint8_t *, int64_t, int64_t, uint8_t **,
                                         kurmcl_conn_info_h ds_conn_info, const std::function<void()> &);
template void shm_allgather<bfloat16_t, 16>(const bfloat16_t *, int64_t, bfloat16_t *, int64_t, int64_t, uint8_t **,
                                            kurmcl_conn_info_h ds_conn_info, const std::function<void()> &);
template void shm_allgather<bfloat16_t, 8>(const bfloat16_t *, int64_t, bfloat16_t *, int64_t, int64_t, uint8_t **,
                                           kurmcl_conn_info_h ds_conn_info, const std::function<void()> &);

void two_phase_allgather(
    int rank, int tid, const uint8_t* src, int64_t src_size, uint8_t **remote_buffers, int node_pair_offset)
{
    int i = 0;
    const int thread_nums = 8;
    KUTACC_CHECK(src_size > 0, "src_size must be greater than 0");
    KUTACC_CHECK(tid >= 0 && tid < thread_nums, "tid out of range");
    int64_t myleader_rank = rank < node_pair_offset ? 0 : node_pair_offset;
    int64_t peerleader_rank = rank < node_pair_offset ? node_pair_offset : 0;
    int base = src_size / thread_nums;
    int rem = src_size % thread_nums;
    int target_buff_len = tid < rem ? (base + 1) : base;
    int target_buff_off = tid < rem ? tid * target_buff_len : rem * (base + 1) + (tid - rem) * base;
    uint8_t *sendbuf_part = (uint8_t *)src + target_buff_off;
    for (i = 0; i < target_buff_len; i += 64) {
        svbool_t pred = svwhilelt_b8(i, target_buff_len);
        auto sve = svld1(pred, sendbuf_part + i);
        for (int peer = myleader_rank; peer < myleader_rank + node_pair_offset; ++peer) {
            if (peer != rank) {
                svst1(pred, (uint8_t *)remote_buffers[peer] + rank * src_size + target_buff_off + i, sve);
            }
        }
    }
    uint8_t *readbuf_part = (uint8_t *)remote_buffers[peerleader_rank + rank - myleader_rank] +
        (peerleader_rank + rank - myleader_rank) * src_size + target_buff_off;
    for (i = 0; i < target_buff_len; i += 64) {
        svbool_t pred = svwhilelt_b8(i, target_buff_len);
        auto sve = svld1(pred, readbuf_part + i);
        for (int peer = myleader_rank; peer < myleader_rank + node_pair_offset; ++peer) {
            svst1(pred, (uint8_t *)remote_buffers[peer] +
                            (peerleader_rank + rank - myleader_rank) * src_size + target_buff_off + i, sve);
        }
    }
}

void shm_dual_allgather(void* src0, int64_t sc0, void* dst0, int64_t dc0, void* src1,
    int64_t sc1, void* dst1, int64_t dc1, void** remote_ptr_buffers0, void **remote_ptr_buffers1,
    shm_datatype_t datatype, const std::function<void()> &barrier, const shm_allgather_request_h &request)
{
    barrier();
    KUTACC_CHECK(datatype == SHM_DATATYPE_UINT8, "only support uint8");
    int rank = request->rank;
    int comm_size = request->comm_size;
    KUTACC_CHECK(comm_size == 16 || comm_size == 8, "comm_size should be 8 or 16");
    int node_pair_offset = comm_size == 16 ? 8 : 4;
    kutacc::parallel_for(0, 16, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        if (tid < 8) {
            // topk_weight allgather
            two_phase_allgather(rank, tid, (uint8_t*)src0, sc0, (uint8_t**)remote_ptr_buffers0, node_pair_offset);
        } else {
            // topk_ids allgather
            two_phase_allgather(rank, tid - 8, (uint8_t*)src1, sc1, (uint8_t**)remote_ptr_buffers1, node_pair_offset);
        }
    });
    barrier();
}

}  // namespace kutacc
