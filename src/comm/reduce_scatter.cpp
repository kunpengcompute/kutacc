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

#include <atomic>
#include <arm_sve.h>
#include <cstring>
#include <functional>
#include <sys/time.h>
#include <kupl.h>

#include "kutacc.h"
#include "utils/check.h"
#include "utils/timer.h"

namespace kutacc {

namespace {
    int fence_num = 4;
}

typedef struct shm_reduce_scatter_request {
    int rank;
    int comm_size;
    size_t fence_buffer_size;
    int16_t **fence_buffers;
    kupl_shm_win_h shm_win_intra_die;
    kupl_shm_win_h shm_win_intra_socket;
    kupl_shm_win_h shm_win_intra_node;
} *shm_reduce_scatter_request_h;

void shm_reduce_scatter_request_create(int rank, int comm_size, shm_datatype_t datatype, size_t &fence_buffer_size,
    shm_reduce_scatter_request_h &request)
{
    KUTACC_CHECK(datatype == SHM_DATATYPE_BFLOAT16, "only support bf16");
    int num_threads = kutacc::get_thread_num();
    KUTACC_CHECK(comm_size == 16 || comm_size == 8, "comm_size should be 8 or 16");
    request = (shm_reduce_scatter_request *)malloc(sizeof(shm_reduce_scatter_request));
    request->rank = rank;
    request->comm_size = comm_size;
    request->fence_buffer_size = num_threads * fence_num * align;
    request->fence_buffers = (int16_t **)malloc(sizeof(int16_t *) * comm_size);
    fence_buffer_size = request->fence_buffer_size;
}

void shm_reduce_scatter_request_init(void **fence_buffers_, kupl_shm_win_h shm_win_intra_die,
    kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node, const shm_reduce_scatter_request_h &request)
{
    memcpy(request->fence_buffers, fence_buffers_, sizeof(int16_t *) * request->comm_size);
    memset(fence_buffers_[request->rank], 0, request->fence_buffer_size);
    request->shm_win_intra_die = shm_win_intra_die;
    request->shm_win_intra_socket = shm_win_intra_socket;
    request->shm_win_intra_node = shm_win_intra_node;
    kupl_shm_fence(shm_win_intra_node);
}

void shm_reduce_scatter_request_destroy(const shm_reduce_scatter_request_h &request)
{
    free(request->fence_buffers);
    free(request);
}

void shm_reduce_scatter_comm8_impl1(int64_t height, int64_t width, bfloat16_t** remote_ptr_buffers,
    const shm_reduce_scatter_request_h &request)
{
    int rank = request -> rank;
    int comm_size = request -> comm_size;
    kupl_shm_fence(request->shm_win_intra_socket); //
    int64_t chunk_size = height / comm_size * width;
    int peer_rank = rank ^ 4;
    int die_leader = rank / 4 * 4;
    int num_threads = kutacc::get_thread_num();
    int thread_chunk_size = chunk_size / num_threads;
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int data_bias = chunk_size * peer_rank + thread_chunk_size * tid;
        int fence_offset = fence_num * tid;
        bfloat16_t *data_buffers[5];
        for (int i = 0; i < 4; ++i) {
            data_buffers[i] = remote_ptr_buffers[die_leader + i] + data_bias;
        }
        reduce<4, 1>(data_buffers, (bfloat16_t *[]){data_buffers[rank % 4]}, thread_chunk_size);
        signal_fence(request->fence_buffers[peer_rank], fence_offset + rank / 4);
        wait_fence(request->fence_buffers[rank], fence_offset + peer_rank / 4);
        data_bias = chunk_size * rank + thread_chunk_size * tid;
        for (int i = 0; i < 4; ++i) {
            data_buffers[i] = remote_ptr_buffers[die_leader + i] + data_bias;
        }
        data_buffers[4] = remote_ptr_buffers[peer_rank] + data_bias;
        reduce<5, 1>(data_buffers, (bfloat16_t *[]){data_buffers[rank % 4]}, thread_chunk_size);
    });
}

void shm_reduce_scatter_comm8_impl2(int64_t height, int64_t width, bfloat16_t** remote_ptr_buffers,
    const shm_reduce_scatter_request_h &request)
{
    int rank = request -> rank;
    int comm_size = request->comm_size;
    
    kupl_shm_fence(request->shm_win_intra_socket); //

    int64_t chunk_size = height / comm_size * width;
    int peer_rank = rank ^ 4;
    int die_leader = rank / 4 * 4;
    int num_threads = kutacc::get_thread_num();
    int chunk_num_threads = num_threads / 2;
    int thread_chunk_size = chunk_size / chunk_num_threads;
    KUTACC_CHECK(thread_chunk_size > 0, "thread_chunk_size should be greater than 0");
    int peer_ranks[2] = {rank, rank ^ 4};
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int local_tid = tid % chunk_num_threads;
        int bid = tid / chunk_num_threads;
        int target_rank = peer_ranks[bid];
        int data_bias = chunk_size * target_rank + thread_chunk_size * local_tid;
        int fence_offset = fence_num * (target_rank/4 * chunk_num_threads + local_tid);
        bfloat16_t *data_buffers[5];
        for (int i = 0; i < 4; ++i) {
            data_buffers[i] = remote_ptr_buffers[die_leader + i] + data_bias;
        }
        reduce<4, 1>(data_buffers, (bfloat16_t *[]){data_buffers[rank%4]}, thread_chunk_size);
        if (target_rank == rank) {
            wait_fence(request->fence_buffers[rank], fence_offset + peer_rank / 4);
            data_buffers[0] = remote_ptr_buffers[rank] + data_bias;
            data_buffers[1] = remote_ptr_buffers[peer_rank] + data_bias;
            reduce<2, 1>(data_buffers, (bfloat16_t *[]){data_buffers[0]}, thread_chunk_size);
        } else {
            signal_fence(request->fence_buffers[target_rank], fence_offset + rank / 4);
        }
    });
}

void shm_reduce_scatter_comm16_impl1(int64_t height, int64_t width, bfloat16_t** remote_ptr_buffers,
    const shm_reduce_scatter_request_h &request)
{
    int rank = request -> rank;
    bfloat16_t* rank_buffer[COMM_MAX_SIZE];
    int comm_size = request -> comm_size;
    kupl_shm_fence(request->shm_win_intra_node); //
    int64_t chunk_size = height / comm_size  * width;
    for (int64_t i = 0; i < comm_size; ++i) {
        rank_buffer[i] = (bfloat16_t*)remote_ptr_buffers[i] + chunk_size * rank;
    }
    bfloat16_t* local_buffer = rank_buffer[rank];
    reduce<16, 1>(rank_buffer, (bfloat16_t *[]){local_buffer}, chunk_size);
}

void shm_reduce_scatter_comm16_impl2(int64_t height, int64_t width, bfloat16_t** remote_ptr_buffers,
    const shm_reduce_scatter_request_h &request)
{
    int rank = request -> rank;
    int comm_size = request -> comm_size;
    int num_threads = kutacc::get_thread_num();
    int chunk_num_threads = num_threads / 4;
    int64_t chunk_size = height / comm_size * width;
    int thread_chunk_size = chunk_size / chunk_num_threads;
    int leader_rank = rank / 4 * 4;
    int peer_ranks[4];
    for (int i = 0; i < 4; ++i) {
        peer_ranks[i] = i * 4 + rank % 4;
    }
    bfloat16_t* local_buffer = remote_ptr_buffers[rank] + chunk_size * width * rank;

    kupl_shm_fence(request->shm_win_intra_node);

    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int local_tid = tid % chunk_num_threads;
        int target_rank = peer_ranks[tid / chunk_num_threads];
        int data_bias = chunk_size * target_rank + thread_chunk_size * local_tid;
        int fence_offset = fence_num * tid + rank / 4;
        bfloat16_t *data_buffers[4];
        for (int i = 0; i < 4; ++i) {
            data_buffers[i] = remote_ptr_buffers[leader_rank + i] + data_bias;
        }
        reduce<4, 1>(data_buffers, (bfloat16_t *[]){remote_ptr_buffers[rank] + data_bias}, thread_chunk_size);
        signal_fence(request->fence_buffers[target_rank], fence_offset);
        if (target_rank == rank) {
            wait_batch_fence(request->fence_buffers[rank], tid * fence_num);
            for (int i = 0; i < 4; ++i) {
                data_buffers[i] = remote_ptr_buffers[peer_ranks[i]] + data_bias;
            }
            reduce<4, 1>(data_buffers, (bfloat16_t *[]){remote_ptr_buffers[rank] + data_bias}, thread_chunk_size);
        }
    });
}

void shm_reduce_scatter_comm16_impl3(
    int64_t height, int64_t width, bfloat16_t** remote_ptr_buffers, const shm_reduce_scatter_request_h &request)
{
    /* 二层reduce改为三层reduce */
    int rank = request -> rank;
    int comm_size = request -> comm_size;
    int num_threads = kutacc::get_thread_num();
    int chunk_num_threads = num_threads / 4;
    int64_t chunk_size = height / comm_size * width;
    int thread_chunk_size = chunk_size / chunk_num_threads;
    KUTACC_CHECK(thread_chunk_size > 0, "thread_chunk_size should be greater than 0");
    int leader_rank = rank / 4 * 4;
    int peer_ranks[4];
    for (int i = 0; i < 4; ++i) {
        peer_ranks[i] = i * 4 + rank % 4;
    }
    bfloat16_t* local_buffer = remote_ptr_buffers[rank] + chunk_size * width * rank;

    kupl_shm_fence(request->shm_win_intra_node);

    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = start;
        int local_tid = tid % chunk_num_threads;
        int target_rank = peer_ranks[tid / chunk_num_threads];
        int data_bias = chunk_size * target_rank + thread_chunk_size * local_tid;
        int fence_offset = fence_num * tid;
        bfloat16_t *data_buffers[4];
        for (int i = 0; i < 4; ++i) {
            data_buffers[i] = remote_ptr_buffers[leader_rank + i] + data_bias;
        }
        reduce<4, 1>(data_buffers, (bfloat16_t *[]){remote_ptr_buffers[rank] + data_bias}, thread_chunk_size);
        if (target_rank == rank || target_rank == (rank^8)) {
            wait_fence(request->fence_buffers[rank], fence_offset + (rank ^ 4) / 4);
            data_buffers[0] = remote_ptr_buffers[rank] + data_bias;
            data_buffers[1] = remote_ptr_buffers[rank^4] + data_bias;
            reduce<2, 1>(data_buffers, (bfloat16_t *[]){remote_ptr_buffers[rank] + data_bias}, thread_chunk_size);
            if (target_rank == (rank^8)) {
                signal_fence(request->fence_buffers[rank^8], fence_offset + rank / 4);
            }
        } else {
            signal_fence(request->fence_buffers[rank^4], fence_offset + rank / 4);
        }
        if (target_rank == rank) {
            wait_fence(request->fence_buffers[rank], fence_offset + (rank ^ 8) / 4);
            data_buffers[0] = remote_ptr_buffers[rank] + data_bias;
            data_buffers[1] = remote_ptr_buffers[rank^8] + data_bias;
            reduce<2, 1>(data_buffers, (bfloat16_t *[]){remote_ptr_buffers[rank] + data_bias}, thread_chunk_size);
        }
    });
}

void shm_reduce_scatter_comm16_baseline(int64_t height, int64_t width, bfloat16_t** remote_ptr_buffers,
    const shm_reduce_scatter_request_h &request)
{
    int rank = request -> rank;
    bfloat16_t* rank_buffer[COMM_MAX_SIZE];
    int comm_size = request -> comm_size;
    kupl_shm_fence(request->shm_win_intra_node); //
    int64_t chunk_size = height / comm_size;
    for (int64_t i = 0; i < comm_size; ++i) {
        rank_buffer[i] = (bfloat16_t*)remote_ptr_buffers[i] + chunk_size * width * rank;
    }
    kutacc::parallel_for(0, chunk_size * width, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i += svcnth()) {
            svbool_t pg16 = svwhilelt_b16(i, end);
            svfloat32_t s0 = svdup_f32(0);
            svfloat32_t s1 = svdup_f32(0);
            svbfloat16_t zero_bf16 = svdup_bf16(0);
            for (int64_t j = 0; j < comm_size; ++j) {
                int64_t k = (rank + j) % comm_size;
                svbfloat16_t v = svld1(pg16, rank_buffer[k] + i);
                svfloat32_t v0 = svreinterpret_f32(svzip1(zero_bf16, v));
                svfloat32_t v1 = svreinterpret_f32(svzip2(zero_bf16, v));
                s0 = svadd_x(svptrue_b32(), s0, v0);
                s1 = svadd_x(svptrue_b32(), s1, v1);
            }
            svst1(pg16, rank_buffer[rank] + i,
                svuzp1(svcvt_bf16_x(svptrue_b16(), s0), svcvt_bf16_x(svptrue_b16(), s1)));
        }
    });
    kupl_shm_fence(request->shm_win_intra_node); //
}

void shm_reduce_scatter(void** remote_ptr_buffers, int64_t height, int64_t width,
    const shm_reduce_scatter_request_h &request)
{
    int comm_size = request -> comm_size;
    KUTACC_CHECK(height % comm_size == 0, "bs % dtp != 0, bs = ", height, ", dtp = ", comm_size);
    if (comm_size == 8) {
        shm_reduce_scatter_comm8_impl2(height, width, (bfloat16_t **)remote_ptr_buffers, request);
    } else if (comm_size == 16) {
        shm_reduce_scatter_comm16_impl3(height, width, (bfloat16_t **)remote_ptr_buffers, request);
    } else {
        KUTACC_CHECK(false, "comm_size should be 8 or 16");
    }
}

} // namespace kutacc