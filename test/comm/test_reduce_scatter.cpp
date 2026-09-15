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
#include <iostream>
#include <cstring>
#include <atomic>
#include <mpi.h>
#include <kupl.h>
#include <arm_sve.h>
#include <sys/time.h>
#include <arm_neon.h>

#include "kutacc.h"
#include "helper.h"

#define FLASH_ASSERT(cond)                                                                                      \
    do {                                                                                                        \
        if (not (cond)) {                                                                                       \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                       \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)

template <typename T>
static int convert(T datatype)
{
    if constexpr(std::is_same<T, kupl_shm_datatype_t>::value) {
        switch (datatype) {
            case KUPL_SHM_DATATYPE_CHAR:
                return MPI_CHAR;
            case KUPL_SHM_DATATYPE_INT:
                return MPI_INT;
            case KUPL_SHM_DATATYPE_LONG:
                return MPI_LONG;
            case KUPL_SHM_DATATYPE_FLOAT:
                return MPI_FLOAT;
            case KUPL_SHM_DATATYPE_DOUBLE:
                return MPI_DOUBLE;
            default:
                return KUPL_ERROR;
        }
    } else if constexpr(std::is_same<T, kutacc::kurmcl_datatype_t>::value) {
        switch (datatype) {
            case kutacc::KURMCL_DATATYPE_CHAR:
                return MPI_CHAR;
            case kutacc::KURMCL_DATATYPE_INT:
                return MPI_INT;
            case kutacc::KURMCL_DATATYPE_LONG:
                return MPI_LONG;
            case kutacc::KURMCL_DATATYPE_FLOAT:
                return MPI_FLOAT;
            case kutacc::KURMCL_DATATYPE_DOUBLE:
                return MPI_DOUBLE;
            default:
                return KUPL_ERROR;
        }
    }
}

template <typename T>
static int oob_allgather_callback(
    const void *sendbuf, void *recvbuf, int size, void *group, T datatype)
{
    int *group_ = (int *)group;
    if (sendbuf == nullptr) {
        sendbuf = (void *)(-1);
    }
    auto mpi_datatype = convert(datatype);
    if (mpi_datatype == KUPL_ERROR) {
        printf("not support datatype");
        return KUPL_ERROR;
    }
    return MPI_Allgather(sendbuf, size, mpi_datatype, recvbuf, size, mpi_datatype, (MPI_Comm)(*group_));
}

static int oob_barrier_callback(void *group)
{
    int *group_ = (int *)group;
    return MPI_Barrier((MPI_Comm)(*group_));
}

template <typename T>
static int oob_alltoall_callback(const void* sendbuf, int sendcount, T send_datatype, void* recvbuf,
    int recvcount, T recv_datatype, void *group)
{
    int *group_ = (int *)group;
    if (sendbuf == nullptr) {
        sendbuf = (void *)(-1);
    }
    auto mpi_send_datatype = convert(send_datatype);
    auto mpi_recv_datatype = convert(recv_datatype);
    if (mpi_send_datatype == KUPL_ERROR || mpi_recv_datatype == KUPL_ERROR) {
        printf("not support datatype");
        return KUPL_ERROR;
    }
    return MPI_Alltoall(
        sendbuf, sendcount, mpi_send_datatype, recvbuf, recvcount, mpi_recv_datatype, (MPI_Comm)(*group_));
}

const int warm_times = 100;
const int test_times = 1000;
MPI_Comm node_comm;
MPI_Comm socket_comm;
MPI_Comm die_comm;

kupl_shm_comm_h kupl_node_comm;
kupl_shm_comm_h kupl_socket_comm;
kupl_shm_comm_h kupl_die_comm;
kupl_shm_win_h kupl_node_win;
kupl_shm_win_h kupl_node_fence_win;
kupl_shm_win_h kupl_socket_win;
kupl_shm_win_h kupl_die_win;
kutacc::kurmcl_conn_info_h usr_conn_info;

int global_rank;
int global_size;
int comm_size;
kutacc::shm_reduce_scatter_request_h request;

bfloat16_t *buffers[16];
int16_t *fence_buffers[16];
bfloat16_t *local_buffer;
int16_t *local_fence_buffer;

constexpr size_t MAX_BUFFER_SIZE = 1 << 28;

void reduce_scatter_test_init(int comm_size_)
{
    int rank = global_rank % comm_size_;
    comm_size = comm_size_;
    size_t fence_buffer_size;
    kutacc::shm_reduce_scatter_request_create(rank, comm_size, kutacc::SHM_DATATYPE_BFLOAT16, fence_buffer_size, request);
    kutacc::shm_reduce_scatter_request_init((void **)fence_buffers, kupl_die_win, kupl_socket_win, kupl_node_win, request);
}

void reduce_scatter_test_finalize()
{
    kutacc::shm_reduce_scatter_request_destroy(request);
}

void shm_init(int comm_size)
{
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    FLASH_ASSERT(global_size % 16 == 0);

    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;
    kutacc::kurmcl_oob_cb_t kurmcl_oob_cbs;
    kutacc::kurmcl_oob_cb_h kurmcl_oob_cbs_h = &kurmcl_oob_cbs;
    kurmcl_oob_cbs_h->oob_allgather = oob_allgather_callback;
    kurmcl_oob_cbs_h->oob_barrier = oob_barrier_callback;
    kurmcl_oob_cbs_h->oob_alltoall = oob_alltoall_callback;

    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 4, global_rank % 4, &die_comm);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 8, global_rank % 8, &socket_comm);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 16, global_rank % 16, &node_comm);

    kupl_shm_comm_create(4, global_rank % 4, global_rank, oob_cbs_h, (void *)(&die_comm), &kupl_die_comm);
    kupl_shm_comm_create(8, global_rank % 8, global_rank, oob_cbs_h, (void *)(&socket_comm), &kupl_socket_comm);
    kupl_shm_comm_create(16, global_rank % 16, global_rank, oob_cbs_h, (void *)(&node_comm), &kupl_node_comm);

    kupl_shm_win_alloc(1, kupl_die_comm, (void **)&local_buffer, &kupl_die_win);
    kupl_shm_win_alloc(1, kupl_socket_comm, (void **)&local_buffer, &kupl_socket_win);
    kupl_shm_win_alloc(MAX_BUFFER_SIZE, kupl_node_comm, (void **)&local_buffer, &kupl_node_win);
    kupl_shm_win_alloc(MAX_BUFFER_SIZE, kupl_node_comm, (void **)&local_fence_buffer, &kupl_node_fence_win);

    MPI_Comm world_comm = MPI_COMM_WORLD;
    kutacc::kurmcl_comm_create(global_size, global_rank, kurmcl_oob_cbs_h, (void *)(&world_comm), &usr_conn_info);

    kupl_shm_fence(kupl_node_win);

    for (int i = 0; i < comm_size; ++i) {
        kupl_shm_win_query(kupl_node_win, i, (void **)&buffers[i]);
        kupl_shm_win_query(kupl_node_fence_win, i, (void **)&fence_buffers[i]);
    }
    if (comm_size == 8 && global_rank % 16 >= 8) {
        for (int i = 0; i < comm_size; ++i) {
            kupl_shm_win_query(kupl_node_win, i + 8, (void **)&buffers[i]);
            kupl_shm_win_query(kupl_node_fence_win, i + 8, (void **)&fence_buffers[i]);
        }
    }
}

void shm_finalize()
{
    kupl_shm_win_free(kupl_node_win);
    kupl_shm_win_free(kupl_socket_win);
    kupl_shm_win_free(kupl_die_win);
    kupl_shm_comm_destroy(kupl_node_comm);
    kupl_shm_comm_destroy(kupl_socket_comm);
    kupl_shm_comm_destroy(kupl_die_comm);
}

int gen(int i, int rank)
{
    return (i ^ rank) % 17;
}

bool check_reduce_scatter_result(int rank, int height, int width, int comm_size, const bfloat16_t* local_buffer)
{
    const int total_elems = height * width;
    const int chunk_size = total_elems / comm_size;
    const int start_idx = rank * chunk_size;
    const int end_idx = (rank + 1) * chunk_size;
    bool is_all_correct = true;

    for (int i = start_idx; i < end_idx; ++i) {
        bfloat16_t expected_val = 0.0f;
        bfloat16_t base = i / chunk_size;
        if (comm_size == 8) {
            expected_val = base * comm_size;
        } else if (comm_size == 16) {
            expected_val = base * comm_size;
        } else {
            printf("comm_size error...\n");
        }
        const bfloat16_t actual_val = local_buffer[i];

        if (fabs(to_float(actual_val) - to_float(expected_val)) > 1e-3) {
            fprintf(stderr, "进程%d校验失败 | 元素索引%d:\n", rank, i);
            fprintf(stderr, "  预期值：%lf | 实际值：%lf\n", to_float(expected_val), to_float(actual_val));
            is_all_correct = false;
        }
    }

    if (is_all_correct) {
        printf("进程%d:分片正确性通过\n", rank);
    }
    return is_all_correct;
}

void reduce_scatter_test(int comm_size, int64_t height, int64_t width)
{
    int64_t rank = global_rank % comm_size;
    auto barrier = [&]() {
        if (comm_size <= 8) {
            kupl_shm_fence(kupl_socket_win);
        } else {
            kupl_shm_fence(kupl_node_win);
        }
    };
    int total_elems = height * width;
    double sum_time = 0;
    // 初始化：rank + 元素索引
    for (int i = 0; i < total_elems; ++i) {
        local_buffer[i] = i / (height/comm_size*width) ;
    }
    // 正确性检查
    barrier();
    kutacc::shm_reduce_scatter((void **)buffers, height, width, request);
    check_reduce_scatter_result(rank, height, width, comm_size, local_buffer);
    barrier();
    // 性能测试
    for (int t = 0; t < warm_times; ++t) {
        for (int i = 0; i < total_elems; ++i) {
            local_buffer[i] = static_cast<bfloat16_t>(rank);
        }
        kutacc::shm_reduce_scatter((void **)buffers, height, width, request);
    }
    for (int t = 0; t < test_times; ++t) {
        for (int i = 0; i < total_elems; ++i) {
            local_buffer[i] = static_cast<bfloat16_t>(rank);
        }
        barrier();
        uint64_t start = kutacc_now_ns();
        kutacc::shm_reduce_scatter((void **)buffers, height, width, request);
        uint64_t end = kutacc_now_ns();
        sum_time += end - start;
    }
    barrier();
    printf("rank = %ld, M = %ld, width = %ld, comm_size = %d, avg_time = %.2lf us\n", rank, height, width, comm_size,
        sum_time / test_times / 1000);
}

int main(int argc, char *argv[])
{
    kutacc::global_parallel_launch([&] {
        MPI_Init(&argc, &argv);
        int comm_size = 16;
        int M = 128;
        int width = 7168;
        shm_init(comm_size);
        kutacc_time_init();
        reduce_scatter_test_init(comm_size);

        // para: comm_size, M, width
        // reduce_scatter_test(8, 128, 1);
        // reduce_scatter_test(8, 64, 7168);
        // reduce_scatter_test(8, 128, 7168);
        reduce_scatter_test(comm_size, M, width);
        reduce_scatter_test_finalize();
        shm_finalize();
        MPI_Finalize();
    });
    return 0;
}