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

const int test_times = 500;
MPI_Comm node_comm, socket_comm, die_comm;

kupl_shm_comm_h kupl_node_comm, kupl_socket_comm, kupl_die_comm;
kupl_shm_win_h kupl_node_win, kupl_node_send_win, kupl_socket_win, kupl_die_win;
kutacc::kurmcl_conn_info_h usr_conn_info;

int global_rank, global_size;

uint8_t *buffers[16];
uint8_t *remote_send[16];
uint8_t *local_buffer;
uint8_t *local_send_buffer;

constexpr size_t MAX_BUFFER_SIZE = 1 << 28;

void shm_init()
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
    kupl_shm_win_alloc(MAX_BUFFER_SIZE, kupl_node_comm, (void **)&local_send_buffer, &kupl_node_send_win);
    MPI_Comm world_comm = MPI_COMM_WORLD;
    kutacc::kurmcl_comm_create(global_size, global_rank, kurmcl_oob_cbs_h, (void *)(&world_comm), &usr_conn_info);
    kupl_shm_fence(kupl_node_win);

    for (int i = 0; i < 16; ++i) {
        kupl_shm_win_query(kupl_node_win, i, (void **)&buffers[i]);
        kupl_shm_win_query(kupl_node_send_win, i, (void **)&remote_send[i]);
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

template <typename scalar_t, int world_size, bool is_hierarchical, bool is_same_buffer, bool is_batch_concat>
void allgather_test(int batch, int sendcount, kutacc::shm_datatype_t datatype)
{
    int rank = global_rank % 16;
    int recvcount = sendcount * world_size;
    int buffer_size = 0;
    double sum_time = 0;
    scalar_t *sendbuf = (scalar_t *)local_buffer;
    scalar_t *recvbuf = nullptr;
    kutacc::shm_allgather_request_h request;
    kutacc::shm_allgather_request_create(rank, world_size, request);
    kutacc::shm_allgather_request_init(kupl_die_win, kupl_socket_win, kupl_node_win, request);
    if (is_batch_concat) {
        if (!is_same_buffer) {
            sendbuf = (scalar_t *)malloc(batch * sendcount * sizeof(scalar_t));
        }
        buffer_size = MAX_BUFFER_SIZE - batch * recvcount * sizeof(scalar_t);
        recvbuf = (scalar_t *)(local_buffer + buffer_size);
        for (int i = 0; i < batch * sendcount; ++i) {
            sendbuf[i] = gen(i, rank);
        }
    }
    kupl_shm_fence(kupl_node_win);
    if (is_batch_concat) {
        for (int t = 0; t < test_times; ++t) {
            if (!is_same_buffer) {
                sendbuf = (scalar_t *)malloc(batch * sendcount * sizeof(scalar_t));
            }
            buffer_size = MAX_BUFFER_SIZE - batch * recvcount * sizeof(scalar_t);
            recvbuf = (scalar_t *)(local_buffer + buffer_size);
            for (int i = 0; i < batch * sendcount; ++i) {
                sendbuf[i] = rank;
            }
            kutacc::shm_batch_allgather(batch, sendbuf, sendcount, recvbuf, recvcount, datatype, nullptr,
                                                  buffers, buffer_size, request, is_hierarchical, true);
        }
        for (int t = 0; t < test_times; ++t) {
            if (!is_same_buffer) {
                sendbuf = (scalar_t *)malloc(batch * sendcount * sizeof(scalar_t));
            }
            buffer_size = MAX_BUFFER_SIZE - batch * recvcount * sizeof(scalar_t);
            recvbuf = (scalar_t *)(local_buffer + buffer_size);
            for (int i = 0; i < batch * sendcount; ++i) {
                sendbuf[i] = gen(i, rank);
            }
            uint64_t start = kutacc_now_ns();
            kutacc::shm_batch_allgather(batch, sendbuf, sendcount, recvbuf, recvcount, datatype, nullptr,
                                                  buffers, buffer_size, request, is_hierarchical, true);
            uint64_t end = kutacc_now_ns();
            sum_time += end - start;
        }
    }
    kupl_shm_fence(kupl_node_win);
    if (is_batch_concat) {
        if (rank < world_size) {
            for (int i = 0; i < batch * recvcount; ++i) {
                int b = i / recvcount;
                int r = rank / world_size * world_size + i % recvcount / sendcount;
                FLASH_ASSERT(recvbuf[i] == (scalar_t)gen(b * sendcount + i % sendcount, r));
            }
        }
    }
    if (rank == 0) {
        if (!is_batch_concat) {
            printf("Package_size = %lu, comm_size = %d, avg_time = %.2lf us\n", sendcount * sizeof(scalar_t),
                world_size, sum_time / test_times / 1000);
        } else {
            printf("Package_size = %lu(batch = %d), comm_size = %d, avg_time = %.2lf us\n",
                batch * sendcount * sizeof(scalar_t), batch, world_size, sum_time / test_times / 1000);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!is_same_buffer) {
        free(sendbuf);
    }
}

int main(int argc, char *argv[])
{
    kutacc::global_parallel_launch([&] {
        MPI_Init(&argc, &argv);
        shm_init();
        kutacc_time_init();
        // cases in DeepSeek inference
        allgather_test<uint8_t, 16, true, false, true>(128, 8, kutacc::SHM_DATATYPE_UINT8);
        allgather_test<bfloat16_t, 16, false, true, true>(128, 16, kutacc::SHM_DATATYPE_BFLOAT16);
        allgather_test<bfloat16_t, 8, false, true, true>(128, 264, kutacc::SHM_DATATYPE_BFLOAT16);
        allgather_test<uint8_t, 8, true, true, true>(64, 528, kutacc::SHM_DATATYPE_UINT8);
        // allgather_test<bfloat16_t, 16, false, true, false>(0, 7168 * 8, kutacc::SHM_DATATYPE_BFLOAT16);
        // allgather_test<bfloat16_t, 8, false, true, false>(0, 7168 * 8, kutacc::SHM_DATATYPE_BFLOAT16);
        shm_finalize();
        MPI_Finalize();
    });
    return 0;
}