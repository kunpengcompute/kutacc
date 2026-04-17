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

#define FLASH_ASSERT(cond)                                                                                      \
    do {                                                                                                        \
        if (not (cond)) {                                                                                       \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                       \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)

static int oob_allgather_callback(
    const void *sendbuf, void *recvbuf, int size, void *group, kupl_shm_datatype_t datatype)
{
	int *group_ = (int *)group;
	if (sendbuf == nullptr) {
		sendbuf = (void *)(-1);
	}
    switch (datatype) {
        case KUPL_SHM_DATATYPE_CHAR:
            return MPI_Allgather(sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR, (MPI_Comm)(*group_));
        case KUPL_SHM_DATATYPE_INT:
            return MPI_Allgather(sendbuf, size, MPI_INT, recvbuf, size, MPI_INT, (MPI_Comm)(*group_));
        case KUPL_SHM_DATATYPE_LONG:
            return MPI_Allgather(sendbuf, size, MPI_LONG, recvbuf, size, MPI_LONG, (MPI_Comm)(*group_));
        case KUPL_SHM_DATATYPE_FLOAT:
            return MPI_Allgather(sendbuf, size, MPI_FLOAT, recvbuf, size, MPI_FLOAT, (MPI_Comm)(*group_));
        case KUPL_SHM_DATATYPE_DOUBLE:
            return MPI_Allgather(sendbuf, size, MPI_DOUBLE, recvbuf, size, MPI_DOUBLE, (MPI_Comm)(*group_));
        default:
            printf("not support datatype");
            return KUPL_ERROR;
    }
}

static int oob_barrier_callback(void *group)
{
	int *group_ = (int *)group;
    return MPI_Barrier((MPI_Comm)(*group_));
}

inline double get_clock_us()
{
    struct timeval clock;
    gettimeofday(&clock, NULL);
    return 1000000.0 * clock.tv_sec + clock.tv_usec;
}

const int test_times = 100;

MPI_Comm node_comm, socket_comm, die_comm;

kupl_shm_comm_h kupl_node_comm, kupl_socket_comm, kupl_die_comm;
kupl_shm_win_h kupl_node_win, kupl_socket_win, kupl_die_win;

int global_rank, global_size;

uint8_t *buffers[16];
uint8_t *local_buffer;

constexpr size_t MAX_BUFFER_SIZE = 1 << 24;

void shm_init()
{
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    FLASH_ASSERT(global_size % 16 == 0);

    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;

    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 4, global_rank % 4, &die_comm);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 8, global_rank % 8, &socket_comm);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 16, global_rank % 16, &node_comm);

    kupl_shm_comm_create(4, global_rank % 4, global_rank, oob_cbs_h, (void *)(&die_comm), &kupl_die_comm);
    kupl_shm_comm_create(8, global_rank % 8, global_rank, oob_cbs_h, (void *)(&socket_comm), &kupl_socket_comm);
    kupl_shm_comm_create(16, global_rank % 16, global_rank, oob_cbs_h, (void *)(&node_comm), &kupl_node_comm);

    kupl_shm_win_alloc(1, kupl_die_comm, (void **)&local_buffer, &kupl_die_win);
    kupl_shm_win_alloc(1, kupl_socket_comm, (void **)&local_buffer, &kupl_socket_win);
    kupl_shm_win_alloc(MAX_BUFFER_SIZE, kupl_node_comm, (void **)&local_buffer, &kupl_node_win);

    kupl_shm_fence(kupl_node_win);

    for (int i = 0; i < 16; ++i) {
        kupl_shm_win_query(kupl_node_win, i, (void **)&buffers[i]);
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

template <typename scalar_t, int world_size, bool is_hierarchical, bool is_same_buffer>
void allgather_test(int batch, int sendcount)
{
    int rank = global_rank % 16;
    int recvcount = sendcount * world_size;
    scalar_t *sendbuf = (scalar_t *)local_buffer;
    if (!is_same_buffer) {
        sendbuf = (scalar_t *)malloc(batch * sendcount * sizeof(scalar_t));
    }
    int buffer_size = MAX_BUFFER_SIZE - batch * recvcount * sizeof(scalar_t);
    scalar_t *recvbuf = (scalar_t *)(local_buffer + buffer_size);
    auto barrier = [&]() {
        if constexpr(world_size <= 8) {
            kupl_shm_fence(kupl_socket_win);
        } else {
            kupl_shm_fence(kupl_node_win);
        }
    };
    for (int i = 0; i < batch * sendcount; ++i) {
        sendbuf[i] = gen(i, rank);
    }
    kupl_shm_fence(kupl_node_win);
    kutacc::shm_allgather<scalar_t, world_size, is_hierarchical>(batch, sendbuf, sendcount, recvbuf, recvcount,
                                                                 rank, buffers, buffer_size, barrier);
    kupl_shm_fence(kupl_node_win);
    for (int i = 0; i < batch * recvcount; ++i) {
        int b = i / recvcount;
        int r = rank / world_size * world_size + i % recvcount / sendcount;
        FLASH_ASSERT(recvbuf[i] == (scalar_t)gen(b * sendcount + i % sendcount, r));
    }

    if (!is_same_buffer) {
        free(sendbuf);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    shm_init();

    // cases in DeepSeek inference
    allgather_test<uint8_t, 16, true, false>(128, 8);
    allgather_test<bfloat16_t, 16, false, true>(128, 16);
    allgather_test<bfloat16_t, 8, false, true>(128, 264);

    shm_finalize();
    MPI_Finalize();
    return 0;
}