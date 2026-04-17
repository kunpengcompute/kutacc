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
#include "sdma_threshold.h"

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

bfloat16_t *local_buffer;
bfloat16_t *intra_node_buffers[16];
bfloat16_t *intra_node_extra_buffers[16];

constexpr size_t MAX_NUM_ELEMENTS = 128 * 7168;
constexpr size_t MAX_BUFFER_SIZE = MAX_NUM_ELEMENTS * sizeof(bfloat16_t);
size_t extra_size;

void allreduce_init()
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

    bool use_sdma;
    kutacc::shm_allreduce_get_metadata(MAX_NUM_ELEMENTS, extra_size, use_sdma);

    kupl_shm_win_alloc(1, kupl_die_comm, (void **)&local_buffer, &kupl_die_win);
    kupl_shm_win_alloc(1, kupl_socket_comm, (void **)&local_buffer, &kupl_socket_win);
    kupl_shm_win_alloc(MAX_BUFFER_SIZE + extra_size, kupl_node_comm, (void **)&local_buffer, &kupl_node_win);

    kupl_shm_fence(kupl_node_win);

    for (int i = 0; i < 16; ++i) {
        kupl_shm_win_query(kupl_node_win, i, (void **)&intra_node_buffers[i]);
        intra_node_extra_buffers[i] = (bfloat16_t *)intra_node_buffers[i] + MAX_NUM_ELEMENTS;
    }
    
    kutacc::shm_allreduce_init(global_rank % 16, MAX_NUM_ELEMENTS, intra_node_extra_buffers);

    SdmaCtlThredInit();
    if (global_rank % 16 == 0) {
        for (int i = 0; i < 4; ++i) {
            SetSdmaThreshold(i, 12);
        }
    }
    MPI_Barrier(node_comm);
}
 
void allreduce_finalize()
{
    kupl_shm_win_free(kupl_node_win);
    kupl_shm_win_free(kupl_socket_win);
    kupl_shm_win_free(kupl_die_win);
    kupl_shm_comm_destroy(kupl_node_comm);
    kupl_shm_comm_destroy(kupl_socket_comm);
    kupl_shm_comm_destroy(kupl_die_comm);
    kutacc::shm_allreduce_finalize();
}

void allreduce()
{
    kutacc::shm_allreduce(global_rank % 16, MAX_NUM_ELEMENTS, intra_node_buffers, kupl_die_win, kupl_socket_win);
}

bfloat16_t gen(int i, int rank)
{
    return (i ^ rank) % 17;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    allreduce_init();
    
    for (int t = 0; t < test_times; ++t) {
        for (int i = 0; i < MAX_NUM_ELEMENTS; ++i) {
            local_buffer[i] = gen(i, global_rank);
        }
        memset(intra_node_extra_buffers[global_rank % 16], 0, extra_size);
        kupl_shm_fence(kupl_node_win);
        allreduce();
    }

    double sum_time = 0;
    for (int t = 0; t < test_times; ++t) {
        for (int i = 0; i < MAX_NUM_ELEMENTS; ++i) {
            local_buffer[i] = gen(i, global_rank);
        }
        memset(intra_node_extra_buffers[global_rank % 16], 0, extra_size);
        kupl_shm_fence(kupl_node_win);

        double start = get_clock_us();
        allreduce();
        double end = get_clock_us();

        sum_time += end - start;
    }

    float max_diff = 0;
    for (int i = 0; i < MAX_NUM_ELEMENTS; ++i) {
        float ans = 0;
        for (int j = 0; j < 16; ++j) {
            ans += gen(i, global_rank / 16 * 16 + j);
        }
        max_diff = std::max(max_diff, abs(local_buffer[i] - ans));
    }
    printf("Rank #%02d: max_diff = %.2e, avg_time = %.2lf us\n", global_rank, max_diff, sum_time / test_times);
    FLASH_ASSERT(max_diff < 1e-5);
    allreduce_finalize();
    MPI_Finalize();
    return 0;
}