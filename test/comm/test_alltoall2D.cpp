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
#include <kutacc.h>

#include "helper.h"

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
    static bool init = false;
    if (!init) {
        kutacc_time_init();
        init = true;
    }
    return kutacc_now_ns() / 1000.0;
}

inline bfloat16_t gen(int64_t i)
{
    return i % 17;
}

int global_rank, global_size;
constexpr int test_times = 100;
constexpr int MAX_COMM_SIZE = 16;

#include <memory>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>

template <bool on_package = true>
void *mmap_huge_page_memory(int64_t size)
{
    constexpr int64_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
    FLASH_ASSERT(size % HUGE_PAGE_SIZE == 0);
    void *addr =
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    int cpu = sched_getcpu();
    int rnode = numa_node_of_cpu(cpu) + (on_package ? 16 : 0);
    unsigned long mask = 1UL << rnode;
    int success = mbind(addr, size, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE);
    FLASH_ASSERT(success == 0);
    return addr;
}

struct CacheFlush {
    static constexpr int64_t ALIGNMENT = 64;
    std::vector<int8_t> xor_sum;
    int64_t size;
    void *buffer;

    CacheFlush(int64_t size_per_thread)
    {
        size = size_per_thread * kutacc::get_thread_num();
        buffer = mmap_huge_page_memory(size);
        memset(buffer, 0, size);
        xor_sum.resize(kutacc::get_thread_num() * ALIGNMENT, 0);
    }

    ~CacheFlush()
    {
        for (auto x : xor_sum) {
            FLASH_ASSERT(x == 0);
        }
        munmap(buffer, size);
    }

    void operator ()()
    {
        kutacc::parallel_for(0, size, 1, [&](int64_t begin, int64_t end) {
            int tid = kutacc::get_thread_id();
            for (int64_t i = begin; i < end; ++i) {
                xor_sum[tid * ALIGNMENT] ^= ((int8_t *)buffer)[i];
            }
        });
    }
};


void alltoall2D_test(int comm_size, int64_t height, int64_t width, int64_t dim)
{
    constexpr int64_t CACHE_FLUSH_SIZE = 10 * 1024 * 1024;
    static CacheFlush cache_flush(CACHE_FLUSH_SIZE);
    MPI_Comm comm;
    MPI_Comm node_comm;
    MPI_Comm socket_comm;
    MPI_Comm die_comm;

    kupl_shm_comm_h kupl_comm;
    kupl_shm_comm_h kupl_node_comm;
    kupl_shm_comm_h kupl_socket_comm;
    kupl_shm_comm_h kupl_die_comm;
    kupl_shm_win_h kupl_win;
    kupl_shm_win_h kupl_node_win;
    kupl_shm_win_h kupl_socket_win;
    kupl_shm_win_h kupl_die_win;

    bfloat16_t *send_buffers[MAX_COMM_SIZE];
    bfloat16_t *recv_buffers[MAX_COMM_SIZE];

    int64_t comm_rank = global_rank % comm_size;
    int64_t buffer_size = height * width * 2 * sizeof(bfloat16_t);

    FLASH_ASSERT(global_size % 16 == 0);

    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;

    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 4, global_rank % 4, &die_comm);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 8, global_rank % 8, &socket_comm);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 16, global_rank % 16, &node_comm);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank / comm_size, global_rank % comm_size, &comm);

    kupl_shm_comm_create(4, global_rank % 4, global_rank, oob_cbs_h, (void *)(&die_comm), &kupl_die_comm);
    kupl_shm_comm_create(8, global_rank % 8, global_rank, oob_cbs_h, (void *)(&socket_comm), &kupl_socket_comm);
    kupl_shm_comm_create(16, global_rank % 16, global_rank, oob_cbs_h, (void *)(&node_comm), &kupl_node_comm);
    kupl_shm_comm_create(comm_size, global_rank % comm_size, global_rank, oob_cbs_h, (void *)(&comm), &kupl_comm);

    kupl_shm_win_alloc(1, kupl_die_comm, (void **)&send_buffers[comm_rank], &kupl_die_win);
    kupl_shm_win_alloc(1, kupl_socket_comm, (void **)&send_buffers[comm_rank], &kupl_socket_win);
    kupl_shm_win_alloc(1, kupl_node_comm, (void **)&send_buffers[comm_rank], &kupl_node_win);
    kupl_shm_win_alloc(buffer_size, kupl_comm, (void **)&send_buffers[comm_rank], &kupl_win);

    kupl_shm_fence(kupl_win);

    for (int i = 0; i < comm_size; ++i) {
        kupl_shm_win_query(kupl_win, i, (void **)&send_buffers[i]);
        recv_buffers[i] = send_buffers[i] + height * width;
    }

    kutacc::shm_alltoall_request_h request;
    size_t extra_buffer_size = 0;
    kutacc::shm_alltoall_request_create(comm_rank, comm_size, height * width, kutacc::SHM_DATATYPE_BFLOAT16,
        extra_buffer_size, request);
    kutacc::shm_alltoall_request_init(nullptr, kupl_win, kupl_die_win, kupl_socket_win, kupl_node_win, request);

    for (int t = 0; t < test_times; ++t) {
        for (int64_t i = 0; i < height; ++i) {
            for (int64_t j = 0; j < width; ++j) {
                send_buffers[comm_rank][i * width + j] = dim == 0 ?
                    (comm_rank * height + i) * width + j : i * comm_size * width + comm_rank * width + j;
                recv_buffers[comm_rank][i * width + j] = 0;
            }
        }
        kupl_shm_fence(kupl_win);
        kutacc::shm_alltoall2D(
            (void **)send_buffers, (void *)recv_buffers[comm_rank], height, width, dim, true, request);
    }
    for (int t = 0; t < test_times; ++t) {
        for (int64_t i = 0; i < height; ++i) {
            for (int64_t j = 0; j < width; ++j) {
                send_buffers[comm_rank][i * width + j] = gen(dim == 0 ?
                    (comm_rank * height + i) * width + j : i * comm_size * width + comm_rank * width + j);
                recv_buffers[comm_rank][i * width + j] = 0;
            }
        }
        cache_flush();
        kupl_shm_fence(kupl_win);

        double start = get_clock_us();
        kutacc::shm_alltoall2D(
            (void **)send_buffers, (void *)recv_buffers[comm_rank], height, width, dim, true, request);
        double end = get_clock_us();
    }

    if (dim == 0) {
        for (int64_t i = 0; i < height * comm_size; ++i) {
            for (int64_t j = 0; j < width / comm_size; ++j) {
                bfloat16_t ans = gen(i * width + comm_rank * (width / comm_size) + j);
                if (recv_buffers[comm_rank][i * (width / comm_size) + j] != ans) {
                    FLASH_ASSERT(0);
                }
            }
        }
    } else {
        for (int64_t i = 0; i < height / comm_size; ++i) {
            for (int64_t j = 0; j < width * comm_size; ++j) {
                bfloat16_t ans = gen((comm_rank * (height / comm_size) + i) * (width * comm_size) + j);
                if (recv_buffers[comm_rank][i * (width * comm_size) + j] != ans) {
                    FLASH_ASSERT(0);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    kupl_shm_win_free(kupl_win);
    kupl_shm_win_free(kupl_socket_win);
    kupl_shm_win_free(kupl_die_win);
    kupl_shm_win_free(kupl_node_win);
    kupl_shm_comm_destroy(kupl_comm);
    kupl_shm_comm_destroy(kupl_socket_comm);
    kupl_shm_comm_destroy(kupl_die_comm);
    kupl_shm_comm_destroy(kupl_node_comm);
}

int main(int argc, char *argv[])
{
    kutacc::global_parallel_launch([&] {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &global_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

        std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> test_params = {
            {8, 64, 16 * 576, 1},
            // {8, 128, 16 * 576, 1},
            // {8, 256, 16 * 576, 1},
            // {8, 8, 128 * 512, 0},
            // {8, 16, 128 * 512, 0},
            // {8, 32, 128 * 512, 0}
        };
        int test_count = 0;
        for (auto [comm_size, height, width, dim] : test_params) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (global_rank == 0) {
                printf("--------------- Alltoall2D Test %d ---------------\n", ++test_count);
                printf("comm_size = %ld\n", comm_size);
                printf("height = %ld\n", height);
                printf("width = %ld\n", width);
                printf("dimension = %ld\n", dim);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            alltoall2D_test(comm_size, height, width, dim);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (global_rank == 0) {
            printf("--------------------------------------------------\n");
        }
        MPI_Finalize();
    });
    return 0;
}