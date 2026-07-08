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

inline bfloat16_t gen(int i, int rank)
{
    return (i ^ rank) % 17;
}

int global_rank, global_size;
constexpr int test_times = 100;
constexpr int MAX_COMM_SIZE = 16;

#include <memory>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>
constexpr int64_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
template <bool on_package = true>
void *mmap_huge_page_memory(int64_t size)
{
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

void allreduce_test(int comm_size, size_t num_elements)
{
    static constexpr int64_t CACHE_FLUSH_SIZE = 10 * 1024 * 1024;
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

    bfloat16_t *local_buffer;
    bfloat16_t *buffers[MAX_COMM_SIZE];
    bfloat16_t *extra_buffers[MAX_COMM_SIZE];

    size_t buffer_size = num_elements * sizeof(bfloat16_t);
    size_t extra_buffer_size;

    kutacc::shm_allreduce_request_h request;

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

    kutacc::shm_allreduce_request_create(global_rank % comm_size, comm_size, num_elements,
        kutacc::SHM_DATATYPE_BFLOAT16, extra_buffer_size, request);

    kupl_shm_win_alloc(1, kupl_die_comm, (void **)&local_buffer, &kupl_die_win);
    kupl_shm_win_alloc(1, kupl_socket_comm, (void **)&local_buffer, &kupl_socket_win);
    kupl_shm_win_alloc(1, kupl_node_comm, (void **)&local_buffer, &kupl_node_win);
    kupl_shm_win_alloc(buffer_size + extra_buffer_size, kupl_comm, (void **)&local_buffer, &kupl_win);

    kupl_shm_fence(kupl_win);

    for (int i = 0; i < comm_size; ++i) {
        kupl_shm_win_query(kupl_win, i, (void **)&buffers[i]);
        extra_buffers[i] = (bfloat16_t *)buffers[i] + num_elements;
    }
    kutacc::shm_allreduce_request_init((void **)extra_buffers, kupl_win, kupl_die_win, kupl_socket_win,
        kupl_node_win, request);

    kupl_shm_fence(kupl_die_win);

    for (int t = 0; t < test_times; ++t) {
        for (size_t i = 0; i < num_elements; ++i) {
            local_buffer[i] = gen(i, global_rank);
        }
        memset(extra_buffers[global_rank % comm_size], 0, extra_buffer_size);
        kupl_shm_fence(kupl_win);
        kutacc::shm_allreduce((void **)buffers, num_elements, request);
    }

    double sum_time = 0;
    for (int t = 0; t < test_times; ++t) {
        for (size_t i = 0; i < num_elements; ++i) {
            local_buffer[i] = gen(i, global_rank);
        }
        memset(extra_buffers[global_rank % comm_size], 0, extra_buffer_size);
        cache_flush();
        kupl_shm_fence(kupl_win);

        double start = get_clock_us();
        kutacc::shm_allreduce((void **)buffers, num_elements, request);
        double end = get_clock_us();

        sum_time += end - start;
    }

    float max_diff = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        float ans = 0;
        for (int j = 0; j < comm_size; ++j) {
            ans += gen(i, global_rank / comm_size * comm_size + j);
        }
        max_diff = std::max(max_diff, abs(local_buffer[i] - ans));
    }
    printf("Rank #%02d: max_diff = %.2e, avg_time = %.2lf us\n", global_rank, max_diff, sum_time / test_times);
    FLASH_ASSERT(max_diff < 1e-5);

    kupl_shm_win_free(kupl_win);
    kupl_shm_win_free(kupl_socket_win);
    kupl_shm_win_free(kupl_die_win);
    kupl_shm_win_free(kupl_node_win);
    kupl_shm_comm_destroy(kupl_comm);
    kupl_shm_comm_destroy(kupl_socket_comm);
    kupl_shm_comm_destroy(kupl_die_comm);
    kupl_shm_comm_destroy(kupl_node_comm);
    kutacc::shm_allreduce_request_destroy(request);
}

int main(int argc, char *argv[])
{
    kutacc::global_parallel_launch([&] {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &global_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

        std::vector<std::pair<int, int>> test_params = {
            {8, 64 * 7168},
            // {8, 128 * 7168},
            // {8, 256 * 7168},
            // {8, 512 * 7168},
        };
        int test_count = 0;
        for (auto [comm_size, num_elements] : test_params) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (global_rank == 0) {
                printf("---------------- AllReduce Test %d ----------------\n", ++test_count);
                printf("comm_size = %d\n", comm_size);
                printf("num_elements = %d\n", num_elements);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            allreduce_test(comm_size, num_elements);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (global_rank == 0) {
            printf("--------------------------------------------------\n");
        }
        MPI_Finalize();
    });
    return 0;
}