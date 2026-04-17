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
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <mpi.h>

#include "parma.h"
#include "check.h"
#include "memory_pool.h"

Params_latency params;
namespace utils {
namespace {
constexpr int64_t ON_PACKAGE_MEMORY_ALIGNMENT = 2 * 1024 * 1024;
constexpr int64_t ON_PACKAGE_MEMORY_SIZE =
	(1'000'000'000 + ON_PACKAGE_MEMORY_ALIGNMENT - 1) / ON_PACKAGE_MEMORY_ALIGNMENT * ON_PACKAGE_MEMORY_ALIGNMENT;
// now use kupl_barrier set 50
constexpr int64_t SHM_SIZE =
    (25'000'000 + ON_PACKAGE_MEMORY_ALIGNMENT - 1) / ON_PACKAGE_MEMORY_ALIGNMENT * ON_PACKAGE_MEMORY_ALIGNMENT;
constexpr int64_t DDR_SIZE = 18'000'000'000;
}  // namespace

u8span on_package_memory_pool{};
u8span on_package_memory_available{};
u8span ddr_pool{};
u8span ddr_available{};
u8span shm_pool{};
u8span shm_available{};

int64_t kupl_shm_total_size, group_id, group_rank, intra_socket_id, intra_socket_rank;
int global_size, global_rank;
kupl_shm_comm_h kupl_comm, kupl_intra_socket_comm;
kupl_shm_win_h kupl_win;
kupl_shm_win_h kupl_win_intra_socket;
void *kupl_shm_baseptr;
void *kupl_intra_socketfence_baseptr;
std::vector<void *> kupl_shm_group_baseptr;
MPI_Comm intra_comm;
MPI_Comm intra_socket_comm;

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

void init_memory_pool()
{
    int intra_size = 16;
    int intra_socket_size = 8;
    kupl_shm_total_size = SHM_SIZE;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &global_size);
    MPI_Comm_rank(global_comm, &global_rank);

    group_id = global_rank / intra_size;
    group_rank = global_rank % intra_size;
    intra_socket_id = global_rank / intra_socket_size;
    intra_socket_rank = global_rank % intra_socket_size;
    MPI_Comm_split(global_comm, group_id, group_rank, &intra_comm);
    MPI_Comm_split(global_comm, intra_socket_id, intra_socket_rank, &intra_socket_comm);

    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;

    kupl_shm_comm_create(intra_size, group_rank, global_rank, oob_cbs_h, (void *)(&intra_comm), &kupl_comm);
    kupl_shm_comm_create(intra_socket_size, intra_socket_rank, global_rank, oob_cbs_h, (void *)(&intra_socket_comm),
        &kupl_intra_socket_comm);

    int success = kupl_shm_win_alloc(64, kupl_intra_socket_comm, (void **)&kupl_intra_socketfence_baseptr,
        &kupl_win_intra_socket);
    success = kupl_shm_win_alloc(SHM_SIZE, kupl_comm, (void **)&kupl_shm_baseptr, &kupl_win);
    KUTACC_CHECK(success == 0, "node ", params.world_rank / params.local_size, ", numa ", params.local_rank);
    memset(kupl_shm_baseptr, 0, SHM_SIZE);
    MPI_Barrier(intra_comm);

    kupl_shm_group_baseptr.resize(intra_size, nullptr);
    for (int i = 0; i < intra_size; ++i) {
        if (i != group_rank) {
            kupl_shm_win_query(kupl_win, i, (void **)&kupl_shm_group_baseptr[i]);
        } else {
            kupl_shm_group_baseptr[i] = kupl_shm_baseptr;
        }
    }
    auto shm = static_cast<uint8_t *>(kupl_shm_baseptr);
    shm_pool = shm_available = u8span{shm, SHM_SIZE};
    auto on_package_memory = static_cast<uint8_t *>(
        mmap(NULL, ON_PACKAGE_MEMORY_SIZE,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0));
    int cpu = sched_getcpu();
    int rnode = numa_node_of_cpu(cpu) + 16;
    unsigned long mask = 1UL << rnode;
    success = mbind(on_package_memory, ON_PACKAGE_MEMORY_SIZE, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT);
    KUTACC_CHECK(success == 0, "node ", params.world_rank / params.local_size, ", numa ", params.local_rank);
    on_package_memory_pool = on_package_memory_available = u8span{on_package_memory, ON_PACKAGE_MEMORY_SIZE};
    auto ddr = new uint8_t[DDR_SIZE];
    ddr_pool = ddr_available = u8span{ddr, DDR_SIZE};
}

void get_peer_shm_baseptr(int64_t peer_rank, void *local_base_ptr, void **remote_base_ptr)
{
    KUTACC_CHECK(local_base_ptr >= kupl_shm_group_baseptr[group_rank],
        "input local_base_ptr must exceed kupl_shm_baseptr");
    KUTACC_CHECK((char *)local_base_ptr < (char *)kupl_shm_group_baseptr[group_rank] + kupl_shm_total_size,
        "input pointer cannot exceed the size of the shared memory pool");
    *remote_base_ptr =
        (char *)local_base_ptr - (char *)kupl_shm_group_baseptr[group_rank] + (char *)kupl_shm_group_baseptr[peer_rank];
}

void *get_first_aligned_address(void *buffer, uintptr_t alignment)
{
    uintptr_t alignment_mask = alignment - 1;
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer);
    uintptr_t aligned_addr = (addr + alignment_mask) & ~alignment_mask;
    return reinterpret_cast<void *>(aligned_addr);
}

void *alloc_aligned_buf(int64_t buf_size, int alignment, u8span &hold)
{
    void *buf_ptr = get_first_aligned_address(hold.ptr, alignment);
    hold = hold.subspan((uint8_t *)buf_ptr - hold.ptr + buf_size);
    return buf_ptr;
}

void finalize_memory_pool()
{
    munmap(on_package_memory_pool.ptr, ON_PACKAGE_MEMORY_SIZE);
    kupl_shm_win_free(kupl_win);
    kupl_shm_win_free(kupl_win_intra_socket);
    kupl_shm_comm_destroy(kupl_comm);
    kupl_shm_comm_destroy(kupl_intra_socket_comm);
    delete[] ddr_pool.ptr;
}
}  // namespace utils
