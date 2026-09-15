#include <mpi.h>
#include <kupl.h>
#include <cstring>
#include <atomic>

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
static int convert(T datatype) {
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
    }
    else if constexpr(std::is_same<T, kutacc::kurmcl_datatype_t>::value) {
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
    return MPI_Alltoall(sendbuf, sendcount, mpi_send_datatype, recvbuf, recvcount, mpi_recv_datatype, (MPI_Comm)(*group_));
}

const int test_times = 500;
MPI_Comm socket_comm;

kupl_shm_comm_h kupl_socket_comm;
kupl_shm_win_h kupl_socket_win;
kutacc::kurmcl_conn_info_h usr_conn_info;

int global_rank, global_size;

uint8_t *buffers[8];
uint8_t *local_buffer;
uint8_t *local_send_buffer;

constexpr size_t MAX_BUFFER_SIZE = 1 << 28;

void shm_init()
{
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    FLASH_ASSERT(global_size % 8 == 0);

    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;
    kutacc::kurmcl_oob_cb_t kurmcl_oob_cbs;
    kutacc::kurmcl_oob_cb_h kurmcl_oob_cbs_h = &kurmcl_oob_cbs;
    kurmcl_oob_cbs_h->oob_allgather = oob_allgather_callback;
    kurmcl_oob_cbs_h->oob_barrier = oob_barrier_callback;
    kurmcl_oob_cbs_h->oob_alltoall = oob_alltoall_callback;

    MPI_Comm_split(MPI_COMM_WORLD, global_rank / 8, global_rank % 8, &socket_comm);

    kupl_shm_comm_create(8, global_rank % 8, global_rank, oob_cbs_h, (void *)(&socket_comm), &kupl_socket_comm);
    kupl_shm_win_alloc(MAX_BUFFER_SIZE, kupl_socket_comm, (void **)&local_buffer, &kupl_socket_win);

    MPI_Comm world_comm = MPI_COMM_WORLD;
    kutacc::kurmcl_comm_create(global_size, global_rank, kurmcl_oob_cbs_h, (void *)(&world_comm), &usr_conn_info);
    kupl_shm_fence(kupl_socket_win);

    for (int i = 0; i < 8; ++i) {
        kupl_shm_win_query(kupl_socket_win, i, (void **)&buffers[i]);
    }
}

void shm_finalize()
{
    kupl_shm_win_free(kupl_socket_win);
    kupl_shm_comm_destroy(kupl_socket_comm);
}

