/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuTACC is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <unistd.h>
#include "../utils/check.h"
#include "kutacc.h"
#include "kupl.h"
#include <mpi.h>

#define KUPL_CHECK(cmd)                                          \
    do {                                                         \
        int status = cmd;                                        \
        KUTACC_CHECK(status == KUPL_OK, std::to_string(status)); \
    } while (0)

int64_t world_size = 1;
int64_t rank = 0;
int64_t buffer_size = 0;

void *kupl_recvbuf;
kupl_shm_win_h kupl_recvbuf_win;
kupl_shm_comm_h kupl_comm;

static int oob_barrier_callback(void *group)
{
    return MPI_Barrier((MPI_Comm)group);
}

static int oob_allgather_callback(const void *sendbuf, void *recvbuf, int size, void *group,
                                  kupl_shm_datatype_t datatype)
{
    switch (datatype) {
        case KUPL_SHM_DATATYPE_CHAR:
            return MPI_Allgather(sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR, (MPI_Comm)group);
        default:
            fprintf(stderr, "not support datatype");
            return KUPL_ERROR;
    }
}

void kutacc_comm_init(int64_t _world_size, int64_t _rank, int64_t _buffer_size)
{
    world_size = _world_size;
    rank = _rank;
    buffer_size = _buffer_size;
    int pid = getpid();
    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;
    KUPL_CHECK(kupl_shm_comm_create((int)world_size, (int)rank, pid, oob_cbs_h, (void *)MPI_COMM_WORLD, &kupl_comm));
    KUPL_CHECK(kupl_shm_win_alloc((size_t)buffer_size, kupl_comm, &kupl_recvbuf, &kupl_recvbuf_win));
}

void kutacc_comm_fini()
{
    KUPL_CHECK(kupl_shm_win_free(kupl_recvbuf_win));
    KUPL_CHECK(kupl_shm_comm_destroy(kupl_comm));
}
