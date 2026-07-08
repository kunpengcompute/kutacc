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
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include "../utils/timer.h"
#include "../utils/memory.h"
#include "../moe/internal.h"
#include "kutacc.h"

namespace kutacc {
static buf_mr_info_t my_allg_send;
static buf_mr_info_t my_allg_recv;
static buf_mr_info_t *allg_remote_buf;
kurmcl_iov_t *iov = NULL;

void kurmcl_allgather_init(void *send_buf, int send_size, void *recv_buf, int recv_size, kurmcl_conn_info_t *conn_info)
{
    int comm_size = conn_info->comm_size;
    int dtp = 8;
    allg_remote_buf = (buf_mr_info_t *)malloc(comm_size * sizeof(buf_mr_info_t));
    kurmcl_reg_mr(&my_allg_send, send_buf, send_size, conn_info);
    kurmcl_reg_mr(&my_allg_recv, recv_buf, recv_size, conn_info);
    kurmcl_exchange_mr_info(conn_info, &my_allg_recv, allg_remote_buf);
    iov = (kurmcl_iov_t *)malloc(comm_size * sizeof(kurmcl_iov_t));  // 申请iovlist
    kurmcl_barrier(conn_info);
}

int kurmcl_allgather(void *send_buf, int send_size, void *recv_buf, int recv_size, kurmcl_conn_info_t *conn_info)
{
    int comm_size = conn_info->comm_size;
    int rank = conn_info->my_rank;
    int dtp = 8;
    int peer_nums = comm_size / dtp;
    int node_id = rank / dtp;
    kutacc::parallel_for(0, peer_nums, 1, [&](int64_t start, int64_t end) {
        for (int scale = start; scale < end; ++scale) {
            // int remote = (scale * dtp + rank) % comm_size;
            int remote = scale * dtp + (rank % dtp);
            if (remote != rank) {
                iov[remote].len = send_size;
                iov[remote].local_buffer = my_allg_send.buffer;
                iov[remote].lkey = my_allg_send.lkey;
                iov[remote].remote_buffer = allg_remote_buf[remote].buffer + node_id * send_size;
                iov[remote].rkey = allg_remote_buf[remote].rkey;
                kurmcl_put(&iov[remote], 1, remote, 1, conn_info);
            } else {
                memcpy((char *)recv_buf + node_id * send_size, send_buf, send_size);
            }
        }
    });
    kutacc::parallel_for(0, peer_nums, 1, [&](int64_t start, int64_t end) {
        for (int scale = start; scale < end; ++scale) {
            // int remote = (scale * dtp + rank) % comm_size;
            int remote = scale * dtp + (rank % dtp);
            if (remote != rank) {
                kurmcl_recv_imm_cnt(1, remote, conn_info);
            }
        }
    });
    kurmcl_flush(peer_nums - 1, conn_info);
    return 0;
}

void kurmcl_allgather_finalize()
{
    free(iov);
    kurmcl_dreg_mr(&my_allg_send);
    kurmcl_dreg_mr(&my_allg_recv);
}

}
