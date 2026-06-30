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

#include "pp.h"
#include <cstdint>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "kutacc.h"

namespace kutacc {

void pp_init(uint8_t *base_ptr, int64_t size, kurmcl_conn_info_h ds_conn_info)
{
    kurmcl_reg_mr(&pp_buf_send, base_ptr, size, ds_conn_info);
    kurmcl_reg_mr(&pp_buf_recv, base_ptr, size, ds_conn_info);
    pp_remote_buf = (buf_mr_info_t*)malloc(ds_conn_info->world_size * sizeof (buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &pp_buf_recv, pp_remote_buf);
    iovlist_for_pp = (kurmcl_iov_t *)malloc(sizeof(kurmcl_iov_t));
    memset(base_ptr, 0, size);
    // for p2p barrier
    int buffer_size = ds_conn_info->world_size * sizeof(char);
    p2p_barrier_sendbuf = (char *)malloc(buffer_size);
    p2p_barrier_recvbuf = (char *)malloc(buffer_size);
    p2p_barrier_remote = (buf_mr_info_t *)malloc(ds_conn_info->world_size * sizeof(buf_mr_info_t));
    kurmcl_reg_mr(&p2p_barrier_send, p2p_barrier_sendbuf, buffer_size, ds_conn_info);
    kurmcl_reg_mr(&p2p_barrier_recv, p2p_barrier_recvbuf, buffer_size, ds_conn_info);
    kurmcl_exchange_mr_info(ds_conn_info, &p2p_barrier_recv, p2p_barrier_remote);
}

void pp_put(int64_t dest_rank, int64_t send_offset, int64_t size, kurmcl_conn_info_h ds_conn_info)
{
    iovlist_for_pp[0].len = size;
    iovlist_for_pp[0].local_buffer = pp_buf_send.buffer + send_offset;
    iovlist_for_pp[0].lkey = pp_buf_send.lkey;
    iovlist_for_pp[0].remote_buffer = pp_remote_buf[dest_rank].buffer + send_offset;
    iovlist_for_pp[0].rkey = pp_remote_buf[dest_rank].rkey;
    kurmcl_put(iovlist_for_pp, 1, dest_rank, 1, ds_conn_info);
}

void pp_recv(int64_t src_rank, kurmcl_conn_info_h ds_conn_info)
{
    kurmcl_recv_imm_cnt(1, src_rank, ds_conn_info);
    kurmcl_flush(1, ds_conn_info);
}

void pp_barrier(int64_t remote_rank, kurmcl_conn_info_h ds_conn_info)
{
    kurmcl_iov_t iov[1];
    iov[0].len = 0;
    iov[0].local_buffer = p2p_barrier_send.buffer;
    iov[0].lkey = p2p_barrier_send.lkey;
    iov[0].remote_buffer = p2p_barrier_remote[remote_rank].buffer;
    iov[0].rkey = p2p_barrier_remote[remote_rank].rkey;
    kurmcl_put(iov, 1, remote_rank, 1, ds_conn_info);
    kurmcl_recv_imm_cnt(1, remote_rank, ds_conn_info);
    kurmcl_flush(1, ds_conn_info);
}

}  // namespace kutacc