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
 
#ifndef PP_H
#define PP_H

#include "core/kurmcl/kurmcl_impl.h"

namespace kutacc {
buf_mr_info_t pp_buf_send, pp_buf_recv;
buf_mr_info_t *pp_remote_buf;
kurmcl_iov_t *iovlist_for_pp = NULL;
char *p2p_barrier_sendbuf, *p2p_barrier_recvbuf;
buf_mr_info_t p2p_barrier_send, p2p_barrier_recv;
buf_mr_info_t *p2p_barrier_remote;

void pp_init(uint8_t *base_ptr, int64_t size, kurmcl_conn_info_h ds_conn_info);
void pp_put(int64_t dest_rank, int64_t send_offset, int64_t size, kurmcl_conn_info_h ds_conn_info);
void pp_recv(int64_t src_rank, kurmcl_conn_info_h ds_conn_info);
void pp_barrier(int64_t remote_rank, kurmcl_conn_info_h ds_conn_info);
}
#endif
