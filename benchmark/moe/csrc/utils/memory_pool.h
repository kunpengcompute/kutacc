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
#pragma once

#include <cstdint>
#include <tuple>
#include <kupl.h>

#include "span.h"

namespace utils {
extern u8span on_package_memory_pool;
extern u8span on_package_memory_available;
extern u8span ddr_pool;
extern u8span ddr_available;
extern u8span shm_pool;
extern u8span shm_available;
extern int64_t kupl_shm_total_size, group_id, group_rank, intra_socket_id, intra_socket_rank;
extern int global_size, global_rank;
extern kupl_shm_win_h kupl_win;
extern kupl_shm_win_h kupl_win_intra_socket;
void init_memory_pool();
void *alloc_aligned_buf(int64_t buf_size, int alignment, u8span &hold);
void finalize_memory_pool();
void get_peer_shm_baseptr(int64_t peer_rank, void *local_base_ptr, void **remote_base_ptr);
}  // namespace utils
