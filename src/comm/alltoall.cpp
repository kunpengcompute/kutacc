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
#include "comm.h"

#include <atomic>
#include <arm_sve.h>
#include <cstring>
#include <functional>
#include <sys/time.h>
#include <kupl.h>

#include "kutacc.h"
#include "utils/check.h"
#include "utils/timer.h"
#include "utils/prf_memcpy.h"

namespace kutacc {

typedef struct shm_alltoall_request {
    int comm_rank;
    int comm_size;
    size_t max_num_elements;
    shm_datatype_t datatype;
    kupl_shm_win_h comm_win;
    kupl_shm_win_h die_win;
    kupl_shm_win_h socket_win;
    kupl_shm_win_h node_win;
} *shm_alltoall_request_h;

void shm_alltoall_request_create(int comm_rank, int comm_size, size_t max_num_elements, shm_datatype_t datatype,
    size_t &extra_buffer_size, shm_alltoall_request_h &request)
{
    request = (shm_alltoall_request *)malloc(sizeof(shm_alltoall_request));
    request->comm_rank = comm_rank;
    request->comm_size = comm_size;
    request->max_num_elements = max_num_elements;
    request->datatype = datatype;
    extra_buffer_size = 0;
}

void shm_alltoall_request_init(void **extra_buffers, kupl_shm_win_h comm_win,
    kupl_shm_win_h die_win, kupl_shm_win_h socket_win, kupl_shm_win_h node_win,
    const shm_alltoall_request_h &request)
{
    request->comm_win = comm_win;
    request->die_win = die_win;
    request->socket_win = socket_win;
    request->node_win = node_win;
    kupl_shm_fence(comm_win);
}

void shm_alltoall_request_destroy(const shm_alltoall_request_h &request)
{
    free(request);
}

template <typename T>
void shm_alltoall2D_impl(int64_t height, int64_t width, int64_t dim, T **send_buffers, T *recv_buffers,
    bool need_comm_fence, const shm_alltoall_request_h &request)
{
    if (need_comm_fence) {
        kupl_shm_fence(request->comm_win);
    }
    int comm_rank = request->comm_rank;
    int comm_size = request->comm_size;
#ifdef ENABLE_EXTRA_PREFETCH
    constexpr auto memcpy_func = utils::prf_memcpy<true, true, 3 * 1024, SV_PLDL2STRM>;
#else
    constexpr auto memcpy_func = memcpy;
#endif
    if (dim == 0) {
        int sub_b = height;
        int h = width;
        int b = sub_b * comm_size;
        int sub_h = h / comm_size;
        kutacc::parallel_for(0, b, 1, [&](int64_t start, int64_t end) {
            for (int k = start; k < end; ++k) {
                int i = k % sub_b;
                int j = k / sub_b;
                T *dst = recv_buffers + k * sub_h;
                T *src = send_buffers[j] + i * h + comm_rank * sub_h;
                memcpy_func(dst, src, sub_h * sizeof(T));
            }
        });
    } else {
        int b = height;
        int sub_h = width;
        int sub_b = b / comm_size;
        int h = sub_h * comm_size;
        kutacc::parallel_for(0, b, 1, [&](int64_t start, int64_t end) {
            for (int k = start; k < end; ++k) {
                int i = k % sub_b;
                int j = k / sub_b;
                T *src = send_buffers[j] + (comm_rank * sub_b + i) * sub_h;
                T *dst = recv_buffers + i * h + j * sub_h;
                memcpy_func(dst, src, sub_h * sizeof(T));
            }
        });
    }
    if (need_comm_fence) {
        kupl_shm_fence(request->comm_win);
    }
}

void shm_alltoall2D(void **send_buffers, void *recv_buffers, int64_t height, int64_t width, int64_t dim,
    bool need_comm_fence, const shm_alltoall_request_h &request)
{
    if (request->datatype == SHM_DATATYPE_BFLOAT16) {
        shm_alltoall2D_impl(
            height, width, dim, (bfloat16_t **)send_buffers, (bfloat16_t *)recv_buffers, need_comm_fence, request);
    } else if (request->datatype == SHM_DATATYPE_INT8) {
        shm_alltoall2D_impl(
            height, width, dim, (int8_t **)send_buffers, (int8_t *)recv_buffers, need_comm_fence, request);
    } else {
        KUTACC_CHECK(0, "Unsupported datatype.");
    }
}

} // namespace kutacc