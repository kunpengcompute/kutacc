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

#define wait_fence(ptr)                                                                         \
    do {                                                                                        \
        int8_t flag = 0;                                                                        \
        do {                                                                                    \
            kutacc_memory_cpu_load_fence();                                                     \
            flag = *(ptr);                                                                      \
        } while (flag != 1);                                                                    \
        *(ptr) = 0;                                                                             \
    } while (0)

#define set_fence(ptr)                                                                          \
    do {                                                                                        \
        std::atomic_thread_fence(std::memory_order_seq_cst);                                    \
        *(ptr) = 1;                                                                             \
    } while (0)

#define USE_SDMA 0

namespace kutacc {

namespace {
    constexpr size_t MAX_NUM_FENCES = 6;
    constexpr size_t ALIGNMENT = 128;
}

typedef struct shm_allreduce_request {
    int comm_rank;
    int comm_size;
    size_t max_num_elements;
    size_t extra_buffer_size;
    size_t extra_element_buffer_size;
    size_t extra_fence_buffer_size;
    bfloat16_t **extra_buffers;
    kupl_event_h *sdma_events;
    kupl_shm_win_h comm_win;
    kupl_shm_win_h die_win;
    kupl_shm_win_h socket_win;
    kupl_shm_win_h node_win;
} *shm_allreduce_request_h;

void shm_allreduce_request_create(int comm_rank, int comm_size, size_t max_num_elements, shm_datatype_t datatype,
    size_t &extra_buffer_size, shm_allreduce_request_h &request)
{
    KUTACC_CHECK(datatype == SHM_DATATYPE_BFLOAT16, "");

    int num_threads = kutacc::get_thread_num();
    KUTACC_CHECK(max_num_elements % (8 * num_threads) == 0, "");
    KUTACC_CHECK(comm_size == 16 || comm_size == 8, "");

    request = (shm_allreduce_request *)malloc(sizeof(shm_allreduce_request));

    request->comm_rank = comm_rank;
    request->comm_size = comm_size;
    request->max_num_elements = max_num_elements;

    size_t buffer_size = max_num_elements * shm_datatype_size(datatype);
    request->extra_element_buffer_size = buffer_size / 8 * 2;
    request->extra_fence_buffer_size = num_threads * MAX_NUM_FENCES * ALIGNMENT;
    request->extra_buffer_size = request->extra_element_buffer_size + request->extra_fence_buffer_size;
    request->extra_buffers = (bfloat16_t **)malloc(sizeof(bfloat16_t *) * comm_size);

    request->sdma_events = (kupl_event_h *)malloc(sizeof(kupl_event_h) * num_threads);
    for (int i = 0; i < num_threads; ++i) {
        request->sdma_events[i] = kupl_event_create();
    }

    extra_buffer_size = request->extra_buffer_size;
}

void shm_allreduce_request_init(void **extra_buffers, kupl_shm_win_h comm_win, kupl_shm_win_h die_win,
                                kupl_shm_win_h socket_win, kupl_shm_win_h node_win,
                                const shm_allreduce_request_h &request)
{
    memcpy(request->extra_buffers, extra_buffers, sizeof(bfloat16_t *) * request->comm_size);
    memset(extra_buffers[request->comm_rank], 0, request->extra_buffer_size);
    request->comm_win = comm_win;
    request->die_win = die_win;
    request->socket_win = socket_win;
    request->node_win = node_win;
    kupl_shm_fence(comm_win);
}

void shm_allreduce_request_destroy(const shm_allreduce_request_h &request)
{
    int num_threads = kutacc::get_thread_num();
    free(request->extra_buffers);
    for (int i = 0; i < num_threads; ++i) {
        kupl_event_destroy(request->sdma_events[i]);
    }
    free(request->sdma_events);
    free(request);
}

inline void shm_allreduce_comm16_impl(bfloat16_t **buffers, size_t N, const shm_allreduce_request_h &request)
{
    int num_threads = kutacc::get_thread_num();
    const int &comm_rank = request->comm_rank;
    bfloat16_t *local_buffer = buffers[comm_rank];
    bfloat16_t *inter_die_buffer = buffers[comm_rank ^ 4];
    bfloat16_t *inter_socket_buffer = buffers[comm_rank ^ 8];
    bfloat16_t *intra_die_buffer[4];

    int local_rank = comm_rank % 4;
    for (int i = 0; i < 4; ++i) {
        intra_die_buffer[i] = buffers[comm_rank - local_rank + i];
    }

    static int flip_flag = 0;
    int fence_id_offset = MAX_NUM_FENCES / 2 * flip_flag;
    int extra_buffer_offset = request->extra_element_buffer_size / 2 / sizeof(bfloat16_t) * flip_flag;
    
    bfloat16_t *local_extra_buffer = request->extra_buffers[comm_rank];
    bfloat16_t *inter_socket_extra_buffer = request->extra_buffers[comm_rank ^ 8];

    local_extra_buffer += extra_buffer_offset;
    inter_socket_extra_buffer += extra_buffer_offset;
    flip_flag ^= 1;

    int8_t *local_fence_buffer = (int8_t *)request->extra_buffers[comm_rank] + request->extra_element_buffer_size;
    int8_t *inter_die_fence_buffer =
        (int8_t *)request->extra_buffers[comm_rank ^ 4] + request->extra_element_buffer_size;
    int8_t *inter_socket_fence_buffer =
        (int8_t *)request->extra_buffers[comm_rank ^ 8] + request->extra_element_buffer_size;

    int n = N / (8 * num_threads);

    kupl_shm_fence(request->die_win);

    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = kutacc::get_thread_id();
        int o = tid * 2 + (comm_rank & 4) / 4;
        int e = o ^ 1;
        bfloat16_t *ptr0[4];
        for (int i = 0; i < 4; ++i) {
            ptr0[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * e;
        }
        bfloat16_t *ptr2[5];
        ptr2[4] = inter_die_buffer + (N / 4) * local_rank + n * o;
        for (int i = 0; i < 4; ++i) {
            ptr2[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * o;
        }
        bfloat16_t *ptr3[4];
        for (int i = 0; i < 4; ++i) {
            ptr3[i] = intra_die_buffer[(local_rank + i) % 4] + (N / 4) * local_rank + n * e;
        }
        int fence_offset = (tid * MAX_NUM_FENCES + fence_id_offset);

        reduce<4, 1>(ptr0, (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * e}, n);

        set_fence(inter_die_fence_buffer + fence_offset * ALIGNMENT);
        wait_fence(local_fence_buffer + fence_offset * ALIGNMENT);

        reduce<5, 1>(ptr2, (bfloat16_t *[]){local_extra_buffer + n * tid}, n);

        set_fence(inter_socket_fence_buffer + (fence_offset + 1) * ALIGNMENT);
        wait_fence(local_fence_buffer + (fence_offset + 1) * ALIGNMENT);

#if USE_SDMA
        kupl_memcpy_async(local_buffer + (N / 4) * local_rank + n * o, inter_socket_extra_buffer + n * tid,
                          n * sizeof(bfloat16_t), nullptr, sdma_event[tid]);
        kupl_event_wait(sdma_event[tid]);
        reduce<2, 4>(
            (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * o, local_extra_buffer + n * tid}, ptr2, n);
#else
        reduce<2, 5>((bfloat16_t *[]){inter_socket_extra_buffer + n * tid, local_extra_buffer + n * tid}, ptr2, n);
#endif

        set_fence(inter_die_fence_buffer + (fence_offset + 2) * ALIGNMENT);
        wait_fence(local_fence_buffer + (fence_offset + 2) * ALIGNMENT);

#if USE_SDMA
        copy<4>(ptr3, inter_die_buffer + (N / 4) * local_rank + n * e, n);
#else
        copy<3>(ptr3 + 1, local_buffer + (N / 4) * local_rank + n * e, n);
#endif
    });

#if USE_SDMA
    kupl_shm_fence(request->socket_win);
#else
    kupl_shm_fence(request->die_win);
#endif
}

inline void shm_allreduce_comm8_impl1(bfloat16_t **buffers, size_t N, const shm_allreduce_request_h &request)
{
    int num_threads = kutacc::get_thread_num();
    const int &comm_rank = request->comm_rank;
    bfloat16_t *local_buffer = buffers[comm_rank];
    bfloat16_t *inter_die_buffer = buffers[comm_rank ^ 4];
    bfloat16_t *intra_die_buffer[4];

    int local_rank = comm_rank % 4;
    for (int i = 0; i < 4; ++i) {
        intra_die_buffer[i] = buffers[comm_rank - local_rank + i];
    }

    static int flip_flag = 0;
    int fence_id_offset = MAX_NUM_FENCES / 2 * flip_flag;
    int extra_buffer_offset = request->extra_element_buffer_size / 2 / sizeof(bfloat16_t) * flip_flag;

    flip_flag ^= 1;

    int8_t *local_fence_buffer = (int8_t *)request->extra_buffers[comm_rank] + request->extra_element_buffer_size;
    int8_t *inter_die_fence_buffer =
        (int8_t *)request->extra_buffers[comm_rank ^ 4] + request->extra_element_buffer_size;

    int n = N / (8 * num_threads);

    kupl_shm_fence(request->die_win);

    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = kutacc::get_thread_id();

        int o = tid * 2 + (comm_rank & 4) / 4;
        int e = o ^ 1;
        bfloat16_t *ptr0[4];
        for (int i = 0; i < 4; ++i) {
            ptr0[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * e;
        }
        bfloat16_t *ptr2[5];
        ptr2[4] = inter_die_buffer + (N / 4) * local_rank + n * o;
        for (int i = 0; i < 4; ++i) {
            ptr2[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * o;
        }
        bfloat16_t *ptr3[4];
        for (int i = 0; i < 4; ++i) {
            ptr3[i] = intra_die_buffer[(local_rank + i) % 4] + (N / 4) * local_rank + n * e;
        }
        int fence_offset = (tid * MAX_NUM_FENCES + fence_id_offset);

#ifdef ENABLE_EXTRA_PREFETCH
        reduce<4, 1, true, true, 1024, -1>(ptr0, (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * e}, n);
#else
        reduce<4, 1>(ptr0, (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * e}, n);
#endif
        
        set_fence(inter_die_fence_buffer + fence_offset * ALIGNMENT);
        wait_fence(local_fence_buffer + fence_offset * ALIGNMENT);

#ifdef ENABLE_EXTRA_PREFETCH
        reduce<5, 4, true, true, 1024, -1>(ptr2, ptr2, n);
#else
        reduce<5, 4>(ptr2, ptr2, n);
#endif

        set_fence(inter_die_fence_buffer + (fence_offset + 2) * ALIGNMENT);
        wait_fence(local_fence_buffer + (fence_offset + 2) * ALIGNMENT);

#ifdef ENABLE_EXTRA_PREFETCH
        copy<4, true, true, 1024, 512>(ptr3, inter_die_buffer + (N / 4) * local_rank + n * e, n);
#else
        copy<4>(ptr3, inter_die_buffer + (N / 4) * local_rank + n * e, n);
#endif
    });

    kupl_shm_fence(request->socket_win);
}

inline void shm_allreduce_comm8_impl2(bfloat16_t **buffers, size_t N, const shm_allreduce_request_h &request)
{
    int num_threads = kutacc::get_thread_num();
    const int &comm_rank = request->comm_rank;
    bfloat16_t *local_buffer = buffers[comm_rank];
    bfloat16_t *inter_die_buffer = buffers[comm_rank ^ 4];
    bfloat16_t *intra_die_buffer[4];

    int local_rank = comm_rank % 4;
    for (int i = 0; i < 4; ++i) {
        intra_die_buffer[i] = buffers[comm_rank - local_rank + i];
    }

    static int flip_flag = 0;
    int fence_id_offset = MAX_NUM_FENCES / 2 * flip_flag;
    int extra_buffer_offset = request->extra_element_buffer_size / 2 / sizeof(bfloat16_t) * flip_flag;

    bfloat16_t *local_extra_buffer = request->extra_buffers[comm_rank];
    bfloat16_t *inter_die_extra_buffer = request->extra_buffers[comm_rank ^ 4];

    local_extra_buffer += extra_buffer_offset;
    inter_die_extra_buffer += extra_buffer_offset;

    flip_flag ^= 1;

    int8_t *local_fence_buffer = (int8_t *)request->extra_buffers[comm_rank] + request->extra_element_buffer_size;
    int8_t *inter_die_fence_buffer =
        (int8_t *)request->extra_buffers[comm_rank ^ 4] + request->extra_element_buffer_size;
    
    int n = N / (4 * num_threads);

    kupl_shm_fence(request->die_win);

    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = kutacc::get_thread_id();
        bfloat16_t *ptr0[4];
        for (int i = 0; i < 4; ++i) {
            ptr0[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * tid;
        }
        bfloat16_t *ptr3[4];
        for (int i = 0; i < 4; ++i) {
            ptr3[i] = intra_die_buffer[(local_rank + i) % 4] + (N / 4) * local_rank + n * tid;
        }
        int fence_offset = (tid * MAX_NUM_FENCES + fence_id_offset);

        reduce<4, 1>(ptr0, (bfloat16_t *[]){local_extra_buffer + n * tid}, n);

        set_fence(inter_die_fence_buffer + fence_offset * ALIGNMENT);
        wait_fence(local_fence_buffer + fence_offset * ALIGNMENT);

        reduce<2, 4>((bfloat16_t *[]){local_extra_buffer + n * tid, inter_die_extra_buffer + n * tid}, ptr3, n);
    });

    kupl_shm_fence(request->die_win);
}

inline void shm_allreduce_comm8_impl3(bfloat16_t **buffers, size_t N, const shm_allreduce_request_h &request)
{
    int num_threads = kutacc::get_thread_num();
    const int &comm_rank = request->comm_rank;

    int n = N / (8 * num_threads);

    kupl_shm_fence(request->socket_win);
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
        int tid = kutacc::get_thread_id();
        bfloat16_t *ptr0[8];
        for (int i = 0; i < 8; ++i) {
            ptr0[i] = buffers[(i & 1) * 4 + (i >> 1)] + (N / 8) * comm_rank + n * tid;
        }
        reduce<8, 8>(ptr0, ptr0, n);
    });

    kupl_shm_fence(request->socket_win);
}

void shm_allreduce(void **buffers, size_t num_elements, const shm_allreduce_request_h &request)
{
    KUTACC_CHECK(num_elements <= request->max_num_elements, "too many elements");
    if (request->comm_size == 16) {
        shm_allreduce_comm16_impl((bfloat16_t **)buffers, num_elements, request);
    } else if (request->comm_size == 8) {
        shm_allreduce_comm8_impl1((bfloat16_t **)buffers, num_elements, request);
    } else {
        KUTACC_CHECK(false, "no implement for comm_size = ", request->comm_size);
    }
}

} // namespace kutacc