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
#include <pthread.h>
#include <stdint.h>
#include <memory>

#include "kutacc.h"

#define FALSE_SHARING_LEN 128
#define MAX_NUM_ASYNC_THREADS 5
#define ALIGNED __attribute__((__aligned__(FALSE_SHARING_LEN)))
#define NCORE_PER_NUMA 38

namespace kutacc {

typedef enum {
    KUASYNC_EXECSTATE_WAITING = 0,
    KUASYNC_EXECSTATE_EXECUTING = 1,
    KUASYNC_EXECSTATE_SHUTDOWN = 2,
} KUASYNC_EXECSTATE;

typedef int32_t kuasync_signal;

struct kuasync_exec;

typedef struct {
    int tid;
    int tnum;
    int core_offset;
    void (*routine)(void *);
    void *args;
    kuasync_signal state;
    kuasync_exec *executor;
} ALIGNED kuasync_slot;

typedef struct kuasync_exec {
    int num_threads;
    int target_slot = 0;
    kuasync_signal state;
    pthread_t threads[MAX_NUM_ASYNC_THREADS];
    kuasync_slot slot[MAX_NUM_ASYNC_THREADS];
} *async_tg_h;

thread_local int kuasync_thread_id = 0;
thread_local int kuasync_num_threads = 0;

static inline kuasync_signal kuasync_load_signal(const kuasync_signal *ptr)
{
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

static inline void kuasync_store_signal_relaxed(kuasync_signal *ptr, kuasync_signal value)
{
    __atomic_store_n(ptr, value, __ATOMIC_RELAXED);
}
static inline void kuasync_store_signal(kuasync_signal *ptr, kuasync_signal value)
{
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

static void kuasync_bind_thread(const int offset)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    const int coreId = sched_getcpu();
    const int dstCoreId = coreId / NCORE_PER_NUMA * NCORE_PER_NUMA + offset;
    CPU_SET(dstCoreId, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

static inline void kuasync_wait_signal(const kuasync_signal *ptr, kuasync_signal value)
{
    while (kuasync_load_signal(ptr) != value) {
        asm volatile("\tnop\n\tnop\n\tnop\n\tnop\n");
    }
}

static inline void kuasync_distribute_task(kuasync_exec *executor)
{
    int target_slot = executor->target_slot;
    for (int i = 0; i < executor->num_threads; ++i) {
        if (i != target_slot) {
            executor->slot[i].routine = executor->slot[target_slot].routine;
            executor->slot[i].args = executor->slot[target_slot].args;
            kuasync_store_signal(&executor->slot[i].state, KUASYNC_EXECSTATE_EXECUTING);
        }
    }
}

static inline void *executor_routine(void *args)
{
    kuasync_slot *slot = (kuasync_slot *)args;
    kuasync_thread_id = slot->tid;
    kuasync_num_threads = slot->tnum;
    kuasync_bind_thread(slot->core_offset);
    kuasync_exec *executor = slot->executor;
    while (true) {
        const int state = kuasync_load_signal(&slot->state);
        if (state == KUASYNC_EXECSTATE_SHUTDOWN) {
            return NULL;
        } else if (state == KUASYNC_EXECSTATE_EXECUTING) {
            if (kuasync_thread_id == executor->target_slot) {
                kuasync_distribute_task(executor);
            }
            void (*routine)(void *) = slot->routine;
            void *task_args = slot->args;
            (*routine)(task_args); /* execute the task */
            kuasync_store_signal(&slot->state, KUASYNC_EXECSTATE_WAITING);
        }
    }
    return NULL;
}

void async_tg_task_submit(async_tg_h executor, void (*routine)(void *), void *args)
{
    executor->state = KUASYNC_EXECSTATE_EXECUTING;
    int target_slot = executor->target_slot;
    executor->slot[target_slot].routine = routine;
    executor->slot[target_slot].args = args;
    kuasync_store_signal(&executor->slot[target_slot].state, KUASYNC_EXECSTATE_EXECUTING);
}

void async_tg_wait(async_tg_h exec)
{
    if (exec->state == KUASYNC_EXECSTATE_WAITING) {
        return ;
    }
    for (int i = 0; i < exec->num_threads; i++) {
        kuasync_wait_signal(&exec->slot[i].state, KUASYNC_EXECSTATE_WAITING);
    }
    exec->state = KUASYNC_EXECSTATE_WAITING;
}

int async_tg_create(async_tg_h &executor, int num_threads, int core_offset)
{
    executor = (kuasync_exec *)malloc(sizeof(kuasync_exec));
    executor->num_threads = num_threads;
    executor->target_slot = 0;
    executor->state = KUASYNC_EXECSTATE_WAITING;
    if (num_threads > MAX_NUM_ASYNC_THREADS) {
        printf("asyncthread error: thread num %d should be less than slots %d\n", num_threads, MAX_NUM_ASYNC_THREADS);
        return 0;
    }
    for (int i = 0; i < num_threads; ++i) {
        executor->slot[i].tid = i;
        executor->slot[i].tnum = num_threads;
        executor->slot[i].executor = executor;
        executor->slot[i].core_offset = core_offset + i;
        kuasync_store_signal_relaxed(&executor->slot[i].state, KUASYNC_EXECSTATE_WAITING);
        const int rv = pthread_create(&executor->threads[i], NULL, executor_routine, &executor->slot[i]);
        if (rv != 0) {
            printf("asyncthread error: pthread create failed\n");
            return 0;
        }
    }
    return 1;
}

void async_tg_destroy(async_tg_h &executor)
{
    void *retval;
    for (int i = 0; i < executor->num_threads; ++i) {
        kuasync_store_signal(&executor->slot[i].state, KUASYNC_EXECSTATE_SHUTDOWN);
    }
    for (int i = 0; i < executor->num_threads; ++i) {
        pthread_join(executor->threads[i], &retval);
    }
    free(executor);
    executor = nullptr;
}

int async_tg_get_thread_id()
{
    return kuasync_thread_id;
}

int async_tg_get_thread_num()
{
    return kuasync_num_threads;
}

} // namespace kutacc