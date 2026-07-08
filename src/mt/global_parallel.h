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
#include <kupl.h>

#include "kutacc.h"
#include "utils/check.h"

namespace kutacc {

namespace internal {

inline int64_t divup(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

} // namespace internal

int64_t get_thread_num()
{
    return kupl_get_num_executors();
}

int64_t get_thread_id()
{
    return kupl_get_executor_num();
}

struct kupl_parallel_info {
    int exes[38];
    kupl_egroup_h eg0 = nullptr;
    kupl_egroup_h eg1 = nullptr;

    kupl_parallel_info()
    {
        int num_threads = static_cast<int>(get_thread_num());
        for (int i = 0; i < num_threads; i++) {
            exes[i] = i;
        }
        eg0 = kupl_egroup_create(exes, num_threads);
        eg1 = kupl_egroup_create(exes, num_threads);
    }

    ~kupl_parallel_info()
    {
        kupl_egroup_destroy(eg0);
        kupl_egroup_destroy(eg1);
    }
};

static kupl_parallel_info *info;

void parallel_barrier()
{
    kupl_egroup_barrier(info->eg0);
}

struct pf_func_args_t {
    int64_t begin = 0;
    int64_t end = 0;
    int64_t chunk_size = 0;
    const std::function<void(int64_t, int64_t)> *g_f = nullptr;
} pf_func_args;


void pf_func(int tid)
{
    auto [begin, end, chunk_size, g_f] = pf_func_args;
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
        (*g_f)(begin_tid, std::min(end, chunk_size + begin_tid));
    }
}

void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)> &f)
{
    KUTACC_CHECK(grain_size > 0, "grain_size invalid: ", grain_size);
    if (begin >= end) {
        return;
    }
    int64_t num_threads = std::min(get_thread_num(), internal::divup(end - begin, grain_size));
    int64_t chunk_size = internal::divup(end - begin, num_threads);
    if (num_threads == 1) {
        f(begin, end);
    } else {
        pf_func_args = {begin, end, chunk_size, &f};
        kupl_egroup_fork_barrier(info->eg1);
        pf_func(0);
        kupl_egroup_join_barrier(info->eg1);
    }
}

static void global_parallel_kernel(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    const std::function<void()> *f = (std::function<void()> *)args;
    static int global_flag;
    if (tid == 0) {
        global_flag = 1;
    }
    kupl_egroup_barrier(info->eg0);
    if (tid == 0) {
        (*f)();
        pf_func_args = {0, 0, 0, nullptr};
        global_flag = 0;
        kupl_egroup_fork_barrier(info->eg1);
        kupl_egroup_join_barrier(info->eg1);
    } else {
        while (global_flag) {
            kupl_egroup_fork_barrier(info->eg1);
            pf_func(tid);
            kupl_egroup_join_barrier(info->eg1);
        }
    }
}


void global_parallel_launch(const std::function<void()> &f)
{
    static kupl_parallel_info info_t;
    info = &info_t;
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = NULL,
        .egroup = NULL,
        .concurrency = static_cast<int>(get_thread_num()),
        .policy = KUPL_LOOP_POLICY_STATIC
    };
    kupl_parallel_for(&desc, global_parallel_kernel, (void *)&f);
}

} // namespace kutacc