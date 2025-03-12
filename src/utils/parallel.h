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

#ifndef KUTACC_PARALLEL_H
#define KUTACC_PARALLEL_H

#ifndef DISABLE_PARALLEL
#include <omp.h>
#endif
#include "check.h"
#include "collapse.h"

namespace kutacc {
namespace internal {
    inline int64_t divup(int64_t x, int64_t y) 
    {
        return (x + y - 1) / y;
    }
}   // namespace internal

inline int64_t get_max_threads()
{
#ifdef DISABLE_PARALLEL
    return 1;
#else
    return omp_get_max_threads();
#endif
}

inline int64_t get_num_threads()
{
#ifdef DISABLE_PARALLEL
    return 1;
#else
    return omp_get_num_threads();
#endif
}

inline void set_num_threads(int64_t num_threads)
{
#ifndef DISABLE_PARALLEL
    int real_threads = static_cast<int>(num_threads);
    return omp_set_num_threads(real_threads);
#endif
}

inline int64_t get_thread_num()
{
#ifdef DISABLE_PARALLEL
    return 0;
#else
    return omp_get_thread_num();
#endif
}

template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const F &f)
{
    KUTACC_CHECK(grain_size > 0, "grain_size invalid: ", grain_size);
    if (begin >= end) {
        return;
    }
#ifdef DISABLE_PARALLEL
    f(begin, end);
#else
    int64_t num_threads = std::min(get_max_threads(), internal::divup(end - begin, grain_size));
    if (num_threads == 1) {
        f(begin, end);
    } else {
#pragma omp parallel // num_threads(num_threads)
        {
            int64_t tid = get_thread_num();
            int64_t chunk_size = internal::divup(end - begin, num_threads);
            int64_t begin_tid = begin + tid * chunk_size;
            if (begin_tid < end) {
                f(begin_tid, std::min(end, chunk_size + begin_tid));
            }
        }
    }
#endif
}

}   // namespace kutacc

#endif
