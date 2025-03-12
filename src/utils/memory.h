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

#ifndef KUTACC_MEMORY_H
#define KUTACC_MEMORY_H

#include <cstdint>
#include <memory>
#include <malloc.h>
#include "kutacc_malloc.h"
#include "check.h"

namespace kutacc {
template <typename T>
struct KuTACC_MallocDeleter {
    void operator()(T *ptr) const
    {
        kutacc_free(ptr);
    }
};

template <typename T>
inline std::unique_ptr<T[], KuTACC_MallocDeleter<T> > alloc(int64_t size)
{
    void *ptr;
    size_t real_size = static_cast<size_t>(size);
    kutacc_posix_memalign(&ptr, 64, real_size * sizeof(T));
    return std::unique_ptr<T[], KuTACC_MallocDeleter<T>>(static_cast<T*>(ptr));
}
}   // namespace kutacc

#endif