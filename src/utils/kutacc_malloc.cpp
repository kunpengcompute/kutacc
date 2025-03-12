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

#include "kutacc_malloc.h"
#include <dlfcn.h>
#include <stdlib.h>
#ifdef USE_OPM
#include <memkind.h>
#endif

int kutacc_posix_memalign(void **memptr, size_t alignment, size_t size)
{
#ifdef USE_OPM
    return memkind_posix_memalign(MEMKIND_HBW_HUGETLB, memptr, alignment, size);
#else
    return posix_memalign(memptr, alignment, size);
#endif
}

void kutacc_free(void *ptr)
{
    if (ptr == nullptr) {
        return;
    }
#ifdef USE_OPM
    memkind_free(MEMKIND_HBW_HUGETLB, ptr);
    ptr = nullptr;
    return;
#else
    free(ptr);
    ptr = nullptr;
    return;
#endif
}
