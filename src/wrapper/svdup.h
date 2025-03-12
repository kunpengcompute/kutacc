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

#ifndef KUTACC_SVDUP_H
#define KUTACC_SVDUP_H

#include "vector_of.h"
#include <arm_sve.h>

namespace kutacc {
template <typename scalar_t>
vector_of_t<scalar_t> svdup(scalar_t);

#define DEFINE_WRAPPER(nbits, short_scalar_t, scalar_t, vector_t)                               \
    template <>                                                                                 \
    inline __attribute__((__always_inline__)) vector_t svdup<scalar_t>(scalar_t value)          \
    {                                                                                           \
        return svdup_##short_scalar_t(value);                                                   \
    }
#include "wrapper-incl.h"
#undef DEFINE_WRAPPER
}   // namespace kutacc

#endif


