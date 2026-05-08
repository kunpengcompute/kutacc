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

#ifndef KUTACC_SVWHILELT_H
#define KUTACC_SVWHILELT_H

#include <arm_sve.h>

namespace kutacc {
template <typename scalar_t>
svbool_t svwhilelt(int32_t a, int32_t b);
template <typename scalar_t>
svbool_t svwhilelt(int64_t a, int64_t b);

#define DEFINE_WRAPPER(nbits, short_scalar_t, scalar_t, vector_t)                                \
    template <>                                                                                  \
    inline __attribute__((__always_inline__)) svbool_t svwhilelt<scalar_t>(int32_t a, int32_t b) \
    {                                                                                            \
        return svwhilelt_b##nbits(a, b);                                                         \
    }                                                                                            \
    template <>                                                                                  \
    inline __attribute__((__always_inline__)) svbool_t svwhilelt<scalar_t>(int64_t a, int64_t b) \
    {                                                                                            \
        return svwhilelt_b##nbits(a, b);                                                         \
    }
#include "wrapper-incl.h"
#undef DEFINE_WRAPPER
} // namespace kutacc

#endif
