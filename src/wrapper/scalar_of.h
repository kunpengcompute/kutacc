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

#ifndef KUTACC_SCALAR_OF_H
#define KUTACC_SCALAR_OF_H

#include <arm_sve.h>

namespace kutacc {
template <typename vector_t>
struct scalar_of {};
template <typename vector_t>
using scalar_of_t = typename scalar_of<vector_t>::type;

#define DEFINE_WRAPPER(nbits, short_scalar_t, scalar_t, vector_t)   \
    template <>                                                     \
    struct scalar_of<vector_t>{                                     \
        using type = scalar_t;                                      \
    };
#include "wrapper-incl.h"
#undef DEFINE_WRAPPER
}   //  namespace kutacc

#endif
