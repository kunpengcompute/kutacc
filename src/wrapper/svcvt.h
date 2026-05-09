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

#ifndef KUTACC_SVCVT_H
#define KUTACC_SVCVT_H

#include "vector_of.h"
#include <arm_sve.h>

namespace kutacc {
template <typename dst_t, typename src_t>
vector_of_t<dst_t> svcvt(svbool_t pg, vector_of_t<src_t> src);

#define DEFINE_SVCVT(short_dst_t, dst_t, src_t)                                                              \
    template <>                                                                                              \
    inline __attribute__((__always_inline__)) vector_of_t<dst_t> svcvt<dst_t, src_t>(svbool_t pg,            \
                                                                                     vector_of_t<src_t> src) \
    {                                                                                                        \
        return svcvt_##short_dst_t##_x(pg, src);                                                             \
    }
DEFINE_SVCVT(bf16, __bf16, float);
#undef DEFINE_SVCVT

template <>
inline __attribute__((__always_inline__)) svfloat32_t svcvt<float, __bf16>(svbool_t pg, svbfloat16_t src)
{
    return svreinterpret_f32(svlsl_x(pg, svreinterpret_u32(src), (uint32_t)16));
}
} // namespace kutacc

#endif
