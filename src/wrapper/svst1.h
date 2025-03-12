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

#ifndef KUTACC_SVST1_H
#define KUTACC_SVST1_H

#include "vector_of.h"
#include <type_traits>

namespace kutacc {
template <typename dst_t, typename src_t>
inline void svst1(svbool_t pg, dst_t *dst, vector_of_t<src_t> src)
{
    if constexpr (std::is_same_v<dst_t, src_t>) {
        ::svst1(pg, dst, src);
    } else if constexpr (sizeof(dst_t) == SIZEOF_BF16 && sizeof(src_t) == SIZEOF_FLOAT32) {
        auto pg_dst = svuzp1_b16(pg, svpfalse());
        auto values = svcvt<dst_t, src_t>(pg, src);
        values = svuzp1(values, values);
        ::svst1(pg_dst, dst, values);
    } else {
        dst_t::not_implemented();
    }
}
}   // namespace kutacc

#endif
