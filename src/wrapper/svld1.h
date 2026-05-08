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

#ifndef KUTACC_SVLD1_H
#define KUTACC_SVLD1_H

#include "vector_of.h"
#include <type_traits>

namespace kutacc {
template <typename dst_t, typename src_t>
inline vector_of_t<dst_t> svld1(svbool_t pg, const src_t *data)
{
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return ::svld1(pg, data);
    } else if constexpr (sizeof(dst_t) == SIZEOF_FLOAT32 && sizeof(src_t) == SIZEOF_BF16) {
        auto pg_src = svuzp1_b16(pg, svpfalse());
        auto values = ::svld1(pg_src, data);
        values = svzip1(values, values);
        return kutacc::svcvt<dst_t, src_t>(pg, values);
    } else {
        dst_t::not_implemented();
    }
}
} // namespace kutacc

#endif
