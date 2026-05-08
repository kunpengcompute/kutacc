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

#ifndef KUTACC_RELU_H
#define KUTACC_RELU_H

#include <algorithm>
#include <type_traits>

#include "../wrapper/wrapper.h"

namespace kutacc {
class ReLU {
public:
    /**
         * @brief Template function `call` can be instantiated as follows
         * 1. svfloat32_t call(svbool_t, svfloat32_t)
         * 2. svfloat16_t call(svbool_t, svfloat16_t)
         */
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, svfloat32_t> || std::is_same_v<T, svfloat16_t>, T> call(svbool_t pg, T x)
    {
        return svmax_x(pg, x, 0);
    }

    /**
         * @brief Template function `call` can be instantiated as follows
         * 1. float call(float)
         * 2. __fp16 call(__fp16)
         */
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, __fp16>, T> call(T x)
    {
        return std::max(x, (T)0);
    }
}
} // namespace kutacc

#endif
