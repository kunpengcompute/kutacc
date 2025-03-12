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

#ifndef KUTACC_SILU_H
#define KUTACC_SILU_H

#include <cmath>
#include <type_traits>

#include "../math/fast_exp.h"
#include "../wrapper/wrapper.h"

namespace kutacc {
    class SiLU {
    public:
        /**
         * @brief Template function `call` can be instantiated as follows
         * 1. svfloat32_t call(svbool_t, svfloat32_t)
         * 2. svfloat16_t call(svbool_t, svfloat16_t)
         */
        template <typename T>
        static std::enable_if_t<std::is_same_v<T, svfloat32_t> || std::is_same_v<T, svfloat16_t>, T> call(svbool_t pg, T x)
        {
            auto exp_neg = armdnn::fast_exp(pg, svneg_x(pg, x));
            return svdiv_x(pg, x, svadd_x(pg, exp_neg, 1));
        }

        /**
         * @brief Template function `call` can be instantiated as follows
         * 1. float cll(float)
         * 2. __fp16 call(__fp16)
         */
        template <typename T>
        static std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, __fp16>, T> call(T x)
        {
            auto exp_neg = std::exp(-x);
            return 1 / (exp_neg + 1);
        }
    }
}   // namespace kutacc

#endif