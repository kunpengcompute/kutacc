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

#ifndef KUTACC_CHECK_H
#define KUTACC_CHECK_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace kutacc {
extern bool kutacc_check_err_set;
namespace internal {
    inline void check_fail_print(std::stringstream &stream)
    {
        stream << std::endl;
    }

    template <typename Arg, typename... Rest>
    inline void check_fail_print(std::stringstream &stream, Arg &&arg, Rest &&...rest)
    {
        stream << std::forward<Arg>(arg);
        check_fail_print(stream, rest...);
    }

    template <typename... Args>
    inline void check_fail(std::string func, std::string file, int line, Args &&...args)
    {
        std::stringstream stream;
        stream << "KUTACC_CHECK fail in " << func << " at " << file << ":" << line << " , ";
        check_fail_print(stream, std::forward<Args>(args)...);
        stream << "\n";
        std::cerr << stream.str();
    }

}   // namespace internal
}   // namespace kutacc

#define KUTACC_CHECK(condition, ...)                                                            \
    do {                                                                                        \
        if (__builtin_expect(!(condition), 0)) {                                                \
            kutacc::internal::check_fail(__func__, __FILE__, __LINE__, __VA_ARGS__);            \
            kutacc::kutacc_check_err_set = true;                                                \
        }                                                                                       \
    } while (0)


#define KUTACC_CHECK_TENSOR_SHAPE(tensor, ...)                                                                          \
    KUTACC_CHECK((tensor).sizes() == c10::IntArrayRef({__VA_ARGS__}), "invalid tensor shape: ", (tensor).sizes(),       \
        ", expect: ", c10::IntArrayRef({__VA_ARGS__}))

#define KUTACC_CHECK_TENSORWRAPPER_SHAPE(tensor, ...)                                           \
    KUTACC_CHECK(c10::IntArrayRef((tensor).sizes) == c10::IntArrayRef({__VA_ARGS__}),           \
        "invalid tensor wrapper shape: ", c10::IntArrayRef((tensor).sizes),                     \
        ", expect: ", c10::IntArrayRef({__VA_ARGS__}))

#endif
