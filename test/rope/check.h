/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

struct parameter_error : std::exception {
    std::string message;

    parameter_error(std::string message) : message(std::move(message)) {}

    const char* what() const noexcept { return message.c_str(); }
};

#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)
#define SOURCE_LOCATION __FILE__ ":" TO_STR(__LINE__)

namespace internal {
template <typename... Args>
inline void parameter_check(bool condition, std::string source_location, Args&&... args)
{
    if (__builtin_expect(!(condition), 0)) {
        std::stringstream stream;
        stream << "PARAMETER_CHECK fail at " << source_location;
        if constexpr (sizeof...(args)) {
            stream << ", ";
            (stream << ... << std::forward<Args>(args));
        }
        throw parameter_error{stream.str()};
    }
}

}  // namespace internal

#ifdef ENABLE_CHECK
#define PARAMETER_CHECK(condition, ...) (::internal::parameter_check((condition), SOURCE_LOCATION, ##__VA_ARGS__))
#else
#define PARAMETER_CHECK(condition, ...) (void(condition))
#endif
