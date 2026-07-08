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

#include <string>
#include <cstdlib>
#include <cassert>

namespace kutacc {

inline int64_t getenv_int64(std::string name)
{
    auto str = std::getenv(name.c_str());
    assert(str != nullptr && ("cannot find environment variable: " + name).c_str());
    return std::atol(str);
}

inline int64_t getenv_int64_with_default(std::string name, int64_t default_value)
{
    auto str = std::getenv(name.c_str());
    if (str == nullptr) {
        return default_value;
    }
    return std::atol(str);
}

}