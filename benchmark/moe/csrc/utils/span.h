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

#include "check.h"

namespace utils {
template <typename T>
struct span {
    T *ptr;
    size_t size;

    span subspan(size_t offset, size_t count = size_t(-1u))
    {
        if (count == size_t(-1u)) {
            PARAMETER_CHECK(offset <= size, offset, " ", size);
            return {ptr + offset, size - offset};
        } else {
            PARAMETER_CHECK(offset + count <= size, offset, " ", count, " ", size);
            return {ptr + offset, count};
        }
    }
};

using u8span = span<uint8_t>;
}  // namespace utils
