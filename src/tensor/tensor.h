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
#include "kutacc.h"
#include "../utils/check.h"

namespace kutacc {

template <typename dtype, int64_t dim>
inline void swap_dataptr(Tensor<dtype, dim>& a, Tensor<dtype, dim>& b)
{
    dtype* ptr_a = a.data_ptr();
    dtype* ptr_b = b.data_ptr();
    a.set_data_ptr(ptr_b);
    b.set_data_ptr(ptr_a);
}

template <typename dtype, int64_t dim, int64_t new_dim = dim - 1>
inline Tensor<dtype, new_dim> tensor_select(const Tensor<dtype, dim>& t, int64_t axis, int64_t index)
{
    KUTACC_CHECK(axis >= 0 && axis < dim, "invalid select axis: ", axis, " dim: ", dim);
    KUTACC_CHECK(index >= 0 && index < t.size(axis), "invalid select index: ", index, " axis size: ", t.size(axis));

    int64_t data_offset = index * t.stride(axis);
    int64_t new_sizes[new_dim];
    int64_t new_strides[new_dim];

    int64_t pos = 0;
    for (int64_t i = 0; i < dim; ++i) {
        if (i == axis) continue;
        new_sizes[pos] = t.size(i);
        new_strides[pos] = t.stride(i);
        pos++;
    }
    return Tensor<dtype, new_dim>(t.data_ptr() + data_offset, new_sizes, new_strides);
}

template <typename dtype, int64_t dim>
inline Tensor<dtype, dim> tensor_slice(const Tensor<dtype, dim>& t, int64_t axis, int64_t start, int64_t end)
{
    KUTACC_CHECK(axis >= 0 && axis < dim, "invalid slice axis: ", axis, " dim: ", dim);
    KUTACC_CHECK(start >= 0 && end <= t.size(axis) && start <= end, "invalid slice range: start=", start, " end=", end,
        " axis size=", t.size(axis));

    int64_t data_offset = start * t.stride(axis);
    int64_t new_sizes[dim];
    int64_t new_strides[dim];

    for (int64_t i = 0; i < dim; ++i) {
        new_sizes[i] = t.size(i);
        new_strides[i] = t.stride(i);
    }
    new_sizes[axis] = end - start;

    return Tensor<dtype, dim>(t.data_ptr() + data_offset, new_sizes, new_strides);
}

inline void tensor_get_view_strides(int64_t dim, const int64_t *sizes, const int64_t *strides,
    int64_t view_dim, const int64_t *view_sizes, int64_t *view_strides)
{
    int64_t view_d = view_dim - 1;
    int64_t r = dim - 1;
    while (r >= 0 && sizes[r] == 1) {
        --r;
    }
    while (r >= 0) {
        int64_t l = r;
        int64_t cur_stride = sizes[r] * strides[r];
        while (l > 0 && (sizes[l - 1] == 1 || strides[l - 1] == cur_stride)) {
            cur_stride *= sizes[--l];
        }
        int64_t cur_view_stride = strides[r];
        while (view_d >= 0 && cur_view_stride * view_sizes[view_d] <= cur_stride) {
            view_strides[view_d] = cur_view_stride;
            cur_view_stride *= view_sizes[view_d--];
        }
        KUTACC_CHECK(cur_view_stride == cur_stride, cur_view_stride, " ", cur_stride);
        r = l - 1;
    }
    KUTACC_CHECK(view_d == -1, view_d);
}

template <typename dtype, int64_t dim, int64_t view_dim>
inline Tensor<dtype, view_dim> tensor_view(const Tensor<dtype, dim> &t, const int64_t *view_sizes)
{
    int64_t view_strides[view_dim];
    tensor_get_view_strides(dim, t.sizes().data(), t.strides().data(), view_dim, view_sizes, view_strides);
    return Tensor<dtype, view_dim>(t.data_ptr(), view_sizes, view_strides);
}

template <typename dtype, int64_t dim, int64_t view_dim>
inline Tensor<dtype, view_dim> tensor_view(const Tensor<dtype, dim> &t, const std::array<int64_t, view_dim> &view_sizes)
{
    return tensor_view<dtype, dim, view_dim>(t, view_sizes.data());
}

} // namespace kutacc