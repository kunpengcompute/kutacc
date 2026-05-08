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

#ifndef KUTACC_TENSOR_H
#define KUTACC_TENSOR_H

#include <vector>
#include "kutacc.h"
#include "utils/check.h"

namespace kutacc {
struct SimpleTensor {
    int64_t dim;
    void *data_ptr;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    DType dtype;

    SimpleTensor(void *data_ptr, std::vector<int64_t> &&sizes, std::vector<int64_t> &&strides, int64_t dim,
                 ::kutacc::DType dtype)
        : dim(dim), data_ptr(data_ptr), sizes(sizes), strides(strides), dtype(dtype)
    {
        KUTACC_CHECK(dim == (int64_t)sizes.size(), dim, " ", sizes.size());
        KUTACC_CHECK(dim == (int64_t)strides.size(), dim, " ", strides.size());
    }
};

struct Tensor {
    SimpleTensor data;

    Tensor(void *data_ptr, std::vector<int64_t> &&sizes, std::vector<int64_t> &&strides, int64_t dim,
           ::kutacc::DType dtype)
        : data(SimpleTensor(data_ptr, std::move(sizes), std::move(strides), dim, dtype)) {};
    void *data_ptr() const;
    const std::vector<int64_t> &sizes() const;
    const std::vector<int64_t> &strides() const;
    std::vector<int64_t> &sizes_ref();
    std::vector<int64_t> &strides_ref();
    DType dtype() const;
    int64_t dim() const;
    int64_t numel() const;
};

Tensor *convertKutaccTensor(void *tensor_);
} // namespace kutacc
#endif
