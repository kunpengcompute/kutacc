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

#include "tensor.h"
#include <vector>
#include "kutacc.h"
#include "utils/check.h"
#include <iostream>

namespace kutacc {
TensorWrapper::TensorWrapper(void *data_ptr, std::vector<int64_t> &&sizes, std::vector<int64_t> &&strides, int64_t dim,
                             DType dtype)
{
    Tensor *t = new Tensor(data_ptr, std::move(sizes), std::move(strides), dim, dtype);
    tensor_ = t;
}

TensorWrapper::TensorWrapper()
{
    tensor_ = nullptr;
}

void *Tensor::data_ptr() const
{
    return data.data_ptr;
}

const std::vector<int64_t> &Tensor::sizes() const
{
    return data.sizes;
}

const std::vector<int64_t> &Tensor::strides() const
{
    return data.strides;
}

std::vector<int64_t> &Tensor::sizes_ref()
{
    return data.sizes;
}

std::vector<int64_t> &Tensor::strides_ref()
{
    return data.strides;
}

DType Tensor::dtype() const
{
    return data.dtype;
}

int64_t Tensor::dim() const
{
    return data.dim;
}

int64_t Tensor::numel() const
{
    if (data.dim == 0) {
        return 0;
    }
    int64_t res = 1;
    for (int64_t i = 0; i < data.dim; i++) {
        res *= data.sizes[(uint64_t)i];
    }
    return res;
}

// void Tensor::SetTensorStrides(std::vector<int64_t> &strides)
// { // 预留接口优化addmm装包解包过程
//     data.strides = strides;
// }

Tensor *convertKutaccTensor(void *tensor_)
{
    return (Tensor *)tensor_;
}

TensorWrapper::~TensorWrapper()
{
    delete static_cast<Tensor *>(tensor_);
}
} // namespace kutacc
