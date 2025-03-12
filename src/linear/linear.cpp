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

#include <vector>
#include "kutacc.h"
#include "tensor/tensor.h"
#include "linear/mm.h"
#include "utils/check.h"
#include "utils/collapse.h"

namespace kutacc{
void linear_kernel(Tensor &act, Tensor &weight, float* bias_data, Tensor& result, int64_t beta)
{
    int64_t m = act.numel() / act.sizes().back();
    int64_t n = weight.numel() / weight.sizes().back();
    int64_t k = act.sizes().back();
    bool bias_flag = bias_data != nullptr ? true : false;
    if (bias_flag) {
        addmm(to_bf16(1),
            Tensor((__bf16 *)act.data_ptr(), {m, k}, {act.strides()[(unsigned long)act.dim() - 2], 1}, 2, kBF16),
            Tensor((__bf16 *)weight.data_ptr(), {k, n}, {1, weight.strides()[(unsigned long)weight.dim() - 2]}, 2, kBF16),
            to_bf16((float)beta),
            Tensor((__bf16 *)result.data_ptr(), {m, n}, {n, 1}, 2, kBF16), 
            BlasExtendParams{.row_bias = bias_flag, .bias = bias_data});
    } else {
        addmm(to_bf16(1),
            Tensor((__bf16 *)act.data_ptr(), {m, k}, {act.strides()[(unsigned long)act.dim() - 2], 1}, 2, kBF16),
            Tensor((__bf16 *)weight.data_ptr(), {k, n}, {1, weight.strides()[(unsigned long)weight.dim() - 2]}, 2, kBF16),
            to_bf16((float)beta),
            Tensor((__bf16 *)result.data_ptr(), {m, n}, {n, 1}, 2, kBF16), 
            BlasExtendParams{.row_bias = bias_flag, .bias = bias_data});
    }
}
}

void kutacc_af2_linear(kutacc_tensor_h act, kutacc_tensor_h weight, float* bias_data, kutacc_tensor_h result, int64_t beta)
{
    kutacc::linear_kernel(*kutacc::convertKutaccTensor(act), *kutacc::convertKutaccTensor(weight), bias_data, 
        *kutacc::convertKutaccTensor(result), beta);
}