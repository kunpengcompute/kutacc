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

#include "kutacc.h"
#include "tensor/tensor.h"
#include <memory>
#include "utils/parallel.h"
#include "linear/mm.h"
#include "utils/collapse.h"

namespace kutacc {
void transition_kernel(Tensor &input_act, const Tensor &linear1_w, Tensor &linear1_b,
        Tensor &linear2_w, Tensor &linear2_b, Tensor &intermediate_act, Tensor &out, 
        int64_t batch, int64_t n_res, int64_t c_o, int64_t c_i)
{
    KUTACC_CHECK(batch > 0 && n_res > 0  && c_o > 0 && c_i > 0 &&
        batch <= INT64_MAX / n_res && c_o <= INT32_MAX && c_i <= INT32_MAX,
        "input param <= 0 or overflow");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    addmm(
        kutacc::to_bf16(1),
        Tensor(input_act.data_ptr(), {batch * n_res, c_o}, {c_o, 1}, 2, kBF16),
        Tensor(linear1_w.data_ptr(), {c_o, c_i}, {1, c_o}, 2, kBF16),
        kutacc::to_bf16(0),
        Tensor(intermediate_act.data_ptr(), {batch * n_res, c_i}, {c_i, 1}, 2, kBF16),
        kutacc::BlasExtendParams{.prepack_b = true,
            .row_bias = true,
            .bias = linear1_b.data_ptr(),
            .relu = true});
    addmm(
        kutacc::to_bf16(1), 
        Tensor(intermediate_act.data_ptr(), {batch * n_res, c_i}, {c_i, 1}, 2, kBF16),
        Tensor(linear2_w.data_ptr(), {c_i, c_o}, {1, c_i}, 2, kBF16),
        kutacc::to_bf16(0),
        Tensor(out.data_ptr(), {batch * n_res, c_o}, {c_o, 1}, 2, kBF16),
        kutacc::BlasExtendParams{.prepack_b = true,
            .row_bias = true,
            .bias = linear2_b.data_ptr()}
    );
}
}

kutacc_export void kutacc_af2_transition(kutacc_af2_trans_act_inputs_t *trans_inputs_ptr, kutacc_af2_trans_weights_t *trans_weights_ptr, kutacc_tensor_h out)
{
    KUTACC_CHECK(trans_inputs_ptr != nullptr && trans_weights_ptr != nullptr && out != nullptr, "af2_transition: input args nullptr error\n");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }

    kutacc_tensor_h input_act = trans_inputs_ptr->input_act;
    kutacc_tensor_h intermediate_act = trans_inputs_ptr->intermediate_act;
    kutacc_tensor_h linear1_w = trans_weights_ptr->linear1_w;
    kutacc_tensor_h linear1_b = trans_weights_ptr->linear1_b;
    kutacc_tensor_h linear2_w = trans_weights_ptr->linear2_w;
    kutacc_tensor_h linear2_b = trans_weights_ptr->linear2_b;

    int64_t batch = trans_inputs_ptr->batch;
    int64_t n_res = trans_inputs_ptr->n_res;
    int64_t c_o = trans_weights_ptr->c_o;
    int64_t c_i = trans_weights_ptr->c_i;

    kutacc::transition_kernel(*kutacc::convertKutaccTensor(input_act), *kutacc::convertKutaccTensor(linear1_w), *kutacc::convertKutaccTensor(linear1_b), 
        *kutacc::convertKutaccTensor(linear2_w), *kutacc::convertKutaccTensor(linear2_b), *kutacc::convertKutaccTensor(intermediate_act), *kutacc::convertKutaccTensor(out), 
        batch, n_res, c_o, c_i);
}