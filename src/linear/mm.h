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

#ifndef KUTACC_MM_H
#define KUTACC_MM_H

#include "kutacc.h"
#include "tensor/tensor.h"
#include <algorithm>
#include <cstdint>
#include <utility>

namespace kutacc {
void addmm(__bf16 alpha, const kutacc::Tensor &a, const kutacc::Tensor &b, __bf16 beta, const kutacc::Tensor &c,
           kutacc::BlasExtendParams param = {});
}

#endif
