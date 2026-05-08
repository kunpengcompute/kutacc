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

#ifndef KUTACC_VECTOR_OF_H
#define KUTACC_VECTOR_OF_H

#include <arm_sve.h>

#define SIZEOF_FLOAT32 4
#define SIZEOF_BF16 2

namespace kutacc {
template <typename scalar_t>
using vector_of_t = decltype(svld1(svbool_t(), static_cast<const scalar_t *>(nullptr)));
} // namespace kutacc

#endif
