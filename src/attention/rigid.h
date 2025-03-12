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

#ifndef KUTACC_RIGID_H
#define KUTACC_RIGID_H

#include "utils/check.h"
#include "wrapper/wrapper.h"

namespace kutacc{
template <typename scalar_t>
inline void rigid_rot_vec_mul_kernel(scalar_t *pt, const float *rot_mat, scalar_t *out, const float *trans = nullptr,
    bool invert = false)
{
    float output[3];
    float input[3];
    {
        auto pg = svwhilelt_b32(0, 3);
        auto pt_values = svld1<float, scalar_t>(pg, pt);
        svst1<float, float>(pg, input, pt_values);
    }
    if (!invert) {
        for (int64_t i = 0; i < 3; i++) {
            float sum = 0;
            for (int64_t j = 0; j < 3; j++) {
                sum += rot_mat[i * 3 + j] * input[j];
            }
            output[i] = sum + (trans != nullptr ? trans[i] : 0);
        }
    } else {
        for (int64_t i = 0; i < 3; i++) {
            float sum = 0;
            for (int64_t j = 0; j < 3; j++) {
                sum += rot_mat[j * 3 + i] * (input[j] - (trans != nullptr ? trans[j] : 0));
            }
            output[i] = sum;
        }
    }
    {
        auto pg = svwhilelt_b32(0, 3);
        auto pt_values = svld1<float, float>(pg, output);
        svst1<scalar_t, float>(pg, out, pt_values);
    }
}

inline void rigid_rot_matmul_kernel(const float *a, const float *b, float *out)
{
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 3; j++) {
            float sum = 0;
            for (int64_t k = 0; k < 3; k++) {
                sum += a[i * 3 + k] * b[k * 3 + j];
            }
            out[i * 3 + j] = sum;
        }
    }
}
}

#endif