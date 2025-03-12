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

#ifndef KUTACC_LAYERNORM_KERNEL_H
#define KUTACC_LAYERNORM_KERNEL_H

#include "wrapper/wrapper.h"
#include <algorithm>
#include <cmath>

namespace kutacc {
template <typename scalar_t>
inline void af2_layernorm_kernel(scalar_t *data, float *gamma, float *beta, int64_t size, float eps, scalar_t *out)
{
    const unsigned long vl = svcntw();
    svfloat32_t vec_sum = svdup_f32(0);
    svfloat32_t vec_sum2 = svdup_f32(0);
    float delta;
    if constexpr (std::is_same_v<scalar_t, __bf16>) {
        delta = to_float(data[0]);
    } else {
        delta = data[0];
    }
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = kutacc::svld1<float, scalar_t>(pg, &data[i]);
        values = svsub_x(pg, values, delta);
        vec_sum = svadd_m(pg, vec_sum, values);
        vec_sum2 = svmla_m(pg, vec_sum2, values, values);
    }
    float sum = svaddv(svptrue_b32(), vec_sum);
    float sum2 = svaddv(svptrue_b32(), vec_sum2);
    float mean = sum / (float) size + delta;
    float var = std::max((sum2 - sum * sum / (float) size) / (float) size, 0.f);
    float rstd = 1 / std::sqrt(var + eps);
    svfloat32_t bias = svdup_f32(-rstd * mean);
    for ( int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = kutacc::svld1<float, scalar_t>(pg, &data[i]);
        values = svmla_x(pg, bias, values, rstd);
        values = svmad_x(pg, values, svld1<float, float>(pg, & gamma[i]), svld1<float, float>(pg, &beta[i]));
        kutacc::svst1<scalar_t, float>(pg, &out[i], values);
    }
}
}

#endif