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

#ifndef KUTACC_SOFTMAX_H
#define KUTACC_SOFTMAX_H

#include "../math/fast_exp.h"
#include "../wrapper/wrapper.h"
#include <cmath>
#include <cstdlib>

namespace kutacc {
template <typename scalar_t>
inline void softmax_with_max(float *buf, scalar_t *out, int64_t size, float max)
{
    auto vl = svcntw();
    // sub max & exp & reduce sum
    svfloat32_t reduce = svdup_f32(0.f);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1_f32(pg, &buf[i]);
        values = kutacc::fast_exp(pg, svsub_x(pg, values, max));
        reduce = svadd_m(pg, reduce, values);
        svst1_f32(pg, &buf[i], values);
    }
    // mul sum_inv & convert to scalar_t
    float sum_inv = 1 / svaddv(svptrue_b32(), reduce);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1_f32(pg, &buf[i]);
        values = svmul_x(pg, values, sum_inv);
        kutacc::svst1<scalar_t, float>(pg, &out[i], values);
    }
}

template <typename scalar_t>
inline void softmax(scalar_t *data, int64_t size, scalar_t *out, float *buf)
{
    auto vl = svcntw();
    // convert to float && reduce max
    svfloat32_t reduce = svdup_f32(-INFINITY);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = kutacc::svld1<float>(pg, &data[i]);
        reduce = svmax_m(pg, reduce, values);
        svst1_f32(pg, &buf[i], values);
    }
    // softmax with max
    softmax_with_max(buf, out, size, svmaxv(svptrue_b32(), reduce));
}
}   // namespace kutacc

#endif
