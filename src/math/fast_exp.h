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

#ifndef KUTACC_FAST_EXP_H
#define KUTACC_FAST_EXP_H

#include <arm_sve.h>

namespace kutacc {
#define USE_HARDWARE_EXP
#ifdef USE_HARDWARE_EXP
inline svfloat32_t fast_exp(svbool_t pg, svfloat32_t values)
{
    constexpr float exp_const = 92.33248f;  // 64 / ln(2)
    constexpr float ln_flt_max = 88.72284f; // ln_flt_max = ln(FLT_MAX) = ln(2 ** 128)
    return svexpa(svcvt_u32_x(pg,
        svrinta_x(pg, svmad_x(pg, svmin_x(pg, values, ln_flt_max), svdup_f32(exp_const), (float)(127 << 6)))));
}

#else
inline svfloat32_t fast_exp(svbool_t pg, svfloat32_t values)
{
    const float factorial_1 = 0.999999701f;
    const float factorial_2 = 0.499991506f;
    const float factorial_3 = 0.166676521f;                                 // 1 / factorial(3)
    const float factorial_4 = 0.0418978221f;                                // 1 / factorial(4)
    const svfloat32_t vec_factorial_5 = svdup_f32(0.00828929059f);          // 1 / factorial(5)
    const svfloat32_t vec_exp_log2ef = svdup_f32(1.4426951f);               //log2(e)
    const svfloat32_t vec_zero = svdup_f32(0.f);
    const svfloat32_t ln2f = svdup_f32(0.6931472f);                         // ln(2)
    const float vec_ln_flt_min = -87.33655;
    const float vec_ln_flt_max = 88.72284;
    const int exp_offset = 0x7f;
    const int n_mantissa_bits = 23;

    svbool_t less_ln_flt_min_mask = svcmplt(pg, values, vec_ln_flt_min);
    svfloat32_t vec_src = svmin_x(pg, values, vec_ln_flt_max);

    svfloat32_t vec_fx = svmad_x(pg, vec_src, vec_exp_log2ef, 0.5f);
    vec_fx = svrintm_x(pg, vec_fx);

    // x = x - fx * ln2
    svfloat32_t vec_exp_poly = svmsb_x(pg, vec_fx, ln2f, vec_src);

    // compute polynomial
    svfloat32_t vec_res = svmad_x(pg, vec_exp_poly, vec_factorial_5, factorial_4);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, factorial_3);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, factorial_2);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, factorial_1);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, 1);

    // compute 2 ^ (n - 1)
    svfloat32_t vec_exp_number = svsub_x(pg, vec_fx, 1);
    svint32_t vec_exp_number_i = svcvt_s32_x(pg, svrintn_x(pg, vec_exp_number));
    svint32_t vec_two_pow_n_i = svadd_x(pg, vec_exp_number_i, exp_offset);
    vec_two_pow_n_i = svlsl_x(pg, vec_two_pow_n_i, n_mantissa_bits);
    svfloat32_t vec_two_pow_n = svreinterpret_f32(vec_two_pow_n_i);

    // y = y * 2 ^ n
    vec_res = svmul_x(pg, vec_res, vec_two_pow_n);
    vec_res = svmul_x(pg, vec_res, 2);
    vec_res = svsel(less_ln_flt_min_mask, vec_zero, vec_res);
    return vec_res;
}

inline svfloat16_t fast_exp(svbool_t pg, svfloat16_t values)
{
    const __fp16 factorial_1 = 0.999999701f;
    const __fp16 factorial_2 = 0.499991506f;
    const __fp16 factorial_3 = 0.166676521f;                                 // 1 / factorial(3)
    const __fp16 factorial_4 = 0.0418978221f;                                // 1 / factorial(4)
    const svfloat16_t vec_factorial_5 = svdup_f16(0.00828929059f);          // 1 / factorial(5)
    const svfloat16_t vec_exp_log2ef = svdup_f16(1.4426951f);               //log2(e)
    const svfloat16_t vec_zero = svdup_f16(0.f);
    const svfloat16_t vec_ln2f = svdup_f16(0.6931472f);                         // ln(2)
    const __fp16 vec_ln_flt_min = -10.f;
    const __fp16 vec_ln_flt_max = 10.f;
    const int exp_offset = 0x4f;
    const int n_mantissa_bits = 10;

    // exp(n + r) = 2 ^ n * exp(r)
    svbool_t less_ln_flt_min_mask = svcmplt(pg, values, vec_ln_flt_min);
    svfloat16_t vec_src = svmin_x(pg, values, vec_ln_flt_max);

    svfloat16_t vec_fx = svmad_x(pg, vec_src, vec_exp_log2ef, 0.5f);
    vec_fx = svrintm_x(pg, vec_fx);

    // x = x - fx * ln2
    svfloat16_t vec_exp_poly = svmsb_x(pg, vec_fx, vec_ln2f, vec_src);

    // compute polynomial
    svfloat16_t vec_res = svmad_x(pg, vec_exp_poly, vec_factorial_5, factorial_4);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, factorial_3);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, factorial_2);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, factorial_1);
    vec_res = svmad_x(pg, vec_exp_poly, vec_res, 1);

    // compute 2 ^ (n - 1)
    svfloat16_t vec_exp_number = svsub_x(pg, vec_fx, 1);
    svint16_t vec_exp_number_i = svcvt_s16_x(pg, svrintn_x(pg, vec_exp_number));
    svint16_t vec_two_pow_n_i = svadd_x(pg, vec_exp_number_i, exp_offset);
    vec_two_pow_n_i = svlsl_x(pg, vec_two_pow_n_i, n_mantissa_bits);
    svfloat16_t vec_two_pow_n = svreinterpret_f16(vec_two_pow_n_i);

    // y = y * 2 ^ n
    vec_res = svmul_x(pg, vec_res, vec_two_pow_n);
    vec_res = svmul_x(pg, vec_res, 2);
    vec_res = svsel(less_ln_flt_min_mask, vec_zero, vec_res);
    return vec_res;
}
#endif
}   // namespace kutacc
#endif
