/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once
#include <cmath>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
namespace kutacc {
inline svfloat32_t fast_exp2(svbool_t pg, svfloat32_t z0)
{
    svfloat32_t z1 = svreinterpret_f32(svdup_u32(1212161984));  // 196735
    svfloat32_t z2 = svreinterpret_f32(svdup_u32(1060205250));  // 0.693157315
    svfloat32_t z3 = svreinterpret_f32(svdup_u32(1047920148));  // 0.240227044
    svfloat32_t z5 = svreinterpret_f32(svdup_u32(1123811328));  // 126
    auto z4 = z1 + z0;
    z1 = z4 - z1;
    z1 = z0 - z1;
    z2 = svmla_x(pg, z2, z1, z3);
    z3 = svreinterpret_f32(svdup_u32(1065353216));  // 1
    z1 = svmad_x(pg, z1, z2, z3);
    z2 = svexpa(svreinterpret_u32(z4));
    z1 = svmul_x(pg, z2, z1);
    auto poverflow = svacge(pg, z0, z5);
    if (__builtin_expect(svptest_any(pg, poverflow), 0)) {
        svbool_t pgt = svcmpgt(pg, z0, z5);
        z1 = svsel(pgt, svdup_f32(INFINITY), z1);
        svbool_t plt = svcmplt(pg, z0, svneg_x(pg, z5));
        z1 = svsel(plt, svdup_f32(0), z1);
    }
    return z1;
}

inline svfloat32_t fast_exp(svbool_t pg, svfloat32_t z0)
{
    return fast_exp2(pg, svmul_x(pg, z0, svdup_f32(1.442695041)));
}
}   // namespace kutacc
#endif