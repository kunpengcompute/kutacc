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
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_neon.h>
#include <algorithm>

#include "kernel/kernel.h"
#include "linear_utils.h"
#include "linear_reduce.h"
#include "linear_pack.h"
#include "kutacc.h"
namespace kutacc {
// thread_task
void s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(
    int64_t thread_id, int64_t m, int64_t n, int64_t k, MatrixTilingBlock t,
    int8_t *act_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t num_threads);

void s8_packed_gemm_bf16_dq_dynamic_n_slice_thread_task(
    int64_t thread_id, int64_t m, int64_t n, int64_t k, MatrixTilingBlock t,
    int64_t n_begin, int64_t n_length,
    int8_t *act_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t num_threads);

// interface
void s8_packed_gemm_bf16_dq_dynamic_tile_n(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t,
    int8_t *act_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc);
}