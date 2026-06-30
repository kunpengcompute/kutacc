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

#include "kutacc.h"
#include "kernel/kernel.h"

namespace kutacc {
// thread_task
template <typename scalar_t, bool with_idx>
void gemm_pack_thread_task(
    int64_t thread_id, int64_t m, int64_t n, int64_t tm, int64_t tn,
    scalar_t *i_ptr, scalar_t *o_ptr, int *idx, int64_t ldi, int thread_nums);
}