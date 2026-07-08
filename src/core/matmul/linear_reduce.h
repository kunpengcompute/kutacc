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
// unroll dispatcher
void reduce_for_kblocks(int64_t m, int64_t n, int64_t blocks_in_k,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t start, int64_t end);

// filter
void reduce_filter(int64_t m, int64_t n, int64_t k, int64_t tile_k,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t start, int64_t end);
void reduce_stride_filter(int64_t m, int64_t n, int64_t k, int64_t tile_k,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t start, int64_t end,
    int64_t n_start, int64_t n_length);

}