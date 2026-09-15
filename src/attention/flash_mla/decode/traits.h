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

#include <cassert>

namespace kutacc {

namespace flash_mla {

template <int q_block_size_, int kv_block_size_, int head_dim_, int head_dim_v_, bool is_causal_, bool is_kv_packed_>
struct KernelTraitsTemplate {
    static constexpr auto q_block_size = q_block_size_;
    static constexpr auto kv_block_size = kv_block_size_;
    static constexpr auto head_dim = head_dim_;
    static constexpr auto head_dim_v = head_dim_v_;
    static constexpr auto is_causal = is_causal_;
    static constexpr auto is_kv_packed = is_kv_packed_;

    static_assert(head_dim_v <= head_dim);

    static_assert(!is_kv_packed || (is_kv_packed && head_dim % 32 == 0 && head_dim_v % 32 == 0));
};

} // namespace flash_mla

} // namespace kutacc