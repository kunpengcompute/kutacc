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

#include "Tensor.h"

void flash_mla_get_metadata(const Tensor &seqlens_k,
                            const int64_t &num_heads_per_head_k,
                            const int64_t &num_heads_k,
                            Tensor &tile_scheduler_metadata,
                            Tensor &num_splits,
                            char *&mem_ptr);

void flash_mla_with_kvcache(Tensor q,                                   // batch_size x num_heads x head_size
                            const Tensor &kvcache,                      // num_blocks x page_block_size x head_size
                            const Tensor &block_table,                  // batch_size x max_num_blocks_per_seq
                            const Tensor &seqlens_k,                    // batch_size
                            const int &head_size_v,
                            const Tensor &tile_scheduler_metadata,      // num_thread_parts x TileSchedulerMetaDataSize
                            const Tensor &num_splits,                   // batch_size + 1
                            const double &softmax_scale,
                            bool is_causal,
                            Tensor &out);                               // batch_size x num_heads x head_size_v

void naive_mla_with_kvcache(const Tensor &q,                            // batch_size x num_heads x head_size
                            const Tensor &kvcache,                      // num_blocks x page_block_size x head_size
                            const Tensor &block_table,                  // batch_size x max_num_blocks_per_seq
                            const Tensor &seqlens_k,                    // batch_size
                            const int &head_size_v,
                            const double &softmax_scale,
                            Tensor &out);                               // batch_size x num_heads x head_size_v