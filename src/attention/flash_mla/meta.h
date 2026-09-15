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

namespace kutacc {

namespace flash_mla {

struct TileSchedMeta {
    int64_t begin_req_idx;
    int64_t begin_token_idx;
    int64_t end_req_idx;
    int64_t end_token_idx;
    int64_t begin_split_idx;
    int64_t _padding[3];
};

static_assert(sizeof(TileSchedMeta) == sizeof(int64_t) * 8);

struct KernelMeta {
    bool is_sparse;
    bool is_kv_packed;
    int64_t q_block_size;
    int64_t kv_block_size;
    int64_t head_dim;
    int64_t head_dim_v;
    int64_t fixed_overhead_num_blocks;
    int64_t bytes_per_thread;
};

} // namespace flash_mla

struct FlashMLAMeta {
    flash_mla::TileSchedMeta *tile_sched_meta;
    int *num_splits;

    int64_t batch_size;
    int64_t max_batch_size;
    int64_t num_thread_parts;
    int64_t threads_buffer_bytes;
    int64_t softmax_lseaccum_bytes;
    int64_t oaccum_bytes;

    flash_mla::KernelMeta kernel_meta;
};

typedef struct FlashMLAMeta *FlashMLAMetaHandle;

} // namespace kutacc