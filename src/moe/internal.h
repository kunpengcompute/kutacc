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

#include <cstdint>
#include <kupl.h>
#include <vector>
#include <utility>
#include <memory>
#include <sys/time.h>
#include "core/kurmcl/kurmcl_impl.h"
#include "utils/env_utils.h"
#include "kutacc.h"

#define ASY_CORE_OFFSET_IN_NUMA_DEFAULT 33
#define ASY_CORE_USED_CNT_DEFAULT 4
#define MAX_NUM_TOPK 12
#define MAX_THREAD_NUMS 32

namespace kutacc {
    inline int16_t dtp;
    inline int16_t *src_info;
    inline int16_t *catch_recv_src_info;
    inline int16_t *catch_recv_src_info_bak;
    inline bfloat16_t *tmpx_for_sum;
    inline int64_t num_local_experts;
    inline int64_t recv_src_info_size;
    inline int64_t recv_src_info_conut;
    inline int64_t recv_src_info_ep_bias;
    inline int64_t packed_recv_x_size;
    inline int64_t disbuf_size;

    inline int64_t max_num_tokens;
    inline int64_t num_tokens;
    inline int64_t num_topk;
    inline int local_rank;
    inline bfloat16_t *combinedx_base_ptr;
    inline uint8_t *combinedx_meta_ptr;
    inline std::vector<bfloat16_t *> combinedx_group_ptr;
    inline std::vector<uint8_t *> combinedx_meta_group_ptr;
    inline std::vector<int> will_recv_from_rank;
    inline int dis_peer_nums = 0;

    inline uint64_t dis_send_stage1 = 0;
    inline uint64_t dis_send_stage2 = 0;
    inline uint64_t dis_recv_stage1 = 0;
    inline uint64_t dis_start_time = 0;
    inline int dis_times = 0;
    inline uint64_t dis_thread_start_time = 0;
    inline uint64_t com_send_stage1 = 0;
    inline uint64_t com_send_stage2 = 0;
    inline uint64_t com_start_time = 0;
    inline uint64_t com_recv_stage1 = 0;
    inline uint64_t com_recv_stage2 = 0;
    inline uint64_t com_recv_stage3 = 0;
    inline int comb_times = 0;
    inline uint64_t com_thread_start_time = 0;
    inline constexpr int thread_max_num = 32;
    inline bool is_prefill_default = false;
    inline bool use_async_disp_default = false;
    inline bool use_async_comb_default = false;
    inline bool use_static_route_default = false;

    inline async_tg_h comm_thread_exec;
    inline int asy_thread_num = getenv_int64_with_default("ASYNC_COMM_CORE_USED_CNT", ASY_CORE_USED_CNT_DEFAULT);
    inline int asy_core_offset_in_numa =
        getenv_int64_with_default("ASYNC_COMM_CORE_OFFSET", ASY_CORE_OFFSET_IN_NUMA_DEFAULT);

    typedef struct moe_config {
        bool is_prefill;
        bool use_async_disp;
        bool use_async_comb;
        bool use_static_route;
    } *moe_config_h, moe_config_t;
    extern moe_config_t cfg;

    void moe_config_init(bool is_prefill, bool use_async_disp, bool use_async_comb, bool use_static_route);
    void task_partition(std::vector<bool>& target_ranks, std::vector<std::vector<int>>& idxs, int parallel_size);
    void cal_task_range(int tid, int num_threads, int num_tasks, int *start, int *end);
}