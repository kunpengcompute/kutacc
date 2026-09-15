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
#include "internal.h"
#include "kutacc.h"

namespace kutacc {

moe_config_t cfg;

void moe_config_init(bool is_prefill, bool use_async_disp, bool use_async_comb, bool use_static_route)
{
    cfg.is_prefill = getenv_int64_with_default("IS_PREFILL", is_prefill);
    cfg.use_async_disp = getenv_int64_with_default("ENABLE_ASYNC_DISPATCH", use_async_disp);
    cfg.use_async_comb = getenv_int64_with_default("ENABLE_ASYNC_COMBINE", use_async_comb);
    cfg.use_static_route = getenv_int64_with_default("ENABLE_STATIC_ROUTING", use_static_route);
}

void cal_task_range(int tid, int num_threads, int num_tasks, int *start, int *end)
{
    if (num_tasks == 0) {
        *start = 0;
        *end = 0;
        return;
    }
    int base = num_tasks / num_threads;
    int remainder = num_tasks % num_threads;
    if (tid < remainder) {
        *start = tid * (base + 1);
        *end = *start + base + 1;
    } else {
        *start = remainder * (base + 1) + (tid - remainder) * base;
        *end = *start + base;
    }
}

void task_partition(std::vector<bool>& target_ranks, std::vector<std::vector<int>>& idxs, int parallel_size)
{
    int data_size = target_ranks.size();
    int sum = std::count(target_ranks.begin(), target_ranks.end(), true);
    if (parallel_size <= 0 || data_size <= 0 || sum <= 0) {
        for (int i = 0; i < thread_max_num; ++i) {
            idxs[i][0] = idxs[i][1] = 0;
        }
        return;
    }
    int target_load = sum / parallel_size;
    int remainder = sum % parallel_size;
    int current_start = 0;
    int current_sum = 0;
    int tid = 0;
    for (int i = 0; i < data_size && tid < parallel_size; ++i) {
        current_sum += target_ranks[i] ? 1:0;
        int current_target = target_load + (tid < remainder ? 1 : 0);
        if ((current_sum >= current_target && tid < parallel_size - 1) ||
            (tid == parallel_size - 1 && i == data_size - 1)) {
            idxs[tid][0] = current_start;
            idxs[tid][1] = i + 1;
            current_start = i + 1;
            current_sum = 0;
            tid++;
        }
    }
    for (; tid < parallel_size; ++tid) {
        idxs[tid][0] = data_size;
        idxs[tid][1] = data_size;
    }
}

}