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
#include "kutacc.h"
#include "utils/check.h"
#include "utils/timer.h"

struct mt_timer_t {
    std::vector<double> thread[38];
    std::vector<double> master;

    mt_timer_t()
    {
        utils::kutacc_time_init();
    }

    ~mt_timer_t()
    {
        int n = master.size();
        double min_cost = INFINITY;
        double max_cost = 0;
        double avg_cost = 0;
        constexpr int drop_num = 100;
        for (int i = drop_num; i < n; ++i) {
            double mx = 0;
            for (int j = 0; j < kutacc::get_thread_num(); ++j) {
                mx = std::max(mx, thread[j][i]);
            }
            double cost = master[i] - mx;
            min_cost = std::min(min_cost, cost);
            max_cost = std::max(max_cost, cost);
            avg_cost += cost;
        }
        if (n > drop_num) {
            avg_cost /= n - drop_num;
            std::cout << n - drop_num << std::endl;
            std::cout << "min_cost = " << min_cost << " ns" << std::endl;
            std::cout << "max_cost = " << max_cost << " ns" << std::endl;
            std::cout << "avg_cost = " << avg_cost << " ns" << std::endl;
        }
    }
};