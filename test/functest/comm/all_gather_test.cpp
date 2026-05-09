/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuTACC is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <iostream>
#include <vector>
#include <mpi.h>
#include <gtest/gtest.h>
#include "kutacc.h"

TEST(AllGatherTest, all_gather_func)
{
    int world_size = 2, rank = 0;
    int m = 4, n = 5, len = 1;
    int64_t dim = 3, buffer_size = m * n * len;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    kutacc_comm_init(world_size, rank, buffer_size);

    std::vector<uint8_t> expect_data(world_size * m * n * len);
    uint8_t val = 0;
    for (int i = 0; i < world_size * m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i * n * len + j * len == m * n * len) val = 32;
            expect_data[i * n * len + j * len] = val++;
        }
    }

    std::vector<int64_t> sizes = {m, n, len};
    std::vector<int64_t> strides = {n * len, len, 1};
    std::vector<uint8_t> data(m * n * len);

    if (rank == 0) {
        val = 0;
    } else if (rank == 1) {
        val = 32;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            data[i * n * len + j * len] = val++;
        }
    }

    std::vector<int64_t> out_sizes = {world_size * m, n, len};
    std::vector<int64_t> out_strides = {n * len, len, 1};
    std::vector<uint8_t> out_data(world_size * m * n * len);

    kutacc::TensorWrapper in(data.data(), std::move(sizes), std::move(strides), dim, kutacc::kBF16);
    kutacc::TensorWrapper out(out_data.data(), std::move(out_sizes), std::move(out_strides), dim, kutacc::kBF16);

    kutacc_af2_all_gather(in.get_tensor(), out.get_tensor());

    int error = 0;
    for (int i = 0; i < world_size * m; ++i) {
        for (int j = 0; j < n; ++j) {
            ASSERT_EQ(out_data[i * n * len + j * len], expect_data[i * n * len + j * len]);
        }
    }

    kutacc_comm_fini();
}
