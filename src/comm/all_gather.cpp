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

#include <cstring>
#include "../utils/parallel.h"
#include "../tensor/tensor.h"
#include "kutacc.h"
#include "kupl.h"
#include <mpi.h>

namespace kutacc {
void af2_all_gather_kernel(Tensor &data, Tensor &out)
{
    int64_t m = out.sizes()[0];
    int64_t n = out.sizes()[1];
    int64_t len = data.sizes()[2];

    if (data.sizes()[0] == m) {
        std::swap(m, n);
        std::swap(data.sizes_ref()[0], data.sizes_ref()[1]);
        std::swap(data.strides_ref()[0], data.strides_ref()[1]);
        std::swap(out.sizes_ref()[0], out.sizes_ref()[1]);
        std::swap(out.strides_ref()[0], out.strides_ref()[1]);
    }

    int64_t block_m = (m + world_size - 1) / world_size;

    KUTACC_CHECK(data.strides()[2] == 1, data.strides()[2]);
    KUTACC_CHECK(out.strides()[2] == 1, out.strides()[2]);

    int64_t subblock_m = buffer_size / (n * len);
    KUTACC_CHECK(subblock_m > 0, buffer_size, " ", n, " ", len);
    for (int64_t block_mi_start = 0; block_mi_start < block_m; block_mi_start += subblock_m) {
        int64_t size_m = std::min(subblock_m, block_m - block_mi_start);

        parallel_for(0, 2 * size_m * n, 1, [&](int64_t start, int64_t end) {
            collapse_for(start, end, 2, size_m, n, [&](int64_t direct, int64_t sbmi, int64_t ni) {
                int64_t bmi = sbmi + block_mi_start;
                if (bmi < m - rank * block_m) {
                    if (!direct) {
                        std::memcpy((uint8_t *)(kupl_recvbuf) + (sbmi * n + ni) * len,
                               (uint8_t *)(data.data_ptr()) + bmi * data.strides()[0] + ni * data.strides()[1],
                               (size_t)len);
                    } else {
                        std::memcpy((uint8_t *)(out.data_ptr()) + (bmi + rank * block_m) * out.strides()[0] +
                                   ni * out.strides()[1],
                               (uint8_t *)(data.data_ptr()) + bmi * data.strides()[0] + ni * data.strides()[1],
                               (size_t)len);
                    }
                }
            });
        });

        MPI_Barrier(MPI_COMM_WORLD);

        int64_t par_size = world_size * size_m * n;
        parallel_for(0, par_size, 1, [&](int64_t start, int64_t end) {
            collapse_for(start, end, world_size, size_m, n, [&](int64_t ri, int64_t sbmi, int64_t ni) {
                int64_t bmi = sbmi + block_mi_start;
                if (bmi < m - ri * block_m) {
                    if (ri != rank) {
                        void *remote_buffer;
                        kupl_shm_win_query(kupl_recvbuf_win, (int)ri, &remote_buffer);
                        std::memcpy((uint8_t *)(out.data_ptr()) + (bmi + ri * block_m) * out.strides()[0] +
                                   ni * out.strides()[1],
                               (uint8_t *)(remote_buffer) + (sbmi * n + ni) * len, (size_t)len);
                    }
                }
            });
        });

        MPI_Barrier(MPI_COMM_WORLD);
    }
}
}

void kutacc_af2_all_gather(kutacc_tensor_h data, kutacc_tensor_h out)
{
    if (data == nullptr || out == nullptr) {
        KUTACC_CHECK(data != nullptr, data);
        KUTACC_CHECK(out != nullptr, out);
        return;
    }
    af2_all_gather_kernel(*kutacc::convertKutaccTensor(data), *kutacc::convertKutaccTensor(out));
}