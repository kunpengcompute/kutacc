#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# KuTACC is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#        http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#

world_size=2
num_threads=64
test_bin_path=../install/bin/

function test()
{
    method=$1
    SHMID=kutacc_comm_test mpirun \
        --allow-run-as-root \
        -n ${world_size} \
        --map-by numa \
        -x OMP_NUM_THREADS=${num_threads} \
        ./${method}
}

function main()
{
    if [ ! -f "${test_bin_path}/kutacc_cal_test" ]; then
        echo "${test_bin_path}/kutacc_cal_test does not exist, please check!"
    else
        echo "========== running kutacc_cal_test =========="
        ${test_bin_path}/kutacc_cal_test
    fi

    if [ ! -f "${test_bin_path}/kutacc_comm_test" ]; then
        echo "${test_bin_path}/kutacc_comm_test does not exist, please check!"
    else
        echo "========== running kutacc_comm_test =========="
        test ${test_bin_path}/kutacc_comm_test
    fi
}

main
