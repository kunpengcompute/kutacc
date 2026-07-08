#!/bin/bash

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

GLOBAL_RANK=$MV2_COMM_WORLD_RANK
LOCAL_RANK=$MV2_COMM_WORLD_LOCAL_RANK
CURRENT_HOST=$(hostname -s)

on_package_memory_node=$(("${LOCAL_RANK}" + 16))
echo 2000 > /sys/devices/system/node/node${on_package_memory_node}/hugepages/hugepages-2048kB/nr_hugepages

BIND_PARA="taskset -c $(($LOCAL_RANK*38+1))-$(($LOCAL_RANK*38+$NUM_THREADS))"

if [ "${KUPL_SHM_BACKEND}" = "POSIX" ]; then
    BIND_PARA="$BIND_PARA numactl -m $(($LOCAL_RANK+16))"
fi

mkdir -p log/$1

EXE_NAME=$1
shift 1
$BIND_PARA ./numa_duplication/$EXE_NAME/numa_$LOCAL_RANK/$EXE_NAME "$@" > log/$EXE_NAME/rank$GLOBAL_RANK.log 2>&1
APP_EXIT_CODE=$?

if [ "${CURRENT_HOST}" != "${RUN_TEST_ORIGIN_HOST}" ]; then
    echo 0 > /sys/devices/system/node/node${on_package_memory_node}/hugepages/hugepages-2048kB/nr_hugepages
fi

exit $APP_EXIT_CODE