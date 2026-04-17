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

export MOE_EP=16
export MOE_DP=1
export MOE_TP=1
export MAX_TOKEN=$((${MOE_DP} * 1 * 128))
export MOE_BALANCE=1

export OMPI_COMM_WORLD_LOCAL_RANK=$MV2_COMM_WORLD_LOCAL_RANK
export OMPI_COMM_WORLD_RANK=$MV2_COMM_WORLD_RANK
if [[ $OMPI_COMM_WORLD_LOCAL_RANK = 0 ]]; then
  ps -aux | grep hydra_pmi_proxy | grep -v 'grep' | awk  '{print $2}' | \
    while read line; do taskset -pc 0 $line > /dev/null; done
fi

on_package_memory_node=$(("${OMPI_COMM_WORLD_LOCAL_RANK}" + 16))

# 开大页
echo 2000 > /sys/devices/system/node/node${on_package_memory_node}/hugepages/hugepages-2048kB/nr_hugepages

fun_c=numa_duplication/numa_$OMPI_COMM_WORLD_LOCAL_RANK/test_dispatch_combine
local_rank=$OMPI_COMM_WORLD_LOCAL_RANK
core=38
mkdir -p log
if [[ "${OMPI_COMM_WORLD_RANK}" = "0" ]]; then
  taskset -c $((1+$local_rank*$core))-$(($1+$local_rank*$core)) $fun_c $2 2>&1 | tee log/${OMPI_COMM_WORLD_RANK}.log
  msg="run test: "
  msg+="taskset -c $((1+$local_rank*$core))-$(($1+$local_rank*$core)) $fun_c $2 2>&1 | tee log/${OMPI_COMM_WORLD_RANK}.log"
  msg+=" ... "
  if [[ $? != 0 ]]; then
    msg+="failed"
  else
    msg+="success"
  fi
  echo $msg
else
  taskset -c $((1+$local_rank*$core))-$(($1+$local_rank*$core)) $fun_c $2 2>&1 > log/${OMPI_COMM_WORLD_RANK}.log
  msg="run test: "
  msg+="taskset -c $((1+$local_rank*$core))-$(($1+$local_rank*$core)) $fun_c $2 2>&1 > log/${OMPI_COMM_WORLD_RANK}.log"
  msg+=" ... "
  if [[ $? != 0 ]]; then
    msg+="failed"
  else
    msg+="success"
  fi
  echo $msg
fi

# 关闭大页
echo 0 > /sys/devices/system/node/node${on_package_memory_node}/hugepages/hugepages-2048kB/nr_hugepages
