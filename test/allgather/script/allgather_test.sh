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

local_rank=$MV2_COMM_WORLD_LOCAL_RANK
core=38
BIND_PARA="taskset -c $(($local_rank * $core + 1))-$(($local_rank * $core + $1))"

if [ "${KUPL_SHM_BACKEND}" = "POSIX" ]; then
    on_package_memory_node=$(($local_rank + 16))
    BIND_PARA="$BIND_PARA numactl -m ${on_package_memory_node}"
fi

msg="run test:"
msg+="$BIND_PARA ./numa_duplication/numa_$local_rank/test_allgather"
msg+=" ... "
$BIND_PARA ./numa_duplication/numa_$local_rank/test_allgather
if [[ $? != 0 ]]; then
    msg+="failed"
else
    msg+="success"
fi
echo $msg