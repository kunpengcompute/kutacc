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
echo "==================KUTACC FlashMLA UT=================="
if [ -z $MULTI_THREADS_BACKEND ]; then
    export MULTI_THREADS_BACKEND=kupl  #  kupl; omp
fi

NUM_THREADS=32
if [ "$MULTI_THREADS_BACKEND" = "omp" ]; then
    export OMP_NUM_THREADS=$NUM_THREADS
    export OMP_PROC_BIND=close
elif [ "$MULTI_THREADS_BACKEND" = "kupl" ]; then
    export KUPL_EXECUTOR_COUNT=$NUM_THREADS
    export KUPL_EXECUTOR_BACKEND=pthread
fi

core=32
for ((i = 0; i < 2; i += 1)); do
    echo "------------------------------"
    h=$(($i + 16))
    echo NUMA $i:
    echo 3 > /proc/sys/vm/drop_caches
    echo 0 > /sys/devices/system/node/node${i}/hugepages/hugepages-2048kB/nr_hugepages
    echo 2000 > /sys/devices/system/node/node${h}/hugepages/hugepages-2048kB/nr_hugepages
    taskset -c $(($i * $core + $core - $NUM_THREADS))-$(($i * $core + $core - 1)) ./numa_duplication/numa_0/test_flash_mla
    echo 0 > /sys/devices/system/node/node${h}/hugepages/hugepages-2048kB/nr_hugepages
    echo ""
done

msg="run test: taskset -c $(($i * $core + $core - $NUM_THREADS))-$(($i * $core + $core - 1)) ./numa_duplication/numa_0/test_flash_mla ... "
if [[ $? != 0 ]]; then
    msg+="failed"
else
    msg+="success"
fi
echo $msg