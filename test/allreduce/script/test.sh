#!/usr/bin/bash -xe

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
echo "==================KUTACC AllReduce UT=================="

MPI_PARA=()
export KUPL_SHM_BACKEND=WORMHOLE        #  POSIX; WORMHOLE
export BACKEND_MULTI_THREADS=KUPL

ENV_FLAG="-env"

if [[ "${KUPL_SHM_BACKEND}" = "POSIX" ]]; then
    MPI_PARA+=("${ENV_FLAG} KUPL_SHM_TYPE=posix")
elif [ "${KUPL_SHM_BACKEND}" = "WORMHOLE" ]; then
    MPI_PARA+=("${ENV_FLAG} KUPL_SHM_TYPE=sls ${ENV_FLAG} KUPL_SHM_ON_PACKAGE=y ${ENV_FLAG} KUPL_SHM_ENABLE_HUGEPAGE=y")
fi

WORLD_SIZE=16
THREAD=32

if [[ "${BACKEND_MULTI_THREADS}" = "KUPL" ]]; then
  MPI_PARA+=("${ENV_FLAG} KUPL_EXECUTOR_COUNT=$THREAD ${ENV_FLAG} KUPL_EXECUTOR_BACKEND=pthread")
elif [[ "${BACKEND_MULTI_THREADS}" = "OMP" ]]; then
  MPI_PARA+=("${ENV_FLAG} OMP_NUM_THREADS=$THREAD ${ENV_FLAG} OMP_PROC_BIND=close")
fi

for i in {0..15}; do
    h=$((${i} + 16))
    echo 0 > /sys/devices/system/node/node${i}/hugepages/hugepages-2048kB/nr_hugepages
    echo 1000 > /sys/devices/system/node/node${h}/hugepages/hugepages-2048kB/nr_hugepages
done

echo 3 > /proc/sys/vm/drop_caches

mpirun ${MPI_PARA[@]} -env MV2_DEBUG_SHOW_BACKTRACE=1 -env MV2_ENABLE_AFFINITY=0 -env MV2_USE_HUGEPAGES=0 -np $WORLD_SIZE  \
    ./allreduce_test.sh $THREAD

for i in {0..15}; do
    h=$((${i} + 16))
    echo 0 > /sys/devices/system/node/node${i}/hugepages/hugepages-2048kB/nr_hugepages
    echo 0 > /sys/devices/system/node/node${h}/hugepages/hugepages-2048kB/nr_hugepages
done