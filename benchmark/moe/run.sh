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

MPI_PARA=()
export KUPL_SHM_BACKEND=WORMHOLE    #  POSIX; WORMHOLE

## 16Node
WORLD_SIZE=256
THREAD=32
HOSTS="xx.xx.xx.1,xx.xx.xx.2,xx.xx.xx.3,xx.xx.xx.4,xx.xx.xx.5,xx.xx.xx.6,xx.xx.xx.7,xx.xx.xx.8,xx.xx.xx.9,xx.xx.xx.10,xx.xx.xx.11,xx.xx.xx.12,xx.xx.xx.13,xx.xx.xx.14,xx.xx.xx.15,xx.xx.xx.16"

ENV_FLAG="-env"
if [[ "${KUPL_SHM_BACKEND}" = "POSIX" ]]; then
  MPI_PARA+=("${ENV_FLAG} KUPL_SHM_TYPE=posix")
elif [ "${KUPL_SHM_BACKEND}" = "WORMHOLE" ]; then
  MPI_PARA+=("${ENV_FLAG} KUPL_SHM_TYPE=sls ${ENV_FLAG} KUPL_SHM_ON_PACKAGE=y ${ENV_FLAG} KUPL_SHM_ENABLE_HUGEPAGE=y")
fi

mpirun -env PATH "$PATH" -env LD_LIBRARY_PATH "$LD_LIBRARY_PATH" ${MPI_PARA[@]} \
  -env MV2_DEBUG_SHOW_BACKTRACE=1 \
  -env MV2_ENABLE_AFFINITY=0 \
  -env MV2_USE_HUGEPAGES=0 \
  -np $WORLD_SIZE -ppn 16 -hosts ${HOSTS} \
  bash ./script/generate.sh $THREAD 1000
