#!/bin/bash -xe

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

TEST_DIR=$(cd $(dirname $0);pwd)

function install_numa_duplication() {
    echo -n "Installing numa duplication of $2 - "
    mkdir -p numa_duplication
    cd numa_duplication
    bash ${TEST_DIR}/script/install_numa_duplication.sh ${TEST_DIR}/build/$1/$2 16 $2
    cd ..
    echo "done"
}

if [ -z $KUTACC_INSTALL_ROOT ]; then
    export KUTACC_INSTALL_ROOT=$(dirname $PWD)/install
fi
export LD_LIBRARY_PATH=$KUTACC_INSTALL_ROOT/lib:$LD_LIBRARY_PATH

ENABLE_EXTRA_PREFETCH=on

bash ../build.sh                                    \
    --cleanup=off                                   \
    --enable_extra_prefetch=$ENABLE_EXTRA_PREFETCH  \
    --parallel_backend=kupl_global

# rm -rf build
cmake -B build . -DCMAKE_BUILD_TYPE=Release -GNinja 
cmake --build build -j144

chmod +x ${TEST_DIR}/script/*.sh

rm -rf numa_duplication
install_numa_duplication comm test_alltoall2D
install_numa_duplication comm test_allreduce
install_numa_duplication comm test_allgather
install_numa_duplication comm test_reduce_scatter
install_numa_duplication moe test_dispatch_combine
install_numa_duplication alphafold3 test_all_gather
install_numa_duplication alphafold3 test_all_gather_dim1
install_numa_duplication alphafold3 test_all2all