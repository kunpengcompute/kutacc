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
set -e

TEST_DIR=$(cd $(dirname $0);pwd)
filter=""

function run_test() {
    if [[ -n "$filter" && ! "$@" =~ "$filter" ]]; then
        return
    fi
    msg="run test: $@ ... "
    $@
    if [[ $? != 0 ]]; then
        msg+="failed"
    else
        msg+="success"
    fi
    echo $msg
}

function test_embedding() {
    run_test taskset -c 0-31 embedding/test_embedding 128 7168 0 8080
    run_test taskset -c 0-31 embedding/test_embedding 128 7168 8080 16160
}

function test_matmul() {
    run_test taskset -c 0-31 matmul/test_igemm_bdq
    run_test taskset -c 0-31 matmul/test_bgemm
    run_test taskset -c 0-31 matmul/test_batched_gemm_woqs8 8 128 512 128   8 128 128 512
    run_test taskset -c 0-31 matmul/test_igemm_pack 128 7168 128 1792   128 1536 128 384
    run_test taskset -c 0-31 matmul/test_bgemm_pack 128 16 128 16   16 7168 16 224   128 8080 128 1010   8080 7168 1010 1792
    run_test taskset -c 0-31 matmul/test_batched_gemm_pack 8 128 512 128   8 128 128 512
}

function test_rmsnorm() {
    run_test taskset -c 0-31 rmsnorm/test_rmsnorm 128 7168 0
    run_test taskset -c 0-31 rmsnorm/test_rmsnorm 128 7168 1
    run_test taskset -c 0-31 rmsnorm/test_rmsnorm_quant 128 7168 0
    run_test taskset -c 0-31 rmsnorm/test_rmsnorm_quant 128 7168 1
}

function test_quant() {
    run_test taskset -c 0-31 utils/test_quant 128 7168
    run_test taskset -c 0-31 utils/test_quant 128 1536
}

function test_allgather() {
    cd allgather/script/
    chmod +x *.sh
    bash ${TEST_DIR}/install-numa-duplication.sh ../test_allgather 16
    bash test.sh
    cd ../..
    echo ""
}

function test_allreduce() {
    cd allreduce/script/
    chmod +x *.sh
    bash ${TEST_DIR}/install-numa-duplication.sh ../test_allreduce 16
    bash test.sh
    cd ../..
    echo ""
}

function test_moe() {
    cd moe/script/
    chmod +x *.sh
    bash ${TEST_DIR}/install-numa-duplication.sh ../test_dispatch_combine 16
    bash run.sh
    cd ../..
    echo ""
}

function test_flash_mla() {
    cd flash_mla/script/
    chmod +x *.sh
    bash ${TEST_DIR}/install-numa-duplication.sh ../test_flash_mla 1
    bash test.sh
    cd ../..
    echo ""
}

function main() {
    echo "==================KUTACC UT=================="
    cd build/
    test_embedding
    test_matmul
    test_rmsnorm
    test_quant
    echo ""
    test_allgather
    test_allreduce
    test_moe
    test_flash_mla
}

main
