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

export KUPL_SHM_BACKEND=WORMHOLE    # WORMHOLE; WORMHOLE
export MULTI_THREADS_BACKEND=KUPL   # KUPL; OMP
export NUM_THREADS=32
export RUN_TEST_ORIGIN_HOST=$(hostname -s)

TEST_DIR=$(cd $(dirname $0);pwd)
TIME_LIMIT=30
filter=""

function run_test() {
    if [[ -n "$filter" && ! "$1" =~ "$filter" ]]; then
        return
    fi

    echo 3 > /proc/sys/vm/drop_caches

    echo -n "  Running test on $1 "
    printf "%*s" $((25-${#1})) | tr " " "."

    {
        if [[ "$2" != mpirun* ]]; then
            timeout -k 5s "${TIME_LIMIT}s" $2 > log/$1.log 2>&1
        else
            timeout -k 5s "${TIME_LIMIT}s" $2
        fi
    } 2>/dev/null

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 124 ]; then
        msg="TIMEOUT(>${TIME_LIMIT}s)"
    elif [ $EXIT_CODE -eq 0 ]; then
        msg="PASSED"
    else
        msg="FAILED"
    fi

    echo " $msg"
}

MPI_PARA=""

if [[ "${KUPL_SHM_BACKEND}" = "POSIX" ]]; then
    MPI_PARA="$MPI_PARA -env KUPL_SHM_TYPE=posix"
elif [ "${KUPL_SHM_BACKEND}" = "WORMHOLE" ]; then
    MPI_PARA="$MPI_PARA -env KUPL_SHM_TYPE=sls -env KUPL_SHM_ON_PACKAGE=y -env KUPL_SHM_ENABLE_HUGEPAGE=y"
fi

if [ "$MULTI_THREADS_BACKEND" = "KUPL" ]; then
    export KUPL_EXECUTOR_COUNT=$NUM_THREADS
    export KUPL_EXECUTOR_BACKEND=pthread
    MPI_PARA+="$MPI_PARA -env KUPL_EXECUTOR_COUNT=$NUM_THREADS -env KUPL_EXECUTOR_BACKEND=pthread"
elif [ "$MULTI_THREADS_BACKEND" = "OMP" ]; then
    export OMP_NUM_THREADS=$NUM_THREADS
    export OMP_PROC_BIND=close
    MPI_PARA+="$MPI_PARA -env OMP_NUM_THREADS=$NUM_THREADS -env OMP_PROC_BIND=close"
fi

MPI_PARA="$MPI_PARA -env MV2_DEBUG_SHOW_BACKTRACE=1 -env MV2_ENABLE_AFFINITY=0 -env MV2_USE_HUGEPAGES=0"

MPI_PARA="$MPI_PARA -env RUN_TEST_ORIGIN_HOST=${RUN_TEST_ORIGIN_HOST}"

CORES="1-$NUM_THREADS"

for ((i = 16; i < 32; i += 1)); do
    echo 2000 > /sys/devices/system/node/node${i}/hugepages/hugepages-2048kB/nr_hugepages
done

rm -rf log
mkdir -p log

echo "===================== KuTACC UT ====================="

echo "===================== AlphaFold3 UT ====================="

echo "[ADAPTIVE_LAYERNORM UT]"
run_test test_adapln_cross "taskset -c $CORES ./build/alphafold3/test_adapln 128 128 128 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"
run_test test_adapln_self "taskset -c $CORES ./build/alphafold3/test_adapln 1 768 384 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"
run_test test_adapln_tran "taskset -c $CORES ./build/alphafold3/test_adapln 32 128 128 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"

echo "[FLASH_ATTENTION UT]"
run_test test_flash_attention_cross "taskset -c $CORES ./build/alphafold3/test_flash_attention 32 128 4 32 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"
run_test test_flash_attention_self "taskset -c $CORES ./build/alphafold3/test_flash_attention 64 64 16 48 1 64 1 128 1 320 1 448 1 640 1 768"
run_test test_flash_attention_grid "taskset -c $CORES ./build/alphafold3/test_flash_attention 96 96 4 16 1 64 1 128 1 320 1 448 1 640 1 768"

echo "[ADAPTIVE_ZEROINIT UT]"
run_test test_adapzi_cross "taskset -c $CORES ./build/alphafold3/test_adapzi 32 128 128 128 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"
run_test test_adapzi_self "taskset -c $CORES ./build/alphafold3/test_adapzi 1 768 768 384 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"
run_test test_adapzi_tran "taskset -c $CORES ./build/alphafold3/test_adapzi 1 768 768 1536 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"

echo "[GLU UT]"
run_test test_glu_permute "taskset -c $CORES ./build/alphafold3/test_glu_permute 128 256 8 64 64 8 64 64 16 128 128 16 128 128 40 320 320 40 320 320 56 448 448 56 448 448 80 640 640 80 640 640 96 768 768 96 768 768"
run_test test_glu_swish "taskset -c $CORES ./build/alphafold3/test_glu_swish 768 1536 5 64 5 128 5 320 5 448 5 640 5 768"

echo "[COMM UT]"
run_test test_all_gather "mpirun $MPI_PARA -np 8 ./script/mpi_launch.sh test_all_gather"
run_test test_all_gather_dim1 "mpirun $MPI_PARA -np 8 ./script/mpi_launch.sh test_all_gather_dim1"
run_test test_all2all "mpirun $MPI_PARA -np 8 ./script/mpi_launch.sh test_all2all"

echo "[EINSUM UT]"
run_test test_einsum "taskset -c $CORES ./build/alphafold3/test_einsum 64 128 320 448 640 768"

echo "[OUTER_PRODUCT_MEAN UT]"
run_test test_outer_product_mean "taskset -c $CORES ./build/alphafold3/test_outer_product_mean 64 128 320 448 640 768"

echo "[BGEMM UT]"
run_test test_bgemm "taskset -c $CORES ./build/alphafold3/test_bgemm 1 64 1 128 1 320 1 448 1 640 1 768 5 64 5 128 5 320 5 448 5 640 5 768"

echo "===================== DEEPSEEK UT ====================="
echo "[MATMUL UT]"
run_test s8_s8_packed_gemm_bf16_dq "taskset -c $CORES ./build/matmul/test_s8_s8_packed_gemm_bf16_dq"
run_test fusedmoe "taskset -c $CORES ./build/matmul/test_fusedmoe"
run_test bf16_packed_gemm "taskset -c $CORES ./build/matmul/test_bf16_packed_gemm"
run_test batch_bf16_s8_packed_gemm_bf16 "taskset -c $CORES ./build/matmul/test_batch_bf16_s8_packed_gemm_bf16 8 128 512 128   8 128 128 512"
run_test s8_gemm_pack "taskset -c $CORES ./build/matmul/test_s8_gemm_pack 128 1536 64 1536   128 7168 128 1792   128 2048 128 1024   128 7168 128 448   128 1024 128 1024   128 128 128 128   128 1152 128 1152   96 1536 48 1536   96 7168 96 1792   96 2048 96 1024   96 7168 96 448   96 1024 96 1024   96 128 96 128   96 1152 96 1152"
run_test bf16_gemm_pack "taskset -c $CORES ./build/matmul/test_bf16_gemm_pack 128 16 128 16   16 7168 16 224   128 8080 128 1010   8080 7168 1010 1792"
run_test batch_bf16_gemm_pack "taskset -c $CORES ./build/matmul/test_batch_bf16_gemm_pack 8 128 512 128   8 128 128 512"

echo "[ATTN UT]"
run_test flash_mla_dense_decode "taskset -c $CORES ./build/flash_mla/dense_decode/test_flash_mla_dense_decode"
run_test flash_mla_sparse_decode "taskset -c $CORES ./build/flash_mla/sparse_decode/test_flash_mla_sparse_decode"
run_test flash_attn "taskset -c $CORES ./build/flash_attn/test_flash_attn"

echo "[COMM UT]"
run_test alltoall2D "mpirun $MPI_PARA -np 16 ./script/mpi_launch.sh test_alltoall2D"
run_test allreduce "mpirun $MPI_PARA -np 16 ./script/mpi_launch.sh test_allreduce"
run_test allgather "mpirun $MPI_PARA -np 16 ./script/mpi_launch.sh test_allgather"
run_test reduce_scatter "mpirun $MPI_PARA -np 16 ./script/mpi_launch.sh test_reduce_scatter"

echo "[RMSNORM UT]"
run_test rmsnorm_case0 "taskset -c $CORES ./build/rmsnorm/test_rmsnorm"
run_test rmsnorm_quant_case0 "taskset -c $CORES ./build/rmsnorm/test_rmsnorm_quant"

echo "[SILU_MUL_QUANT UT]"
run_test silu_mul_case0 "taskset -c $CORES ./build/silu_mul/test_silu_mul_quant"

echo "[ROPE_OUT UT]"
run_test rope_case0 "taskset -c $CORES ./build/rope/test_rope"

echo "[MUL_SCALAR_ADD UT]"
run_test mul_scalar_add_case0 "taskset -c $CORES ./build/mul_scalar_add/test_mul_scalar_add"

echo "[SCALE_SOFTMAX UT]"
run_test scale_softmax_case0 "taskset -c $CORES ./build/scale_softmax/test_scale_softmax"

echo "[EMBEDDING UT]"
run_test embedding_case0 "taskset -c $CORES ./build/embedding/test_embedding"

echo "[QUANT UT]"
run_test quant_case0 "taskset -c $CORES ./build/quant/test_quant"

echo "[MOE UT]"
run_test disp_comb_decode "mpirun $MPI_PARA -np 16 ./script/mpi_launch.sh test_dispatch_combine 0 0 0 0 16 2 1 64 1 16 100"
# HOSTS=$(echo 29.204.23.{10..17} 29.204.23.{20..27} | tr ' ' ',')
# run_test disp_comb_32p_prefill "mpirun $MPI_PARA -np 256 -ppn 16 -hosts ${HOSTS} ./script/mpi_launch.sh test_dispatch_combine 1 0 0 0 256 16 1 4096 0 256 10"
# HOSTS=$(echo 29.204.23.{10..17} 29.204.23.{20..27} 29.204.23.{30..37} 29.204.23.{40..47} | tr ' ' ',')
# run_test disp_comb_64p_decode_case0 "mpirun $MPI_PARA -np 512 -ppn 16 -hosts ${HOSTS} ./script/mpi_launch.sh test_dispatch_combine 0 0 0 0 256 32 1 64 1 256 10"
# run_test disp_comb_64p_decode_case1 "mpirun $MPI_PARA -np 512 -ppn 16 -hosts ${HOSTS} ./script/mpi_launch.sh test_dispatch_combine 0 0 0 1 256 32 1 64 1 256 10"
# run_test disp_comb_64p_decode_case2 "mpirun $MPI_PARA -np 512 -ppn 16 -hosts ${HOSTS} ./script/mpi_launch.sh test_dispatch_combine 0 1 1 0 256 32 1 64 1 256 10"
# run_test disp_comb_64p_decode_case3 "mpirun $MPI_PARA -np 512 -ppn 16 -hosts ${HOSTS} ./script/mpi_launch.sh test_dispatch_combine 0 1 1 1 256 32 1 64 1 256 10"

echo "====================================================="

for ((i = 16; i < 32; i += 1)); do
    echo 0 > /sys/devices/system/node/node${i}/hugepages/hugepages-2048kB/nr_hugepages
done