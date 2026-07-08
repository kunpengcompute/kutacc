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

add_compile_options(-Wextra -Wpedantic)

include_directories(${CMAKE_SOURCE_DIR}/tool)

# test_s8_s8_packed_gemm_bf16_dq
set(TEST_S8_S8_PACKED_GEMM_BF16_DQ test_s8_s8_packed_gemm_bf16_dq)
add_executable(${TEST_S8_S8_PACKED_GEMM_BF16_DQ} test_s8_s8_packed_gemm_bf16_dq.cpp)

# test_fusedmoe
set(TEST_FUSEDMOE test_fusedmoe)
add_executable(${TEST_FUSEDMOE} test_fusedmoe.cpp)

# test_bf16_packed_gemm
set(TEST_BF16_PACKED_GEMM test_bf16_packed_gemm)
add_executable(${TEST_BF16_PACKED_GEMM} test_bf16_packed_gemm.cpp)

# test_batch_bf16_s8_packed_gemm_bf16
set(TEST_BATCHED_BF16_S8_GEMM_BF16 test_batch_bf16_s8_packed_gemm_bf16)
add_executable(${TEST_BATCHED_BF16_S8_GEMM_BF16} test_batch_bf16_s8_packed_gemm_bf16.cpp)

# test_s8_gemm_pack
set(TEST_S8_GEMM_PACK test_s8_gemm_pack)
add_executable(${TEST_S8_GEMM_PACK} test_s8_gemm_pack.cpp)

# test_bf16_gemm_pack
set(TEST_BF16_GEMM_PACK test_bf16_gemm_pack)
add_executable(${TEST_BF16_GEMM_PACK} test_bf16_gemm_pack.cpp)

# test_batch_bf16_gemm_pack
set(TEST_BATCHED_BF16_GEMM_PACK test_batch_bf16_gemm_pack)
add_executable(${TEST_BATCHED_BF16_GEMM_PACK} test_batch_bf16_gemm_pack.cpp)