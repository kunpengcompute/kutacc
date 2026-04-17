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

set(CMAKE_CXX_FLAGS "-std=c++17 -stdlib=libc++")
add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options(-march=armv8.2-a+sve2+bf16+sme -msve-vector-bits=512)
add_link_options(-rtlib=compiler-rt -unwindlib=libunwind)

include_directories(${CMAKE_SOURCE_DIR}/tool)

# test_igemm_bdq
set(TEST_IGEMM_DBQ test_igemm_bdq)
add_executable(${TEST_IGEMM_DBQ} test_igemm_bdq.cpp)
target_link_directories(${TEST_IGEMM_DBQ} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_IGEMM_DBQ} PRIVATE -lkutacc)

# test_bgemm
set(TEST_BGEMM test_bgemm)
add_executable(${TEST_BGEMM} test_bgemm.cpp)
target_link_directories(${TEST_BGEMM} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_BGEMM} PRIVATE -lkutacc)

# test_batched_gemm_woqs8
set(TEST_BATCHED_GEMM_WOQS8 test_batched_gemm_woqs8)
add_executable(${TEST_BATCHED_GEMM_WOQS8} test_batched_gemm_woqs8.cpp)
target_link_directories(${TEST_BATCHED_GEMM_WOQS8} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_BATCHED_GEMM_WOQS8} PRIVATE -lkutacc)

# test_igemm_pack
set(TEST_IGEMM_PACK test_igemm_pack)
add_executable(${TEST_IGEMM_PACK} test_igemm_pack.cpp)
target_link_directories(${TEST_IGEMM_PACK} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_IGEMM_PACK} PRIVATE -lkutacc)

# test_bgemm_pack
set(TEST_BGEMM_PACK test_bgemm_pack)
add_executable(${TEST_BGEMM_PACK} test_bgemm_pack.cpp)
target_link_directories(${TEST_BGEMM_PACK} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_BGEMM_PACK} PRIVATE -lkutacc)

# test_batched_gemm_pack
set(TEST_BATCHED_GEMM_PACK test_batched_gemm_pack)
add_executable(${TEST_BATCHED_GEMM_PACK} test_batched_gemm_pack.cpp)
target_link_directories(${TEST_BATCHED_GEMM_PACK} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_BATCHED_GEMM_PACK} PRIVATE -lkutacc)
