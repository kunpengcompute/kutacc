cmake_minimum_required(VERSION 3.22)

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -stdlib=libc++ -mcpu=native")
add_compile_options(-Wno-unused-parameter -Wno-vla-cxx-extension -Wno-undefined-arm-za -Wno-undefined-arm-streaming)

include_directories($ENV{MPI_PATH}/include)
add_link_options(
	-Wl,--whole-archive
    $ENV{MPI_PATH}/lib/libmpi.a
    -Wl,--no-whole-archive
    -ludev -lnuma -lm -lpthread
)
link_libraries(-lkupl)

include_directories(${CMAKE_SOURCE_DIR}/tool)
include_directories($ENV{KUTACC_INSTALL_ROOT}/include)
link_directories($ENV{KUTACC_INSTALL_ROOT}/lib)

set(TEST_ADAPLN test_adapln)
add_executable(${TEST_ADAPLN} test_adapln.cpp)

set(TEST_FLASH_ATTENTION test_flash_attention)
add_executable(${TEST_FLASH_ATTENTION} test_flash_attn.cpp)

set(TEST_ADAPZI test_adapzi)
add_executable(${TEST_ADAPZI} test_adapzi.cpp)

set(TEST_GLU_PERMUTE test_glu_permute)
add_executable(${TEST_GLU_PERMUTE} test_glu_permute.cpp)

set(TEST_GLU_SWISH test_glu_swish)
add_executable(${TEST_GLU_SWISH} test_glu_swish.cpp)

set(TEST_ALL_GATHER test_all_gather)
add_executable(${TEST_ALL_GATHER} test_all_gather.cpp)

set(TEST_ALL_GATHER_DIM1 test_all_gather_dim1)
add_executable(${TEST_ALL_GATHER_DIM1} test_all_gather_dim1.cpp)

set(TEST_ALL2ALL test_all2all)
add_executable(${TEST_ALL2ALL} test_all2all.cpp)

set(TEST_BGEMM test_bgemm)
add_executable(${TEST_BGEMM} test_bgemm.cpp)

set(TEST_EINSUM test_einsum)
add_executable(${TEST_EINSUM} test_einsum.cpp)

set(TEST_OUTER_PRODUCT_MEAN test_outer_product_mean)
add_executable(${TEST_OUTER_PRODUCT_MEAN} test_outer_product_mean.cpp)