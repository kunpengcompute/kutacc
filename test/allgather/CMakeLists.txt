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

set(TEST_ALLGATHER test_allgather)
add_executable(${TEST_ALLGATHER} csrc/main.cpp)

set(CMAKE_CXX_FLAGS "-std=c++17 -stdlib=libc++")
target_compile_features(${TEST_ALLGATHER} PRIVATE cxx_std_17)
target_compile_options(${TEST_ALLGATHER} PRIVATE -Wall -Wno-unused-variable -Wno-unused-function)
target_compile_options(${TEST_ALLGATHER} PRIVATE -Wno-undefined-arm-za -Wno-undefined-arm-streaming)
target_compile_options(${TEST_ALLGATHER} PRIVATE -O3)
target_compile_options(${TEST_ALLGATHER} PRIVATE -mcpu=native)
target_link_options(${TEST_ALLGATHER} PRIVATE -rtlib=compiler-rt -unwindlib=libunwind)

# mpi
message(STATUS "Allgather UT use MVAPICH.")
target_include_directories(${TEST_ALLGATHER} PRIVATE $ENV{MPI_PATH}/include)
target_link_libraries(${TEST_ALLGATHER} PRIVATE
	-Wl,--whole-archive
    $ENV{MPI_PATH}/lib/libmpi.a
    -Wl,--no-whole-archive
    -ludev -lnuma -lm -lpthread
)

# numa
target_link_libraries(${TEST_ALLGATHER} PRIVATE -lnuma)

# kupl
target_link_libraries(${TEST_ALLGATHER} PRIVATE -lkupl)

# kutacc
target_include_directories(${TEST_ALLGATHER} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/include)
target_link_directories(${TEST_ALLGATHER} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_ALLGATHER} PRIVATE -lkutacc)