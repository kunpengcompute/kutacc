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

include_directories($ENV{MPI_PATH}/include)
add_compile_options(-Wno-unused-variable)
add_link_options(
	-Wl,--whole-archive
    $ENV{MPI_PATH}/lib/libmpi.a
    -Wl,--no-whole-archive
    -ludev -lnuma -lm -lpthread
)
link_libraries(-lkupl)

# test_alltoall2D
set(TEST_ALLTOALL2D test_alltoall2D)
add_executable(${TEST_ALLTOALL2D} test_alltoall2D.cpp)

# test_allreduce
set(TEST_ALLREDUCE test_allreduce)
add_executable(${TEST_ALLREDUCE} test_allreduce.cpp)

# test_allgather
set(TEST_ALLGATHER test_allgather)
add_executable(${TEST_ALLGATHER} test_allgather.cpp)

# test_reduce_scatter
set(TEST_REDUCE_SCATTER test_reduce_scatter)
add_executable(${TEST_REDUCE_SCATTER} test_reduce_scatter.cpp)