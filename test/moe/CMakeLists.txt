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

add_compile_options(-Wno-unused-variable -Wno-unused-function)
add_compile_options(-Wno-undefined-arm-za -Wno-undefined-arm-streaming)
include_directories($ENV{MPI_PATH}/include)
add_link_options(
    -Wl,--whole-archive
    $ENV{MPI_PATH}/lib/libmpi.a
    -Wl,--no-whole-archive
    -ludev -lnuma -lm -lpthread
)


include_directories($ENV{KUPL_PATH}/include)
link_directories($ENV{KUPL_PATH}/lib)
link_libraries(-lkupl)

set(TEST_DISPATCH_COMBINE test_dispatch_combine)
add_executable(${TEST_DISPATCH_COMBINE} 
    test_dispatch_combine.cpp 
    low_latency/low_latency.cpp 
    utils/memory_pool.cpp
)