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

# test_quant
set(TEST_QUANT test_quant)
add_executable(${TEST_QUANT} test_quant.cpp)
target_link_directories(${TEST_QUANT} PRIVATE $ENV{KUTACC_INSTALL_ROOT}/lib)
target_link_libraries(${TEST_QUANT} PRIVATE -lkutacc)
