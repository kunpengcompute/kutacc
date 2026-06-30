cmake_minimum_required(VERSION 3.20)

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

project(kutacc_test VERSION 1.0 LANGUAGES C CXX ASM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -stdlib=libc++")

add_compile_options(-mcpu=native)
add_compile_options(-Wall -Wextra -Wpedantic)

add_link_options(-rtlib=compiler-rt -unwindlib=libunwind)

include_directories($ENV{KUTACC_PATH}/install/include)
include_directories($ENV{KUTACC_PATH}/test/tool)
link_directories($ENV{KUTACC_PATH}/install/lib)
link_libraries(-lkutacc)

add_subdirectory(alphafold3)
add_subdirectory(flash_mla)
add_subdirectory(flash_attn)
add_subdirectory(comm)
add_subdirectory(matmul)
add_subdirectory(rmsnorm)
add_subdirectory(embedding)
add_subdirectory(quant)
add_subdirectory(moe)
add_subdirectory(silu_mul)
add_subdirectory(rope)
add_subdirectory(mul_scalar_add)
add_subdirectory(scale_softmax)