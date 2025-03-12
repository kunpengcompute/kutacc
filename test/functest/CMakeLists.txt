#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# KuTACC is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#        http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#

# 最低CMake版本要求（支持C++17 + find_package(GTest)）
cmake_minimum_required(VERSION 3.14)
project(kutacc_gtest_test)

# ========== 1. 基础编译配置（C++17 + 标准库 + ABI） ==========
# 强制C++17（统一项目和库的编译标准）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用编译器扩展（如gnu++17）

# 编译器指定（毕昇编译器/Clang，可选：若环境已默认则可注释）
set(CMAKE_CXX_COMPILER "mpic++")
set(CMAKE_C_COMPILER "mpicc")

# 统一ABI + libc++（解决__cxx11符号/标准库兼容问题）
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_options(-stdlib=libstdc++)  # Clang/libc++
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++ -lstdc++ -lpthread")

# ========== 2. 从环境变量读取GTest库根目录并验证 ==========
set(GTEST_ROOT_ENV $ENV{GTEST_ROOT})
if(NOT GTEST_ROOT_ENV)
    message(FATAL_ERROR "环境变量GTEST_ROOT未设置！请先export GTEST_ROOT=你的GTest安装根目录")
endif()
set(GTEST_ROOT ${GTEST_ROOT_ENV} CACHE PATH "GTest根目录" FORCE)
message(STATUS "✅ 从环境变量读取GTEST_ROOT: ${GTEST_ROOT}")

# ========== 3. 查找GTest库（标准方式） ==========
find_package(GTest REQUIRED)
if(NOT GTEST_FOUND)
    message(FATAL_ERROR "未找到GTest库！请检查GTEST_ROOT是否正确")
endif()
message(STATUS "✅ 找到GTest头文件路径: ${GTEST_INCLUDE_DIRS}")
message(STATUS "✅ 找到GTest库文件: ${GTEST_LIBRARIES}")

# ========== 4. 查找kutacc库（手动方式，非标准CMake模块） ==========
# 4.1 读取kutacc安装目录
set(KUTACC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../install)
if(NOT KUTACC_ROOT)
    message(FATAL_ERROR "未找到${KUTACC_ROOT}，请确认你的kutacc安装根目录")
endif()
message(STATUS "✅ 读取到kutacc安装路径: ${KUTACC_ROOT}")

# 4.2 kutacc头文件路径
set(KUTACC_INCLUDE_DIR "${KUTACC_ROOT}/include")
if(NOT EXISTS ${KUTACC_INCLUDE_DIR})
    message(FATAL_ERROR "kutacc头文件路径不存在: ${KUTACC_INCLUDE_DIR}")
endif()
message(STATUS "✅ 找到kutacc头文件路径: ${KUTACC_INCLUDE_DIR}")

# 4.3 kutacc库文件（优先找动态库.so，其次静态库.a）
find_library(KUTACC_LIB
    NAMES kutacc libkutacc  # 库名（根据实际库文件名调整，如libkutacc.so则写kutacc）
    PATHS "${KUTACC_ROOT}/lib" "${KUTACC_ROOT}/lib64"  # 库文件目录（常见lib/lib64）
    NO_DEFAULT_PATH  # 只从指定路径找，避免系统库干扰
)
if(NOT KUTACC_LIB)
    message(FATAL_ERROR "未找到kutacc库！请检查KUTACC_ROOT或库文件名是否正确")
endif()
message(STATUS "✅ 找到kutacc库文件: ${KUTACC_LIB}")

# ========== 5. 定义测试目标 ==========
file(GLOB_RECURSE TEST_CAL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cal/*.cpp"
)

file(GLOB_RECURSE TEST_COMM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/comm/*.cpp"
)

add_executable(kutacc_cal_test ${TEST_CAL_SOURCES})
add_executable(kutacc_comm_test ${TEST_COMM_SOURCES})

# ========== 6. 添加头文件路径（GTest + kutacc） ==========
target_include_directories(kutacc_cal_test
    PRIVATE
    ${GTEST_INCLUDE_DIRS}    # GTest头文件
    ${KUTACC_INCLUDE_DIR}    # kutacc头文件
    ${CMAKE_SOURCE_DIR}      # 若测试代码有本地头文件，添加此路径
)

target_include_directories(kutacc_comm_test
    PRIVATE
    ${GTEST_INCLUDE_DIRS}    # GTest头文件
    ${KUTACC_INCLUDE_DIR}    # kutacc头文件
    ${CMAKE_SOURCE_DIR}      # 若测试代码有本地头文件，添加此路径
)

# ========== 7. 链接依赖库（核心！顺序：标准库→kutacc→GTest→线程库） ==========
target_link_libraries(kutacc_cal_test
    PRIVATE
    stdc++                   # libc++标准库（Clang必需）
    ${KUTACC_LIB}            # kutacc业务库
    ${GTEST_LIBRARIES}       # GTest核心库
    ${GTEST_MAIN_LIBRARIES}  # GTest自带的main函数（无需自己写main）
    pthread                  # GTest依赖线程库，必须加
)

target_link_libraries(kutacc_comm_test
    PRIVATE
    stdc++                   # libc++标准库（Clang必需）
    ${KUTACC_LIB}            # kutacc业务库
    ${GTEST_LIBRARIES}       # GTest核心库
    ${GTEST_MAIN_LIBRARIES}  # GTest自带的main函数（无需自己写main）
    pthread                  # GTest依赖线程库，必须加
)

# ========== 8. 兜底：强制目标编译标准（避免覆盖） ==========
set_target_properties(kutacc_cal_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

set_target_properties(kutacc_comm_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# ========== 9. 安装 ==========
set(KUTACC_TEST_INSTALL_BIN_PATH ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS kutacc_cal_test DESTINATION ${KUTACC_TEST_INSTALL_BIN_PATH})
install(TARGETS kutacc_comm_test DESTINATION ${KUTACC_TEST_INSTALL_BIN_PATH})
