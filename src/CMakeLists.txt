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

# Check debug mode
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ENABLE_KUTACC_DEBUG ON)
    set(ENABLE_KUTACC_ASSERT ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wfloat-equal -fno-common -march=armv8.2-a+bf16+sve -msve-vector-bits=512 -fPIC -fopenmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wfloat-equal -fno-common -march=armv8.2-a+bf16+sve -msve-vector-bits=512 -fPIC -fopenmp")
    add_compile_options(-O0 -g -gdwarf-4)
    message("-- Caution : using debug mode, performance is bad, just for develop")
elseif (CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+bf16+sve -msve-vector-bits=512 -fopenmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+bf16+sve -msve-vector-bits=512 -fopenmp")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-missing-field-initializers")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wfloat-equal -fno-common -Weffc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual -Wnon-virtual-dtor -Wdelete-non-virtual-dtor")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=2 -Wunused -Wswitch-default -Wundef -Wdate-time -Wshadow")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align -Wcast-qual -pipe -Wconversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wframe-larger-than=16384 -Wvla -fstack-protector-strong")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-prototypes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-designated-field-initializers")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-strong")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,-Bsymbolic -rdynamic -Wl,--no-undefined")
    add_compile_options(-O3)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -stdlib=libstdc++")

if (KUTACC_BUILD_KIND MATCHES "src")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 19.0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-designated-field-initializers")
    endif()
    if (KUTACC_BUILD_KIND MATCHES "test")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage")
    elseif (KUTACC_BUILD_KIND MATCHES "fuzz")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage -fsanitize-coverage=trace-pc")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage -fsanitize-coverage=trace-pc")
    endif()
endif()

if (CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2")
endif()

include_directories(${KUTACC_ROOT_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${KUTACC_ROOT_DIR}/src)

include(${KUTACC_CMAKE_INCLUDE_PATH}/libkutacc_source_files.cmake)

libkutacc_source_files(KUTACC_SRCS)
add_library(${LIBKUTACC_NAME} SHARED ${KUTACC_SRCS})

target_link_libraries(${LIBKUTACC_NAME} kutacc_core)
target_link_libraries(${LIBKUTACC_NAME} numa)
target_link_libraries(${LIBKUTACC_NAME} mpi)
target_link_libraries(${LIBKUTACC_NAME} kupl)

set_target_properties(${LIBKUTACC_NAME} PROPERTIES VERSION 1.26.1.RC1 SOVERSION 1)

#Install
libkutacc_header_install()
install(TARGETS ${LIBKUTACC_NAME} DESTINATION ${KUTACC_INSTALL_LIBDIR})
