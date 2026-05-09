# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# KuTACC is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#        http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

function(libkutacc_source_files src_files)
    file(GLOB_RECURSE kutacc_srcs
        attention/*.cpp
        comm/*.cpp
        linear/*.cpp
        tensor/*.cpp
        utils/*.c
        utils/*.cpp
        version/*.cpp
        normalization/*.cpp
        math/*.cpp
        activation/*.cpp
    )

    set(${src_files} ${kutacc_srcs} PARENT_SCOPE)
endfunction()

function(libkutacc_header_install)

    file(GLOB BASE_HEADER ${KUTACC_ROOT_DIR}/include/*.h)

    install(FILES ${BASE_HEADER} DESTINATION ${KUTACC_INSTALL_INCLUDEDIR} PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)
endfunction()
