#!/bin/bash
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
set -e

print_help(){
    echo "build.sh -- A tool for building KUTACC"
    echo "Parameters:"
    echo "--compiler=[gcc[default]|clang]                           |   build with different compiler"
    echo "--cleanup=[off|on(default)]                               |   cleanup and install path before build"
    echo "--build_kind=[src(default)|test|cida|fuzz]                |   build src or test or cida or fuzz suite"   
    echo "--build_type=[Release(default)|Debug|RelWithDebInfo]      |   build release or debug version"
    echo "--install_path=absolute_path                              |   an absolute path for creating install folder"
    echo "--help                                                    |   print this help message"
    return
}

export KUTACC_PROJ_PATH=$(cd "$(dirname ${0})"; pwd -P)
export KUTACC_CLEANUP="on"
export KUTACC_COMPILER="clang"
export KUTACC_BUILD_TYPE="Release"
export KUTACC_BUILD_KIND="src"
export KUTACC_INSTALL_PATH="$KUTACC_PROJ_PATH/install"
export KUTACC_GENERATOR="Unix Makefiles"


function install()
{
    if [[ "$KUTACC_GENERATOR" == "Unix Makefiles" ]]; then
        make -j && make install
    elif [[ "$KUTACC_GENERATOR" == "Ninja" ]]; then
        ninja && ninja install
    fi
}

function build_src()
{
    echo ""
    echo "build src"
    export LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_PATH
    local source_path=$KUTACC_PROJ_PATH
    local build_path=$KUTACC_PROJ_PATH/build/src
    mkdir -p ${build_path} && cd ${build_path}
    cmake -G "${KUTACC_GENERATOR}"                      \
        -S ${source_path} -B ${build_path}              \
        -DCMAKE_C_COMPILER=${KUTACC_C_COMPILER}         \
        -DCMAKE_CXX_COMPILER=${KUTACC_CXX_COMPILER}     \
        -DCMAKE_BUILD_TYPE=${KUTACC_BUILD_TYPE}         \
        -DKUTACC_BUILD_KIND=${KUTACC_BUILD_KIND}        \
        -DCMAKE_INSTALL_PREFIX=${KUTACC_INSTALL_PATH}
    install
}

function build_test()
{
    echo ""
    echo "build test"
    local source_path=$KUTACC_PROJ_PATH/test/functest
    local build_path=$KUTACC_PROJ_PATH/build/functest
    mkdir -p ${build_path} && cd ${build_path}
    cmake -G "${KUTACC_GENERATOR}"                      \
        -S ${source_path} -B ${build_path}              \
        -DCMAKE_C_COMPILER=${KUTACC_MPI_C_COMPILER}     \
        -DCMAKE_CXX_COMPILER=${KUTACC_MPI_CXX_COMPILER} \
        -DCMAKE_INSTALL_PREFIX=${KUTACC_INSTALL_PATH}
    install
}

function build_fuzz()
{
    echo ""
    echo "build fuzz"
    local source_path=$KUTACC_PROJ_PATH/test/fuzz
    local build_path=$KUTACC_PROJ_PATH/build/fuzz
    mkdir -p ${build_path} && cd ${build_path}
    cmake -G "${KUTACC_GENERATOR}"                      \
        -S ${source_path} -B ${build_path}              \
        -DCMAKE_C_COMPILER=${KUTACC_C_COMPILER}         \
        -DCMAKE_CXX_COMPILER=${KUTACC_CXX_COMPILER}     \
        -DCMAKE_BUILD_TYPE=${KUTACC_BUILD_TYPE}         \
        -DCMAKE_INSTALL_PREFIX=${KUTACC_INSTALL_PATH}
    install
}

function cleanup()
{
    echo ""
    echo "do cleanup"
    local build_path=$KUTACC_PROJ_PATH/build
    if [ -d "${build_path}" ];then
        rm -rf ${build_path}
        echo "cleanup ${build_path}"
    fi
    if [ -d "${KUTACC_INSTALL_PATH}" ];then
        rm -rf ${KUTACC_INSTALL_PATH}
        echo "cleanup ${KUTACC_INSTALL_PATH}"
    fi
}

function set_compiler()
{
    case "$1" in
        gcc)
            export CC=$(which gcc)
            export CXX=$(which g++)
            KUTACC_C_COMPILER=$(which gcc)
            KUTACC_CXX_COMPILER=$(which g++)
            KUTACC_MPI_C_COMPILER=$(which mpicc)
            KUTACC_MPI_CXX_COMPILER=$(which mpic++)
            ;;
        clang)
            export CC=$(which clang)
            export CXX=$(which clang++)
            KUTACC_C_COMPILER=$(which clang)
            KUTACC_CXX_COMPILER=$(which clang++)
            KUTACC_MPI_C_COMPILER=$(which mpicc)
            KUTACC_MPI_CXX_COMPILER=$(which mpic++)
            ;;
        *)
            echo "Unsupported compiler $1."
            exit 1
            ;;
    esac
    KUTACC_COMPILER="$1"
}

function set_build_type()
{
    case "$1" in
        r|release|Release)
            KUTACC_BUILD_TYPE="Release"
            ;;
        d|debug|Debug)
            KUTACC_BUILD_TYPE="Debug"
            ;;
        rwdi|RelWithDebInfo)
            KUTACC_BUILD_TYPE="RelWithDebInfo"
            ;;
        *)
            echo "Unsupported build type: $1."
            exit 1
            ;;
    esac
}

function parse_args()
{
    for i in "$@"; do
        case "$i" in
            --cleanup=*)
                if [[ "${i#*=}" == "off" ]]; then
                    KUTACC_CLEANUP="off"
                fi
                ;;
            --compiler=*)
                KUTACC_COMPILER="${i#*=}"
                ;;
            --install_path=*)
                KUTACC_INSTALL_PATH="${i#*=}"
                ;;
            --build_type=*)
                if [[ "${KUTACC_BUILD_KIND}" == "src" ]]; then
                    set_build_type "${i#*=}"
                fi
                ;;
            --build_kind=*)
                KUTACC_BUILD_KIND="${i#*=}"
                case "$KUTACC_BUILD_KIND" in
                    src)
                        ;;
                    test)
                        KUTACC_BUILD_TYPE="Debug"
                        ;;
                    fuzz)
                        KUTACC_BUILD_TYPE="Debug"
                        ;;
                    *)
                        echo "invalid build kind $KUTACC_BUILD_KIND"
                        exit 1;
                        ;;
                esac
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                echo "Unknown option: $i"
                echo ""
                exit 1
                ;;
        esac
    done
}

function main()
{
    echo "PROJ_PATH:    "   $KUTACC_PROJ_PATH
    echo "INSTALL_PATH: "   $KUTACC_INSTALL_PATH
    echo "BUILD_TYPE:   "   $KUTACC_BUILD_TYPE
    echo "BUILD_KIND:   "   $KUTACC_BUILD_KIND
    echo "GENERATOR:    "   $KUTACC_GENERATOR
    if [[ "${KUTACC_CLEANUP,,}" == "on" ]]; then
        cleanup
    fi
    set_compiler $KUTACC_COMPILER
    build_src
    case "$KUTACC_BUILD_KIND" in
        test)
            build_test
            ;;
        fuzz)
            build_fuzz
            ;;
    esac
}

parse_args $@
main
