#!/bin/bash

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

# Function to get all dependencies of an executable
function get_dependencies() {
    local path=$1
    for dep in $(ldd $path | awk '{print $3}' | grep -v 'not a dynamic executable'); do
        if [ -f $dep ]; then
            local resolved_dep=$(readlink -f $dep)
            if ! [[ " ${dependencies[*]} " =~ " ${dep} " ]]; then
                dependencies+=("$dep")
                get_dependencies $resolved_dep
            fi
        fi
    done
}

# Function to replace needed libraries in an executable
function replace_needed() {
    local elf_path=$1
    local numa_lib_path=$2
    local elf_dependencies=($(patchelf --print-needed $elf_path))
    for dep in "${elf_dependencies[@]}"; do
        if [ "$dep" == "ld-linux-aarch64.so.1" ]; then
            continue
        fi

        so_real_name=${dependencies_hash[$dep]}
        if [ -z "$so_real_name" ]; then
            echo "Error: Dependency $dep not found. $elf_path"
            exit 1
        fi

        patchelf --replace-needed "$dep" "$numa_lib_path/$so_real_name" $elf_path
    done
}

# Get the number of duplications
if [ -n "$2" ]; then
    numa_node_num=$2
else
    numa_node_num=$(numactl --hardware | grep "available:" | awk '{print $2}')
fi

# Get the path of the application
app_path=$1
if [ ! -f "$app_path" ]; then
    echo "Error: Application $app_path not found."
    exit 1
fi
app_name=$(basename $app_path)

# Declare and initialize global variables
declare -g dependencies=()
get_dependencies $app_path
echo ${dependencies[@]} > /dev/null

declare -A dependencies_hash
for dep in "${dependencies[@]}"; do
    so_name=$(basename $dep)
    dependencies_hash[$so_name]=$(basename $(readlink -f $dep))
done

# Copy the application to each NUMA node and replace dependencies
rm -rf numa_duplication
for i in $(seq 0 $((numa_node_num-1)))
do
    path_numa="./numa_duplication/numa_$i"
    path_numa_lib="./numa_duplication/numa_$i/lib"
    mkdir -p $path_numa
    mkdir -p $path_numa_lib
    cp $app_path $path_numa
    replace_needed "$path_numa/$app_name" "$path_numa/lib"
    for so_item in "${dependencies[@]}"
    do

        resolved_dep=$(readlink -f $so_item)
        cp $resolved_dep $path_numa_lib
        so_name=$(basename $resolved_dep)
        replace_needed "$path_numa_lib/$so_name" "$path_numa_lib"
    done
done