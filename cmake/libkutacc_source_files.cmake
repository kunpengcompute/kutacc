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

# find all source files to compiler kutacc library
function(libkutacc_source_files src_files)
    file(GLOB_RECURSE kutacc_srcs
        glu/*.cpp
        alphafold3/*.cpp
        async/*.cpp
        comm/*.cpp
        core/*.cpp
        core/matmul/*.cpp
        core/matmul/*.S
        embedding/*.cpp
        math/*.cpp
        rope/*.cpp
        silu_mul/*.cpp
        mem_op/*.cpp
        tensor/*.cpp
        rmsnorm/*.cpp
        utils/*.cpp
        version/*.cpp
        attention/*.cpp
        fusedmoe/*.cpp
        mtp_op/*.cpp
        sample/*.cpp
        moe/*.cpp
        pp/*.cpp
        mt/*.cpp
        core/kurmcl/*.cpp
        mul_scalar_add/*.cpp
    )
    set(${src_files} ${kutacc_srcs} PARENT_SCOPE)
endfunction()

# install header
function(libkutacc_header_install)
    #install kutacc.h
    file(GLOB BASE_HEADER ${KUTACC_ROOT_DIR}/include/*.h)

    install(FILES ${BASE_HEADER} DESTINATION ${KUTACC_INSTALL_INCLUDEDIR}
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)
endfunction()