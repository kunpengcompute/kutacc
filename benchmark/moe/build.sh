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

if [ -z $KUTACC_INSTALL_ROOT ]; then
    export KUTACC_INSTALL_ROOT=$(dirname $(dirname $PWD))/install
    echo $KUTACC_INSTALL_ROOT
fi
export LD_LIBRARY_PATH=$KUTACC_INSTALL_ROOT/lib:$LD_LIBRARY_PATH

rm -rf build
cmake -B build
cmake --build build -j

bash ../install-numa-duplication.sh build/test_dispatch_combine