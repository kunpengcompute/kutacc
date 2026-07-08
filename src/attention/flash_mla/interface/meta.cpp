/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "kutacc.h"

#include "attention/flash_mla/utils.h"
#include "attention/flash_mla/config.h"
#include "attention/flash_mla/meta.h"

namespace kutacc {

void flash_mla_meta_create(FlashMLAMetaHandle &meta)
{
    meta = (FlashMLAMeta *)aligned_alloc(flash_mla::ALIGNMENT, sizeof(FlashMLAMeta));
    meta->tile_sched_meta = (flash_mla::TileSchedMeta *)aligned_alloc(flash_mla::ALIGNMENT,
        kutacc::get_thread_num() * sizeof(flash_mla::TileSchedMeta));
    meta->max_batch_size = 0;
    meta->num_splits = nullptr;
}

void flash_mla_meta_destory(FlashMLAMetaHandle meta)
{
    if (meta->num_splits != nullptr) {
        free(meta->num_splits);
    }
    free(meta->tile_sched_meta);
    free(meta);
}

} // namespace kutacc