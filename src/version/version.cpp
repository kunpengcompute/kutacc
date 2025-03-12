/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuTACC is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "kutacc.h"

const kutacc_version_t g_version = {
    .product_name = "Kunpeng HPCKit", 
    .product_version = "26.1.RC1",
    .component_name = "KuTACC",
    .component_version = "26.1.RC1",
    .component_abi_version = "1",
#if defined(__clang__)
    .component_appendinfo = "bisheng",
#elif defined(__GNUC__)
    .component_appendinfo = "gcc",
#endif
};

int kutacc_get_version(kutacc_version_t *version) 
{
    if (version == nullptr) {
        return KUTACC_ERROR;
    }
    *version = g_version;
    return KUTACC_OK;
}
