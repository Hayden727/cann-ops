/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifndef BASE_KUNLUN_H
#define BASE_KUNLUN_H

#include <cstdio>
#include <cstdint>
#include <string>
#include <set>
#include <vector>
#include <functional>
#include <algorithm>

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

inline void CHECK(bool BOOL_CONDITION, const char* TAG, const char* MSG){                                                        \
    if(!(BOOL_CONDITION)){                                                                        \
        exit(-1);                                                                               \
    }                                                                                           \
}

namespace kunlun{
    template <typename T>
    using List = std::vector<T>;

    template <typename T, typename F>
    using Pair = std::pair<T,F>;

    using Int = uint32_t;


    using gert::TilingContext;
    
    inline constexpr Int INT_MAX = Int(0) - 1;

} // namespace kunlun


#endif// BASE_KUNLUN_H
