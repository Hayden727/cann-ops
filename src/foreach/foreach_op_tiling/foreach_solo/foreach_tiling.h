/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_H_

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;
struct ForeachSoloCompileInfo {
    uint32_t coreNum;
};

BEGIN_TILING_DATA_DEF(ForeachSoloTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, inputsTensorUbSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachMulScalarInplace, ForeachSoloTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLogInplace, ForeachSoloTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSubListInplace, ForeachSoloTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulListInplace, ForeachSoloTilingData)
REGISTER_TILING_DATA_CLASS(ForeachDivListInplace, ForeachSoloTilingData)
}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_H_
