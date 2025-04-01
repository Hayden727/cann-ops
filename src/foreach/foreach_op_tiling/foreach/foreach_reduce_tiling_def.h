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
 * \file foreach_reduce_tiling_def.h
 * \brief
 */
 
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_DEF_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_DEF_H_

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;
struct ForeachNormCompileInfo {
    uint32_t coreNum;
};
BEGIN_TILING_DATA_DEF(ForeachReduceTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalTensorCount);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 256, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, tensorEndOffsetList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, tensorMiddleCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, tensorMiddleStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, coreMiddleOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachNorm, ForeachReduceTilingData)
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_REDUCE_DEF_H_
