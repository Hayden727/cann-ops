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
 * \file mulitensor_apply_tilingdata.h
 * \brief
 */
#ifndef ASL_OPS_CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_FOREACH_OP_MULTITENSOR_APPLY_TILINGDATA_H
#define ASL_OPS_CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_FOREACH_OP_MULTITENSOR_APPLY_TILINGDATA_H

#include "register/tilingdata_base.h"

namespace optiling {
// TensorListMetadata has to be less than 4KB - the limit for kernel launch argument
constexpr int DEPTH_TO_MAX_TENSORS[5] = {256, 128, 84, 64, 50};
constexpr int MAX_CORE_COUNT = 48;

BEGIN_TILING_DATA_DEF(MultiTensorApplyTilingData)
    TILING_DATA_FIELD_DEF_ARR(int64_t, DEPTH_TO_MAX_TENSORS[0], tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, listStartIdx);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, listEndIdx);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorStartOffset);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorEndOffset);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, opCode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachBinaryOp, MultiTensorApplyTilingData)
struct Tiling4ForeachBinaryOpCompileInfo {
    uint32_t coreNum;
};

}

#endif // ASL_OPS_CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_FOREACH_OP_MULTITENSOR_APPLY_TILINGDATA_H