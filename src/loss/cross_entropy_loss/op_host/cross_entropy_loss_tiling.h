/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file cross_entropy_loss_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CROSS_ENTROPY_LOSS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CROSS_ENTROPY_LOSS_H

#include <iostream>
#include <cstring>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossEntropyLossTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, targetNum);
  TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, frontBatchNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailBatchNum);
  TILING_DATA_FIELD_DEF(uint64_t, inputUbSize);
  TILING_DATA_FIELD_DEF(uint64_t, castTmpBufByte);
  TILING_DATA_FIELD_DEF(uint64_t, lnTmpBufSize);
  TILING_DATA_FIELD_DEF(uint64_t, weightTmpBufSize);
  TILING_DATA_FIELD_DEF(uint64_t, weight4SmoothingBufSize);
  TILING_DATA_FIELD_DEF(uint64_t, totalTmpBufByte);
  TILING_DATA_FIELD_DEF(uint64_t, ubLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubTailNum);
  TILING_DATA_FIELD_DEF(uint64_t, vecLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, vecTailNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailVecLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailVecTailNum);
  TILING_DATA_FIELD_DEF(uint64_t, reduction);
  TILING_DATA_FIELD_DEF(int64_t, ignoreIndex);
  TILING_DATA_FIELD_DEF(float, labelSmoothing);
  TILING_DATA_FIELD_DEF(uint32_t, defaultWeight);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossEntropyLoss, CrossEntropyLossTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CROSS_ENTROPY_LOSS_H