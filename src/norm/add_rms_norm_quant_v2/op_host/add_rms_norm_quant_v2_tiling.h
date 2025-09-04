/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file add_rms_norm_quant_v2_tiling.h
 * \brief Tiling data structures for the AddRmsNormQuantV2 operator.
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_QUANT_V2_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_QUANT_V2_H_

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AddRmsNormQuantV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, numRow);
TILING_DATA_FIELD_DEF(uint32_t, numCol);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, rowFactor);
TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(uint32_t, hasZeroPoints1);
TILING_DATA_FIELD_DEF(uint32_t, hasBias);
END_TILING_DATA_DEF;

struct AddRmsNormQuantV2CompileInfo {
  uint32_t totalCoreNum = 0;
  uint64_t totalUbSize = 0;
  platform_ascendc::SocVersion socVersion =
      platform_ascendc::SocVersion::ASCEND910B;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context) {
  return context->GetCompiledInfo<T>();
}

REGISTER_TILING_DATA_CLASS(AddRmsNormQuantV2, AddRmsNormQuantV2TilingData)

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_QUANT_V2_H_