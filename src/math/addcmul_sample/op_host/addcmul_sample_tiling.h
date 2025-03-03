/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADDCMUL_SAMPLE_TILING_H
#define ADDCMUL_SAMPLE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddcmulSampleTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  TILING_DATA_FIELD_DEF(uint32_t, input_data_length);
  TILING_DATA_FIELD_DEF(uint32_t, x1_length);
  TILING_DATA_FIELD_DEF(uint32_t, x2_length);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddcmulSample, AddcmulSampleTilingData)
}
// namespace optiling
#endif // ADDCMUL_SAMPLE_TILING_H