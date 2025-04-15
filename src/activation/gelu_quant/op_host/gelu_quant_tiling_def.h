/* *
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

/* !
 * \file gelu_quant_tiling_def.h
 * \brief
 */

#ifndef GELU_QUANT_TILING_DEF_H
#define GELU_QUANT_TILING_DEF_H

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluQuantTilingData)

TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, endAxisLen);
TILING_DATA_FIELD_DEF(int64_t, endAxisLenAligned);
TILING_DATA_FIELD_DEF(int64_t, rowOuter);
TILING_DATA_FIELD_DEF(int64_t, colOuter);
TILING_DATA_FIELD_DEF(uint32_t, rowInner);
TILING_DATA_FIELD_DEF(uint32_t, rowTail);
TILING_DATA_FIELD_DEF(uint32_t, colInner);
TILING_DATA_FIELD_DEF(uint32_t, colTail);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, coexistentNodeNum);
TILING_DATA_FIELD_DEF(uint32_t, coexistentNodeElementNum);
TILING_DATA_FIELD_DEF(uint32_t, quantMode);
TILING_DATA_FIELD_DEF(uint32_t, approximate);
TILING_DATA_FIELD_DEF(uint32_t, inputScaleType);
TILING_DATA_FIELD_DEF(uint32_t, inputOffsetType);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluQuant, GeluQuantTilingData)
} // namespace optiling
#endif // GELU_QUANT_TILING_DEF_H