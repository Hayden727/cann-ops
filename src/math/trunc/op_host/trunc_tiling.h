/* 
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef TRUNC_TILING_H
#include "register/tilingdata_base.h"

#define MAX_INPUT_DIM 8
#define BROADCAST_TILING_KEY 10

namespace optiling {
BEGIN_TILING_DATA_DEF(TruncTilingData)
     // 核内切分参数
    TILING_DATA_FIELD_DEF(uint32_t, Len);
    TILING_DATA_FIELD_DEF(uint32_t, fNum);
    TILING_DATA_FIELD_DEF(uint32_t, fLen);
    TILING_DATA_FIELD_DEF(uint32_t, tLen);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Trunc, TruncTilingData)

}
#endif // TRUNC_TILING_H