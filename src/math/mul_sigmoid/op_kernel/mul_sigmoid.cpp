/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file mul_sigmoid.cpp
 */

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "mul_sigmoid.h"

extern "C" __global__ __aicore__ void 
mul_sigmoid(GM_ADDR x1, GM_ADDR x2, GM_ADDR out_buf, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data_in, tiling);

  if (TILING_KEY_IS(1)) {
    MulSigmoid op;  
    op.init(x1, x2, out_buf, workspace, tiling_data_in);
    op.process();
  }
}