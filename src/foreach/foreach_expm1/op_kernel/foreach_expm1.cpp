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
 * \file foreach_expm1.cpp
 * \brief
 */

 #include "kernel_operator.h"
 #include "lib/math/kernel_operator_asin_intf.h"
 
 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_implict_output.h"
 
using namespace AscendC;
using namespace Common::OpKernel;

template <typename T>
__aicore__ void Expm1Adapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& uValue) {
    T scalarVal = T(-1);
    pipe_barrier(PIPE_V);
    Exp(dstLocal, srcLocal, uValue);
    pipe_barrier(PIPE_V);
    Adds(dstLocal, srcLocal, scalarVal, uValue);
}

extern "C" __global__ __aicore__ void foreach_expm1(GM_ADDR x,  GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachImplictOutput<half, half, Expm1Adapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachImplictOutput<float, float, Expm1Adapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } 
    #if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachImplictOutput<bfloat16_t, float, Expm1Adapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
    #endif
}
