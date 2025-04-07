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
 * @file reflection_pad3d_grad.cpp
 */
#include "reflection_pad3d_grad_mid.h"
#include "reflection_pad3d_grad_small.h"

extern "C" __global__ __aicore__ void reflection_pad3d_grad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(0)) {
        ReflectionPad3dGrad<float> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.SmallProcess();
    } else if(TILING_KEY_IS(1)) {
        ReflectionPad3dGrad<float> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.MidProcess();
    } else if(TILING_KEY_IS(2)) {
        ReflectionPad3dGrad<half> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.SmallProcess();
    } else if(TILING_KEY_IS(3)) {
        ReflectionPad3dGrad<half> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.MidProcess();
    } else if(TILING_KEY_IS(4)) {
        ReflectionPad3dGrad<bfloat16_t> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.SmallProcess();
    } else if(TILING_KEY_IS(5)) {
        ReflectionPad3dGrad<bfloat16_t> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.MidProcess();
    }
}