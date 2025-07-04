/* 
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "kernel_operator.h"
#include "copy_kl.h"
#include "type_kl.h"
#include "sync_kl.h"
using namespace AscendC;

#include "trunc_.cpp"
#include "trunc_f32.cpp"

extern "C" __global__ __aicore__ void trunc(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe; // Pipeline
    GET_TILING_DATA(tiling_data, tiling); // Tiling data

    uint32_t Len;
    uint32_t fNum, fLen, tLen;
    {
        bool is_former = GetBlockIdx() < tiling_data.fNum;
        Len = tiling_data.Len;
        fNum = tiling_data.fNum;
        fLen = tiling_data.fLen;
        tLen = tiling_data.tLen;
    }

     // 初始化&计算
    if constexpr(IS_TYPE(DTYPE_INPUT_X, float)){
        KernelTruncF32 op;
        op.Init(&pipe,
            input_x, output_y, 
            fLen, fNum, tLen, 
            Len // 核内切分参数
            
        );
        op.Process();
    }else{
        KernelTrunc<DTYPE_INPUT_X, DTYPE_OUTPUT_Y> op;
        op.Init(&pipe,
            input_x, output_y, 
            fLen, fNum, tLen, 
            Len // 核内切分参数
            
        );
        op.Process();
    }
}
