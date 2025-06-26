/**
* @file fill.cpp
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/


#include "kernel_operator.h"
#include "fill.h"
#include "fillint64.h"
#include "kernel_operator.h"
// tensor num for each queue
extern "C" __global__ __aicore__ void fill(GM_ADDR dims, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    if(TILING_KEY_IS(0))
    {
        if constexpr (std::is_same_v<DTYPE_VALUE, bool> || std::is_same_v<DTYPE_VALUE, int8_t>) {
            KernelFill<int8_t, false> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_VALUE, int64_t>) {
            KernelFill1_INT64<false> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);
            op.Process();
        } else {
            KernelFill<DTYPE_VALUE, false> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);
            op.Process(); 
        }
    }
     else if(TILING_KEY_IS(1))
    {
        if constexpr (std::is_same_v<DTYPE_VALUE, bool> || std::is_same_v<DTYPE_VALUE, int8_t>) {
            KernelFill<int8_t, true> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_VALUE, int64_t>) {
            KernelFill1_INT64<true> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);
            op.Process();
        } else {
            KernelFill<DTYPE_VALUE, true> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);
            op.Process(); 
        }
    }

}