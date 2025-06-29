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
 * @file edge_sub.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelEdgeSub
{
public:
    __aicore__ inline KernelEdgeSub() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TPipe *pipeIn, int64_t smallCoreDataNum,
                                int64_t bigCoreDataNum, int64_t finalBigTileNum,
                                int64_t finalSmallTileNum, int64_t tileDataNum,
                                int64_t smallTailDataNum, int64_t bigTailDataNum,
                                int64_t tailBlockNum)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum)
        {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->pipe = pipeIn;
        x1Gm.SetGlobalBuffer((__gm__ uint8_t *)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ uint8_t *)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y + globalBufferIndex, this->coreDataNum);
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(tmp1Buf, this->tileDataNum * sizeof(half));
        pipe->InitBuffer(tmp2Buf, this->tileDataNum * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++)
        {
            if (i == this->tileNum - 1)
            {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:

    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<uint8_t> x1Local = inQueueX1.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> x2Local = inQueueX2.AllocTensor<uint8_t>();
        
        DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<uint8_t> x1Local = inQueueX1.DeQue<uint8_t>();
        LocalTensor<uint8_t> x2Local = inQueueX2.DeQue<uint8_t>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        LocalTensor<half> tmp1Local = tmp1Buf.Get<half>();
        LocalTensor<half> tmp2Local = tmp2Buf.Get<half>();

        Cast(tmp1Local, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tmp2Local, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        Sub(tmp1Local, tmp2Local, tmp1Local, this->processDataNum);
        Adds(tmp1Local, tmp1Local, (half)-10, this->processDataNum);
        Maxs(tmp1Local, tmp1Local, (half)0, this->processDataNum);
        Cast(yLocal, tmp1Local, RoundMode::CAST_ROUND, this->processDataNum);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1Buf, tmp2Buf;
    GlobalTensor<uint8_t> x1Gm, x2Gm;
    GlobalTensor<uint8_t> yGm;
    int64_t coreDataNum;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
};

extern "C" __global__ __aicore__ void edge_sub(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipeIn;
    KernelEdgeSub op;
    op.Init(input, other, out, &pipeIn, tiling_data.smallCoreDataNum, 
        tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
        tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
        tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
        tiling_data.tailBlockNum);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    op.Process();
}