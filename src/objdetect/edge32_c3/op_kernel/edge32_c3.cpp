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
 * @file edge32_c3.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

class KernelEdge32C3
{
public:
    __aicore__ inline KernelEdge32C3() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR y, TPipe *pipeIn, int64_t smallCoreDataNum,
                                int64_t bigCoreDataNum, int64_t finalBigTileNum,
                                int64_t finalSmallTileNum, int64_t tileDataNum,
                                int64_t smallTailDataNum, int64_t bigTailDataNum,
                                int64_t tailBlockNum)
    {
        uint32_t coreNum = GetBlockIdx();
        
        globalBufferIndex = bigCoreDataNum * GetBlockIdx() + 73305;
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
        x1Gm.SetGlobalBuffer((__gm__ uint8_t *)x1);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y + globalBufferIndex, this->coreDataNum);
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * 3 * sizeof(uint8_t));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * 3 * sizeof(uint8_t));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(tmpBuf1, this->tileDataNum * 3 * sizeof(half));
        pipe->InitBuffer(tmpBuf2, this->tileDataNum * 3 * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        this->processDataNumLarge = this->tileDataNum * 3;
        for (int32_t i = 0; i < loopCount; i++)
        {
            if (i == this->tileNum - 1)
            {
                this->processDataNum = this->tailDataNum;
                this->processDataNumLarge = this->tailDataNum * 3;
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
        
        DataCopy(x1Local, x1Gm[(globalBufferIndex + progress * this->tileDataNum) * 3 - 3], this->processDataNumLarge);
        
        DataCopy(x2Local, x1Gm[(globalBufferIndex + progress * this->tileDataNum) * 3 + 3], this->processDataNumLarge);
        
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<uint8_t> x1Local = inQueueX1.DeQue<uint8_t>();
        LocalTensor<uint8_t> x2Local = inQueueX2.DeQue<uint8_t>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        LocalTensor<half> tmp1Local = tmpBuf1.Get<half>();
        LocalTensor<half> tmp2Local = tmpBuf2.Get<half>();
        LocalTensor<half> fLocal = tmp2Local;
        Cast(tmp1Local, x1Local, RoundMode::CAST_NONE, this->processDataNumLarge);
        Cast(tmp2Local, x2Local, RoundMode::CAST_NONE, this->processDataNumLarge);
        Sub(tmp1Local, tmp1Local, tmp2Local, this->processDataNumLarge);
        Abs(tmp1Local, tmp1Local, this->processDataNumLarge);
        int signleLength = this->processDataNum / 2;
        int twoSignleLength = signleLength + signleLength;
        int threeSignleLength = twoSignleLength + signleLength;
        int largeLength = threeSignleLength + signleLength;
        TransposeParamsExt transposeParams;
        transposeParams.nSize = 2;
        transposeParams.cSize = 3;
        transposeParams.hSize = signleLength;
        transposeParams.wSize = 1;
        transposeParams.transposeType = TransposeType::TRANSPOSE_NHWC2NCHW;
        Transpose(tmp2Local, tmp1Local, x1Local, transposeParams);
        // 2 3 3264 --- 2 9792

        Max(fLocal, tmp2Local, tmp2Local[signleLength], signleLength);
        Max(fLocal, fLocal, tmp2Local[twoSignleLength], signleLength);
        
        Max(fLocal[signleLength], tmp2Local[threeSignleLength], tmp2Local[largeLength], signleLength);
        Max(fLocal[signleLength], fLocal[signleLength], tmp2Local[largeLength + signleLength], signleLength);
        
        Mins(fLocal, fLocal, (half)255.0, this->processDataNum);
        Cast(yLocal, fLocal, RoundMode::CAST_NONE, this->processDataNum);
        int offset = globalBufferIndex + progress * this->tileDataNum;
        int start_id = offset / 4887;
        int end_id = (offset + this->processDataNum) / 4887 + 1;
        for(int i = start_id, index =  start_id * 4887; i <= end_id; i++, index += 4887){
            int real_index = index - offset;
            if(real_index - 1 >= 0 && real_index - 1 < this->processDataNum){
                yLocal.SetValue(real_index - 1, static_cast<uint8_t>(0));
            }
            if(real_index >= 0 && real_index < this->processDataNum){
                yLocal.SetValue(real_index, static_cast<uint8_t>(0));
            }
        }
        if(GetBlockIdx() == 14 && progress == this->tileNum - 1){
            for(int i = processDataNum; i + offset >= 16733087; i--){
                yLocal.SetValue(i, static_cast<uint8_t>(0));
            }
        }

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
    TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2, indexBuf;
    GlobalTensor<uint8_t> x1Gm;
    GlobalTensor<uint8_t> yGm;
    int64_t coreDataNum;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    int64_t processDataNumLarge;
    int64_t globalBufferIndex;
};
extern "C" __global__ __aicore__ void edge32_c3(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipeIn;
    KernelEdge32C3 op;
    op.Init(input, out, &pipeIn, tiling_data.smallCoreDataNum, 
        tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
        tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
        tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
        tiling_data.tailBlockNum);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    op.Process();
}