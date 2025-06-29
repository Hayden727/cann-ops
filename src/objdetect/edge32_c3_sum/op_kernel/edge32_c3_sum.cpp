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
 * @file edge32_c3_sum.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

class KernelEdge32C3Sum
{
public:
    __aicore__ inline KernelEdge32C3Sum() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR y, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->tileDataNum = 14688;
        this->processDataNum = 14688;
        x1Gm.SetGlobalBuffer((__gm__ uint8_t *)x1);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        // pipe->InitBuffer(inQueueSum, BUFFER_NUM, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(tmpBuf, this->tileDataNum * sizeof(half));
        pipe->InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(sumBuf, this->tileDataNum * sizeof(float));
        sumLocal = sumBuf.Get<float>();
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        int coreNum = GetBlockIdx();
        int id = coreNum * 250 + 7;
        int endId = id + 250;
        if(coreNum > 7){
            id = 2000 + (coreNum - 8) * 203;
            endId = id + 203;
        }
        if(coreNum == 14){
            endId = id + 206;
        }
        ProcessFirst(id - 7);
        for(int i = id - 6; i <= id + 8; i++)
        {
            ProcessSum(i);
        }
        for (int32_t i = id; i < endId; i++)
        {
            Compute(i);
            CopyOut(i);
            ProcessUpdate(i + 9);
        }
    }

private:
    __aicore__ inline void ProcessFirst(int id){
        LocalTensor<uint8_t> xLocal = inQueueX1.AllocTensor<uint8_t>();
        DataCopy(xLocal, x1Gm[id * 14661], this->processDataNum);
        inQueueX1.EnQue(xLocal);
        
        xLocal = inQueueX1.DeQue<uint8_t>();
        LocalTensor<half> tmpLocal = tmpBuf.Get<half>();
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(sumLocal, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        inQueueX1.FreeTensor(xLocal);
    }
    __aicore__ inline void ProcessSum(int id){
        LocalTensor<uint8_t> x1Local = inQueueX1.AllocTensor<uint8_t>();
        DataCopy(x1Local, x1Gm[id * 14661], this->processDataNum);
        inQueueX1.EnQue(x1Local);
        
        x1Local = inQueueX1.DeQue<uint8_t>();
        LocalTensor<half> tmpLocal = tmpBuf.Get<half>();
        LocalTensor<float> tmpLocal2 = tmpBuf2.Get<float>();
        Cast(tmpLocal, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tmpLocal2, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        Add(sumLocal, sumLocal, tmpLocal2, this->processDataNum);
        inQueueX1.FreeTensor(x1Local);
    }
    __aicore__ inline void ProcessUpdate(int id){
        LocalTensor<uint8_t> x1Local = inQueueX1.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> x2Local = inQueueX2.AllocTensor<uint8_t>();
        
        DataCopy(x1Local, x1Gm[(id - 16) * 14661], this->processDataNum);
        DataCopy(x2Local, x1Gm[id * 14661], this->processDataNum);
        
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
        
        x1Local = inQueueX1.DeQue<uint8_t>();
        x2Local = inQueueX2.DeQue<uint8_t>();
        LocalTensor<half> tmpLocal = tmpBuf.Get<half>();
        LocalTensor<float> tmpLocal2 = tmpBuf2.Get<float>();
        
        Cast(tmpLocal, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tmpLocal2, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        
        Sub(sumLocal, sumLocal, tmpLocal2, this->processDataNum);
        
        Cast(tmpLocal, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tmpLocal2, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        
        Add(sumLocal, sumLocal, tmpLocal2, this->processDataNum);
        
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        LocalTensor<half> tmpLocal = tmpBuf.Get<half>();
        LocalTensor<float> tmpLocalFP32 = tmpBuf2.Get<float>();
        
        Muls(tmpLocalFP32, sumLocal, 0.0625f, this->processDataNum);
        Cast(tmpLocalFP32.template ReinterpretCast<int32_t>(), tmpLocalFP32, RoundMode::CAST_FLOOR, this->processDataNum);
        Cast(tmpLocalFP32, tmpLocalFP32.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, this->processDataNum);        
        Cast(tmpLocal, tmpLocalFP32, RoundMode::CAST_NONE, this->processDataNum);
        Cast(yLocal, tmpLocal, RoundMode::CAST_FLOOR, this->processDataNum);
        
        outQueueY.EnQue<uint8_t>(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t id)
    {
        int offset = id * 14661;
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        DataCopy(yGm[offset], yLocal, 14656);
        for(int i = 14656; i < 14661; i++){
            yGm.SetValue(offset + i, yLocal.GetValue(i));
        }
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE>(yGm[offset + 14656]);
        if((offset + 14656) / 64 != (offset + 14660) /64){
          DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE>(yGm[offset + 14660]);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf, tmpBuf2, sumBuf;
    GlobalTensor<uint8_t> x1Gm;
    GlobalTensor<uint8_t> yGm;
    LocalTensor<float> sumLocal;
    int64_t tileDataNum;
    int64_t processDataNum;
};
extern "C" __global__ __aicore__ void edge32_c3_sum(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipeIn;
    KernelEdge32C3Sum op;
    op.Init(input, out, &pipeIn);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    op.Process();
}
