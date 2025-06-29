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
 * @file edge_sub_dilate_ver_c1.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

class KernelEdgeSubDilateVerC1
{
public:
    __aicore__ inline KernelEdgeSubDilateVerC1() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR y, TPipe *pipeIn)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * 2101410;
        // 262144
        this->tileDataNum = 4896; // 11组
        this->processDataNum = 4896;
        this->pipe = pipeIn;
        x1Gm.SetGlobalBuffer((__gm__ uint8_t *)x1 + globalBufferIndex, 2101440);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y + globalBufferIndex, 2101440);
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(inQueueX3, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(inQueueX4, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(inQueueX5, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        pipe->InitBuffer(tmpBuf1, this->tileDataNum * sizeof(half));
        pipe->InitBuffer(tmpBuf2, this->tileDataNum * sizeof(half));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < 430; i++)
        {
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
        LocalTensor<uint8_t> x3Local = inQueueX3.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> x4Local = inQueueX4.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> x5Local = inQueueX5.AllocTensor<uint8_t>();
        int offset = progress * 4887;
        DataCopy(x1Local, x1Gm[offset], this->processDataNum);
        
        DataCopy(x2Local, x1Gm[offset + 1], this->processDataNum);
        
        DataCopy(x3Local, x1Gm[offset + 2], this->processDataNum);
        
        DataCopy(x4Local, x1Gm[offset + 3], this->processDataNum);
        
        DataCopy(x5Local, x1Gm[offset + 4], this->processDataNum);
        
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
        inQueueX3.EnQue(x3Local);
        inQueueX4.EnQue(x4Local);
        inQueueX5.EnQue(x5Local);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<uint8_t> x1Local = inQueueX1.DeQue<uint8_t>();
        LocalTensor<uint8_t> x2Local = inQueueX2.DeQue<uint8_t>();
        LocalTensor<uint8_t> x3Local = inQueueX3.DeQue<uint8_t>();
        LocalTensor<uint8_t> x4Local = inQueueX4.DeQue<uint8_t>();
        LocalTensor<uint8_t> x5Local = inQueueX5.DeQue<uint8_t>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();

        LocalTensor<half> tmp1Local = tmpBuf1.Get<half>();
        LocalTensor<half> tmp2Local = tmpBuf2.Get<half>();
        
        Cast(tmp1Local, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tmp2Local, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        Max(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
        Cast(tmp2Local, x3Local, RoundMode::CAST_NONE, this->processDataNum);
        Max(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
        Cast(tmp2Local, x4Local, RoundMode::CAST_NONE, this->processDataNum);
        Max(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
        Cast(tmp2Local, x5Local, RoundMode::CAST_NONE, this->processDataNum);
        Max(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
        Cast(yLocal, tmp1Local, RoundMode::CAST_NONE, this->processDataNum);
        PipeBarrier<PIPE_ALL>();
        
        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        inQueueX3.FreeTensor(x3Local);
        inQueueX4.FreeTensor(x4Local);
        inQueueX5.FreeTensor(x5Local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        int offset = progress * 4887;
        DataCopy(yGm[offset + 2], yLocal, 4864);
        
        for(int i = 4866; i < 4885; i++){
            yGm.SetValue(offset + i, yLocal.GetValue(i - 2));
        }
    
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE>(yGm[offset + 4866]);
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE>(yGm[offset + 4884]);
        
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2, inQueueX3, inQueueX4, inQueueX5;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2;
    GlobalTensor<uint8_t> x1Gm;
    GlobalTensor<uint8_t> yGm;
    int64_t tileDataNum;
    int64_t processDataNum;
};

extern "C" __global__ __aicore__ void edge_sub_dilate_ver_c1(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipeIn;
    KernelEdgeSubDilateVerC1 op;
    op.Init(input, out, &pipeIn);
    op.Process();
}