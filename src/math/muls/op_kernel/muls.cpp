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
 * @file muls.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename TYPE_X>
class KernelMuls
{
public:
    __aicore__ inline KernelMuls() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
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
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X *)y + globalBufferIndex, this->coreDataNum);
        valueGm.SetGlobalBuffer((__gm__ TYPE_X *)value);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        //tmp1用于临时存储数据用，方便类型的转换，用于转换成float
        #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t>){
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float32_t));
        }else if constexpr (std::is_same_v<TYPE_X, int64_t>){
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(int32_t));
        }
        #endif
    }
    __aicore__ inline void Process()
    {
        //在process侧实现分流，实现对复数类型和常规数据类型的处理
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t>)
        {
            
            LocalTensor<float32_t> p1 = tmp1.Get<float32_t>();
            Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(p1, p1,ToFloat(valueGm.GetValue(0)) , this->processDataNum);
            Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
        }
        else if constexpr (std::is_same_v<TYPE_X, int64_t>)
        {
            LocalTensor<int32_t> p2 = tmp2.Get<int32_t>();
            Cast(p2, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(p2, p2,(int32_t)valueGm.GetValue(0) , this->processDataNum);
            Cast(yLocal, p2, RoundMode::CAST_NONE, this->processDataNum);
        }
        else
        {
            Muls(yLocal, xLocal, (TYPE_X)valueGm.GetValue(0), this->processDataNum);
        }
        #endif
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1;
    TBuf<QuePosition::VECCALC> tmp2;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> valueGm;
    GlobalTensor<TYPE_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};
// 在这里编写一个适配complex32类型的算子
template <typename TYPE_X>
class KernelMulsComplex32
{
public:
    __aicore__ inline KernelMulsComplex32() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
        uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
        uint32_t finalSmallTileNum, uint32_t tileDataNum,
        uint32_t smallTailDataNum, uint32_t bigTailDataNum,
        uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
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
        xGm.SetGlobalBuffer((__gm__ float *)x + globalBufferIndex * BUFFER_NUM, this->coreDataNum * BUFFER_NUM); // 1 complex = 2 float
        yGm.SetGlobalBuffer((__gm__ float *)y + globalBufferIndex * BUFFER_NUM, this->coreDataNum * BUFFER_NUM);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * BUFFER_NUM * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * BUFFER_NUM * sizeof(float));
        valueGm.SetGlobalBuffer((__gm__ TYPE_X *)value);
        // tBufXReal, tBufXImag,tBufYReal, tBufYImag,tBufRealOffset, tBufImagOffset
        pipe.InitBuffer(tBufXReal, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tBufXImag, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tBufRealOffset, this->tileDataNum * BUFFER_NUM * sizeof(uint32_t));
        pipe.InitBuffer(tBufImagOffset, this->tileDataNum * BUFFER_NUM * sizeof(uint32_t));
    }
    __aicore__ inline void Process()
    {
        //在process侧实现分流，实现对复数类型和常规数据类型的处理
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

        DataCopy(xLocal, xGm[progress * this->tileDataNum * 2], this->processDataNum * 2);

        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        // tBufXReal, tBufXImag,tBufYReal, tBufYImag,tBufRealOffset, tBufImagOffset
        LocalTensor<float> xRealLocal = tBufXReal.Get<float>();
        LocalTensor<float> xImagLocal = tBufXImag.Get<float>();
        LocalTensor<uint32_t> realOffsetLocal = tBufRealOffset.Get<uint32_t>();
        LocalTensor<uint32_t> imagOffsetLocal = tBufImagOffset.Get<uint32_t>();
        // 设置每个元素的偏移量，用于从复数数据中提取实部和虚部
        uint32_t COMPLEX32_ELEMENT_SIZE = 4;  // Size in bytes for complex32
        uint32_t COMPLEX32_REAL_OFFSET = 0;   // Offset for real part
        uint32_t COMPLEX32_IMAG_OFFSET = 2;   // Offset for imaginary part
        for (size_t i = 0; i < this->processDataNum; i++)
        {
            realOffsetLocal.SetValue(i, i * COMPLEX32_ELEMENT_SIZE + COMPLEX32_REAL_OFFSET);
            imagOffsetLocal.SetValue(i, i * COMPLEX32_ELEMENT_SIZE + COMPLEX32_IMAG_OFFSET);
        }
        // Gather 实部数据：从 xLocal 中按偏移量提取出实部到 xRealLocal 中
        Gather(xRealLocal, xLocal, realOffsetLocal, (uint32_t)0, this->processDataNum);
        // Gather 虚部数据：从 xLocal 中按偏移量提取出虚部到 xImagLocal 中
        Gather(xImagLocal, xLocal, imagOffsetLocal, (uint32_t)0, this->processDataNum);
        Muls(xRealLocal, xRealLocal, valueGm.GetValue(0), this->processDataNum);
        Muls(xImagLocal, xImagLocal, valueGm.GetValue(0), this->processDataNum);
        uint32_t COMPLEX_NUM_COMPONENTS = 2;  // Number of components (real + imaginary)
        uint32_t COMPLEX_IMAG_INDEX = 1;     // Index offset for imaginary part
        for (size_t i = 0; i < this->processDataNum; i++)
        {

            yLocal.SetValue(i * COMPLEX_NUM_COMPONENTS, xRealLocal.GetValue(i));
            yLocal.SetValue(i * COMPLEX_NUM_COMPONENTS + COMPLEX_IMAG_INDEX, xImagLocal.GetValue(i));
        }

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();

        DataCopy(yGm[progress * this->tileDataNum * BUFFER_NUM], yLocal, this->processDataNum * BUFFER_NUM);
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm;
    GlobalTensor<TYPE_X> valueGm;
    GlobalTensor<float> yGm;
    TBuf<QuePosition::VECCALC> tBufXReal;
    TBuf<QuePosition::VECCALC> tBufXImag;
    TBuf<QuePosition::VECCALC> tBufRealOffset;
    TBuf<QuePosition::VECCALC> tBufImagOffset;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};
//在这里编写一个适配complex64类型的算子
template <typename TYPE_X>
class KernelMulsComplex64
{
public:
    __aicore__ inline KernelMulsComplex64() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
        uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
        uint32_t finalSmallTileNum, uint32_t tileDataNum,
        uint32_t smallTailDataNum, uint32_t bigTailDataNum,
        uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
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
        xGm.SetGlobalBuffer((__gm__ float *)x + globalBufferIndex * BUFFER_NUM, this->coreDataNum * BUFFER_NUM); // 1 complex = 2 float
        yGm.SetGlobalBuffer((__gm__ float *)y + globalBufferIndex * BUFFER_NUM, this->coreDataNum * BUFFER_NUM);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * BUFFER_NUM * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * BUFFER_NUM * sizeof(float));
        valueGm.SetGlobalBuffer((__gm__ TYPE_X *)value);
        // tBufXReal, tBufXImag,tBufYReal, tBufYImag,tBufRealOffset, tBufImagOffset
        pipe.InitBuffer(tBufXReal, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tBufXImag, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tBufRealOffset, this->tileDataNum * BUFFER_NUM * sizeof(uint32_t));
        pipe.InitBuffer(tBufImagOffset, this->tileDataNum * BUFFER_NUM * sizeof(uint32_t));
    }
    __aicore__ inline void Process()
    {
        //在process侧实现分流，实现对复数类型和常规数据类型的处理
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

        DataCopy(xLocal, xGm[progress * this->tileDataNum * BUFFER_NUM], this->processDataNum * BUFFER_NUM);

        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        // tBufXReal, tBufXImag,tBufYReal, tBufYImag,tBufRealOffset, tBufImagOffset
        LocalTensor<float> xRealLocal = tBufXReal.Get<float>();
        LocalTensor<float> xImagLocal = tBufXImag.Get<float>();
        LocalTensor<uint32_t> realOffsetLocal = tBufRealOffset.Get<uint32_t>();
        LocalTensor<uint32_t> imagOffsetLocal = tBufImagOffset.Get<uint32_t>();
        // 设置每个元素的偏移量，用于从复数数据中提取实部和虚部
        uint32_t COMPLEX64_ELEMENT_SIZE = 8;  // Size in bytes for complex64
        uint32_t COMPLEX64_REAL_OFFSET = 0;   // Offset for real part
        uint32_t COMPLEX64_IMAG_OFFSET = 4;   // Offset for imaginary part
        for (size_t i = 0; i < this->processDataNum; i++)
        {
            realOffsetLocal.SetValue(i, i * COMPLEX64_ELEMENT_SIZE + COMPLEX64_REAL_OFFSET);
            imagOffsetLocal.SetValue(i, i * COMPLEX64_ELEMENT_SIZE + COMPLEX64_IMAG_OFFSET);
        }
        // Gather 实部数据：从 xLocal 中按偏移量提取出实部到 xRealLocal 中
        Gather(xRealLocal, xLocal, realOffsetLocal, (uint32_t)0, this->processDataNum);
        // Gather 虚部数据：从 xLocal 中按偏移量提取出虚部到 xImagLocal 中
        Gather(xImagLocal, xLocal, imagOffsetLocal, (uint32_t)0, this->processDataNum);

        Muls(xRealLocal, xRealLocal, valueGm.GetValue(0), this->processDataNum);

        Muls(xImagLocal, xImagLocal, valueGm.GetValue(0), this->processDataNum);
        uint32_t COMPLEX_NUM_COMPONENTS = 2;  // Number of components (real + imaginary)
        uint32_t COMPLEX_IMAG_INDEX = 1;     // Index offset for imaginary part
        for (size_t i = 0; i < this->processDataNum; i++)
        {
            yLocal.SetValue(i * COMPLEX_NUM_COMPONENTS, xRealLocal.GetValue(i));
            yLocal.SetValue(i * COMPLEX_NUM_COMPONENTS + COMPLEX_IMAG_INDEX, xImagLocal.GetValue(i));
        }
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileDataNum * BUFFER_NUM], yLocal, this->processDataNum * BUFFER_NUM);
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm;
    GlobalTensor<TYPE_X> valueGm;
    GlobalTensor<float> yGm;
    TBuf<QuePosition::VECCALC> tBufXReal;
    TBuf<QuePosition::VECCALC> tBufXImag;
    TBuf<QuePosition::VECCALC> tBufRealOffset;
    TBuf<QuePosition::VECCALC> tBufImagOffset;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};
extern "C" __global__ __aicore__ void muls( GM_ADDR x,
      GM_ADDR value,
      GM_ADDR y, 
      GM_ADDR workspace, 
      GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if(TILING_KEY_IS(0)){
        KernelMuls<bfloat16_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();
    }else if(TILING_KEY_IS(1)){
        KernelMuls<float16_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();
    }else if(TILING_KEY_IS(2)){
        KernelMuls<float32_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();
    }else if(TILING_KEY_IS(3)){
        KernelMuls<int16_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();
    }else if(TILING_KEY_IS(4)){
        KernelMuls<int32_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();
    }else if(TILING_KEY_IS(5)){
        KernelMuls<int64_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();}
    else if(TILING_KEY_IS(6)){
        KernelMulsComplex32<float> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();
    }else if(TILING_KEY_IS(7)){
        KernelMulsComplex64<float> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
        op.Process();
    }
    #endif
}
#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void muls_do(uint32_t blockDim,
    void *l2ctrl,
    void *stream,
    uint8_t *x,
    uint8_t *value,
    uint8_t *y,
    uint8_t *workspace,
    uint8_t *tiling)
{
    muls<<<blockDim, l2ctrl, stream>>>(x,value, y, workspace, tiling);
}
#endif