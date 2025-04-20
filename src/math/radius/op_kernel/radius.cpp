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
 * @file radius.cpp
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelRadius
{
    using T = TYPE_X;
public:
    __aicore__ inline KernelRadius() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR out, GM_ADDR shape_out,
                                float r, uint32_t ignoreSameIndex, uint32_t maxNumNeighbors,
                                uint32_t xSize, uint32_t ySize, uint32_t itemLength, uint32_t ptrXLen, uint32_t ptrYLen)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->r = r * r;
        this->ignoreSameIndex = ignoreSameIndex;
        this->maxNumNeighbors = maxNumNeighbors;
        this->xSize = xSize;
        this->ySize = ySize;
        this->itemLength = itemLength;
        this->maxSize = maxNumNeighbors * ySize;
        this->totalCount = 0;
        this->ptrXLen = ptrXLen;
        this->ptrYLen = ptrYLen;
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x);
        yGm.SetGlobalBuffer((__gm__ TYPE_X *)y);
        outGm.SetGlobalBuffer((__gm__ TYPE_X *)out, maxSize * 2);
        shapeOutGm.SetGlobalBuffer((__gm__ uint64_t *)shape_out, 3);
        if(ptrXLen!=0){
            ptrXGm.SetGlobalBuffer((__gm__ int32_t *)ptr_x, ptrXLen);
        }
        if(ptrYLen!=0){
            ptrYGm.SetGlobalBuffer((__gm__ int32_t *)ptr_y, ptrYLen);
        }
        pipe.InitBuffer(itemBuf, this->itemLength * sizeof(TYPE_X));
        pipe.InitBuffer(xItemBuf, this->itemLength * sizeof(TYPE_X));
        pipe.InitBuffer(xItemFpBuf, this->itemLength * sizeof(float));
        pipe.InitBuffer(itemFpBuf, this->itemLength * sizeof(float));
        pipe.InitBuffer(resBuf, 32);
    }
    __aicore__ inline void Process()
    {
        int loopCount = ptrYLen - 1, yStart, yEnd, xStart, xEnd;
        uint64_t mask[2] = { 1, 0};
        UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };
        if(ptrYLen == 0){
            loopCount = 1;
        }else{
            yEnd = ptrYGm.GetValue(0);
            xEnd = ptrXGm.GetValue(0);
        }
        for(int q = 0; q < loopCount; q++)
        {
            if(ptrYLen == 0){
                yStart = 0;
                yEnd = ySize;
                xStart = 0;
                xEnd = xSize;
            }else{
                yStart = yEnd;
                yEnd = ptrYGm.GetValue(q + 1);
                xStart = xEnd;
                xEnd = ptrXGm.GetValue(q + 1);
            }
            if(yStart == yEnd || xStart == xEnd) continue;
            for(int i = yStart; i < yEnd; i++){
                LocalTensor<TYPE_X> itemLocal = itemBuf.Get<TYPE_X>();
                for(int j = 0; j < itemLength; j ++){
                    itemLocal.SetValue(j, (TYPE_X)yGm.GetValue(i * itemLength + j));
                }
                int32_t count = 0;

                for(int j = xStart; j < xEnd; j++){
                    LocalTensor<TYPE_X> xItemLocal = xItemBuf.Get<TYPE_X>();
                    LocalTensor<int8_t> resLocal = resBuf.Get<int8_t>();
                    if(j == i && ignoreSameIndex) continue;
                    for(int k = 0; k < itemLength; k++){
                        xItemLocal.SetValue(k, (TYPE_X)xGm.GetValue(j * itemLength + k));
                    }
                    if constexpr(std::is_same_v<TYPE_X, int32_t>){
                        LocalTensor<float> xItemLocalFp = xItemFpBuf.Get<float>();
                        LocalTensor<float> itemLocalFp = itemFpBuf.Get<float>();
                        Cast(xItemLocalFp, xItemLocal, RoundMode::CAST_RINT, itemLength);
                        Cast(itemLocalFp, itemLocal, RoundMode::CAST_RINT, itemLength);
                        Sub(xItemLocalFp, xItemLocalFp, itemLocalFp, itemLength);
                        Mul(xItemLocalFp, xItemLocalFp, xItemLocalFp, itemLength);
                        ReduceSum<float>(xItemLocalFp, xItemLocalFp, xItemLocalFp, itemLength);
                        CompareScalar(resLocal, xItemLocalFp, (float)r, AscendC::CMPMODE::LE, mask, 1, repeatParams);
                        int8_t value = resLocal.GetValue(0);
                        if(value & 0b00000001){
                            outGm.SetValue(totalCount, (TYPE_X)j);
                            outGm.SetValue(maxSize + totalCount, (TYPE_X)i);
                            count++;
                            totalCount++;
                            
                            if(count == maxNumNeighbors){
                                break;
                            }
                        }
                    } else if constexpr(std::is_same_v<TYPE_X, float>){
                        Sub(xItemLocal, xItemLocal, itemLocal, itemLength);
                        Mul(xItemLocal, xItemLocal, xItemLocal, itemLength);
                        ReduceSum<float>(xItemLocal, xItemLocal, xItemLocal, itemLength);
                        CompareScalar(resLocal, xItemLocal, (float)r, AscendC::CMPMODE::LE, mask, 1, repeatParams);
                        int8_t value = resLocal.GetValue(0);
                        if(value & 0b00000001){
                            outGm.SetValue(totalCount, (float)j);
                            outGm.SetValue(maxSize + totalCount, (float)i);
                            count++;
                            totalCount++;
                            if(count == maxNumNeighbors){
                                break;
                            }
                        }
                    } else if constexpr(std::is_same_v<TYPE_X, half>){
                        LocalTensor<float> xItemLocalFp = xItemFpBuf.Get<float>();
                        LocalTensor<float> itemLocalFp = itemFpBuf.Get<float>();
                        Cast(xItemLocalFp, xItemLocal, RoundMode::CAST_NONE, itemLength);
                        Cast(itemLocalFp, itemLocal, RoundMode::CAST_NONE, itemLength);
                        Sub(xItemLocalFp, xItemLocalFp, itemLocalFp, itemLength);
                        Mul(xItemLocalFp, xItemLocalFp, xItemLocalFp, itemLength);
                        ReduceSum<float>(xItemLocalFp, xItemLocalFp, xItemLocalFp, itemLength);
                        CompareScalar(resLocal, xItemLocalFp, (float)r, AscendC::CMPMODE::LE, mask, 1, repeatParams);
                        int8_t value = resLocal.GetValue(0);
                        if(value & 0b00000001){
                            outGm.SetValue(totalCount, (TYPE_X)j);
                            outGm.SetValue(maxSize + totalCount, (TYPE_X)i);
                            count++;
                            totalCount++;
                            if(count == maxNumNeighbors){
                                break;
                            }
                        }
                    }
                    
                }
                
            }
        }
        
        if(totalCount < maxSize){
            for(int i = 0; i < totalCount; i++){
                outGm.SetValue(totalCount + i, (TYPE_X)outGm.GetValue(maxSize + i));
            }
        }
        shapeOutGm.SetValue(0, (uint64_t)2);
        shapeOutGm.SetValue(1, (uint64_t)2);
        shapeOutGm.SetValue(2, (uint64_t)totalCount);
    }

private:
    TPipe pipe;
    GlobalTensor<TYPE_X> xGm, yGm, outGm;
    GlobalTensor<int32_t> ptrXGm, ptrYGm;
    GlobalTensor<uint64_t> shapeOutGm;
    TBuf<QuePosition::VECCALC> itemBuf, xItemBuf, itemFpBuf, xItemFpBuf, resBuf;
    float r;
    uint32_t ignoreSameIndex, maxNumNeighbors;
    uint32_t ySize, itemLength, totalCount, xSize, maxSize, ptrXLen, ptrYLen;
};

extern "C" __global__ __aicore__ void radius(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR out, GM_ADDR shape_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRadius<DTYPE_X> op;
    op.Init(x, y, ptr_x, ptr_y, out, shape_out, 
        tiling_data.r, tiling_data.ignore_same_index, tiling_data.max_num_neighbors,
        tiling_data.xSize, tiling_data.ySize, tiling_data.itemLength, tiling_data.ptrXLen, tiling_data.ptrYLen);  
    op.Process();
}