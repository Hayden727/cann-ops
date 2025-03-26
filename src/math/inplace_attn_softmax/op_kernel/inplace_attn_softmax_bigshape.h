/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INPLACE_ATTN_SOFTMAX_BIG_SHAPE_H
#define INPLACE_ATTN_SOFTMAX_BIG_SHAPE_H

#include <typeinfo>
#include "kernel_operator.h"
#include "inplace_attn_softmax_bash.h"

namespace InplaceAttnSoftmaxOpt {
using namespace AscendC;

template <typename inType, typename outType, bool isCast, bool isBigshape>
class InplaceAttnSoftmaxBigShape : public InplaceAttnSoftmaxBase<inType, outType, isCast, isBigshape> {
public:
    __aicore__ inline InplaceAttnSoftmaxBigShape(TPipe *pipe)
    {
        this->pPipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x,GM_ADDR workspace, const InplaceAttnSoftmaxTilingData *__restrict tilingData)
    {
        this->ParseTilingData(tilingData);
        this->InitParamsComm();
        InitAndSetBuffer(x, workspace);
    }

    __aicore__ inline void Process()
    {
        ProcessCoreMultiUbMultiBigShape();
    }

private:


    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR workspace_gm)
    {
        // gm数据
        xGm.SetGlobalBuffer((__gm__ inType *)x,this->tilingData_.rowLen * this->tilingData_.colLen);
        // queue
        this->pPipe->InitBuffer(inQueueA, BUFFER_NUM, this->basicColLen * sizeof(inType));
        this->pPipe->InitBuffer(outQueueA, BUFFER_NUM, this->tileLength * sizeof(inType));
        this->pPipe->InitBuffer(sharedTempBuf, this->basicColLen * sizeof(inType));
        tmpALocal = sharedTempBuf.Get<float>(this->basicColLen);
        if constexpr(isCast) {
            this->pPipe->InitBuffer(sharedBTempBuf, this->basicColLen * sizeof(float));
            tmpCLocal = sharedBTempBuf.Get<float>(this->basicColLen);
        } 
    }

    __aicore__ inline void ProcessCoreMultiUbMultiBigShape()
    {
        uint32_t offsetRow = 0;
        DataCopyParams splitCopyinParams;
        DataCopyParams splitCopyoutParams;

        splitCopyinParams = {1,(uint16_t)(this->basicColLen * sizeof(inType)),0,0};
        splitCopyoutParams = {1,(uint16_t)(this->basicColLen * sizeof(outType)),0,0};

        for (uint32_t ridx = 0; ridx < this->rowLoop; ridx++) {
            // 每个核心每次循环的起始偏移地址
            for(uint32_t cidx = 0; cidx < this->colLoop; cidx++){
                ComputeVecInGmOffset(ridx,cidx);
                maxCopyIn(this->offsetParam,splitCopyinParams,ridx,cidx);
            }
            sumperrow = 0;
            for(uint32_t cidx = 0; cidx < this->colLoop; cidx++){
                ComputeVecInGmOffset(ridx,cidx);
                subCopyIn(this->offsetParam,splitCopyinParams,ridx,cidx);
            }
            for(uint32_t cidx = 0; cidx < this->colLoop; cidx++){
                ComputeVecInGmOffset(ridx,cidx);
                mulCopyIn(this->offsetParam,splitCopyinParams,ridx,cidx);
            }
        }
    }

    __aicore__ inline void getSplitCopyinParams(uint32_t cidx, DataCopyParams &splitCopyinParams) 
    {
        if(cidx == this->colLoop - 1 ){
            splitCopyinParams = {1,(uint16_t)(this->lastcolLen * sizeof(inType)),0,0};
        }else{
            splitCopyinParams = {1,(uint16_t)(this->basicColLen * sizeof(inType)),0,0};
        }
    }

    __aicore__ inline void maxCopyIn(InplaceAttnSoftmaxOffsetParam &offsetParam,DataCopyParams &splitCopyinParams,uint32_t ridx,uint32_t cidx)
    {
        LocalTensor<inType> aLocal = inQueueA.template AllocTensor<inType>();
        getSplitCopyinParams(cidx, splitCopyinParams);
        padParams = {true, 0, 0, 0};
        DataCopyPad(aLocal, xGm[offsetParam.tmpVecGmOffset], splitCopyinParams, padParams);
        // PipeBarrier<PIPE_ALL>();
        if constexpr(isCast) {
            AscendC::Cast(tmpCLocal, aLocal, AscendC::RoundMode::CAST_NONE, aLocal.GetSize());
            if(cidx == this->colLoop - 1 && this->lastcolLen != 0){
                ReduceMax(tmpALocal, tmpCLocal, tmpALocal, this->lastcolLen, false); 
            }else {
                ReduceMax(tmpALocal, tmpCLocal, tmpALocal, this->basicColLen, false);
                }
        } else
        {
            if(cidx == this->colLoop - 1 && this->lastcolLen != 0){
                ReduceMax(tmpALocal, aLocal, tmpALocal, this->lastcolLen, false); 
            } else {
                ReduceMax(tmpALocal, aLocal, tmpALocal, this->basicColLen, false);
                }
        }
        inQueueA.FreeTensor(aLocal);
        if(cidx == 0){
            maxperrow = tmpALocal.GetValue(0);
        }else {
            if(static_cast<float>(tmpALocal.GetValue(0)) > maxperrow){
                maxperrow = tmpALocal.GetValue(0);
            } 
        }
    }

    __aicore__ inline void subCopyIn(InplaceAttnSoftmaxOffsetParam &offsetParam, DataCopyParams &splitCopyinParams, 
                                    uint32_t ridx, uint32_t cidx)
    {
        LocalTensor<inType> aLocal = inQueueA.template AllocTensor<inType>();
        LocalTensor<inType> outLocal = outQueueA.template AllocTensor<inType>(); 
        getSplitCopyinParams(cidx, splitCopyinParams);
        padParams = {true, 0, 0, 0};
        DataCopyPad(aLocal, xGm[offsetParam.tmpVecGmOffset], splitCopyinParams, padParams);
        // PipeBarrier<PIPE_ALL>();
        if constexpr(isCast) {
            AscendC::Cast(tmpCLocal, aLocal, AscendC::RoundMode::CAST_NONE, aLocal.GetSize());
            inQueueA.FreeTensor(aLocal);
            Adds<float>(tmpCLocal, tmpCLocal, static_cast<float>(-1*maxperrow), this->basicColLen);
            Exp<float>(tmpCLocal, tmpCLocal, this->basicColLen);
            // PipeBarrier<PIPE_ALL>();
            if(cidx == this->colLoop - 1 && this->lastcolLen != 0){
                ReduceSum(tmpALocal, tmpCLocal, tmpALocal, this->lastcolLen);
            }else {
                ReduceSum(tmpALocal, tmpCLocal, tmpALocal, this->basicColLen); 
            }
            sumperrow = sumperrow + static_cast<float>(tmpALocal.GetValue(0));
            AscendC::Cast(outLocal, tmpCLocal, AscendC::RoundMode::CAST_RINT, aLocal.GetSize());
            DataCopyPad(xGm[offsetParam.tmpVecGmOffset], outLocal, splitCopyinParams);
            // PipeBarrier<PIPE_ALL>();
            outQueueA.FreeTensor(outLocal);
        } else
        {
            Adds<inType>(outLocal, aLocal, static_cast<float>(-1*maxperrow), this->basicColLen);
            inQueueA.FreeTensor(aLocal);
            Exp<inType>(outLocal, outLocal, this->basicColLen);
            // PipeBarrier<PIPE_ALL>();
            DataCopyPad(xGm[offsetParam.tmpVecGmOffset], outLocal, splitCopyinParams);
            // PipeBarrier<PIPE_ALL>();
            if(cidx == this->colLoop - 1 && this->lastcolLen != 0){
                ReduceSum(tmpALocal, outLocal, tmpALocal, this->lastcolLen);
            }else {
                ReduceSum(tmpALocal, outLocal, tmpALocal, this->basicColLen); 
            }
            outQueueA.FreeTensor(outLocal);
            sumperrow = sumperrow + static_cast<float>(tmpALocal.GetValue(0));
        }
        // inQueueA.FreeTensor(aLocal);
    }

    __aicore__ inline void mulCopyIn(InplaceAttnSoftmaxOffsetParam &offsetParam,DataCopyParams &splitCopyinParams,uint32_t ridx,uint32_t cidx)
    {
        LocalTensor<inType> aLocal = inQueueA.template AllocTensor<inType>();
        LocalTensor<inType> outLocal = outQueueA.template AllocTensor<inType>(); 
        getSplitCopyinParams(cidx, splitCopyinParams);
        padParams = {true, 0, 0, 0};
        DataCopyPad(aLocal, xGm[offsetParam.tmpVecGmOffset], splitCopyinParams, padParams);
        // PipeBarrier<PIPE_ALL>();
        if constexpr(isCast) {
            AscendC::Cast(tmpCLocal, aLocal, AscendC::RoundMode::CAST_NONE, aLocal.GetSize());
            inQueueA.FreeTensor(aLocal);
            Muls<float>(tmpCLocal, tmpCLocal, static_cast<float>(1 / sumperrow), this->basicColLen);
            // PipeBarrier<PIPE_ALL>();
            AscendC::Cast(outLocal, tmpCLocal, AscendC::RoundMode::CAST_RINT, aLocal.GetSize());
        } else 
        {
            Muls<inType>(outLocal, aLocal, static_cast<float>(1 / sumperrow), this->basicColLen);
            // PipeBarrier<PIPE_ALL>();
            inQueueA.FreeTensor(aLocal);
        }
        if(cidx == this->colLoop - 1 && this->lastcolLen != 0){
            DataCopyPad(xGm[offsetParam.tmpVecGmOffset], outLocal, {1,(uint16_t)(this->lastcolLen * sizeof(inType)),0,0});
            // PipeBarrier<PIPE_ALL>();
        }else {
            DataCopyPad(xGm[offsetParam.tmpVecGmOffset], outLocal, {1,(uint16_t)(this->basicColLen * sizeof(inType)),0,0});
            // PipeBarrier<PIPE_ALL>();
        }
        outQueueA.FreeTensor(outLocal);
    }

    __aicore__ inline void ComputeVecInGmOffset(uint32_t ridx,uint32_t cidx)
    {
        if (this->coreIdx < this->headCoreNum) {
            this->offsetParam.tmpVecGmOffset = this->coreIdx * this->rowLenPerCore * this->colLen + ridx * this->colLen + cidx * this->basicColLen;
        } else {
            this->offsetParam.tmpVecGmOffset = this->headCoreNum * this->tilingData_.rowLenPerHeadCore * this->colLen +
                                         (this->coreIdx - this->headCoreNum) * this->rowLenPerCore * this->colLen +
                                         + ridx * this->colLen + cidx * this->basicColLen;
        }
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueA;
    TBuf<TPosition::VECCALC> sharedTempBuf; 
    TBuf<TPosition::VECCALC> sharedBTempBuf;
    LocalTensor<float> tmpCLocal;
    LocalTensor<float> tmpALocal;
    GlobalTensor<inType> xGm;
    float maxperrow = 0;
    float sumperrow;
    DataCopyPadParams padParams;
};
}

#endif  