/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_foreach_elewise.h
 * \brief
 */

#ifndef KERNEL_FOREACH_ELEWISE_H
#define KERNEL_FOREACH_ELEWISE_H

#include "kernel_foreach_base_v2.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint8_t INPUT_PARAMETER_COUNT = 2;
constexpr bool NEED_COPY_OUT = true;

template <typename T, typename Predicate, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class KernelForeachElewise : public KernelForeachBaseV2<T> {
protected:
    using Base = KernelForeachBaseV2<T>;

    explicit __aicore__ inline KernelForeachElewise(Predicate &p): Base(), pred(p) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const ForeachCommonV2TilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor);
    __aicore__ inline void InitQueueBuffer();
    __aicore__ inline void InitCoreParams(GM_ADDR x, uint32_t needCoreNum);
    __aicore__ inline void AssignDataToEachCore(uint32_t needCoreNum);
private:
    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder);
    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void BeforeProcess();
    __aicore__ inline void AfterProcess();
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart);

protected:
    TQue<QuePosition::VECIN, bufferNum> dataQueue;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};

    uint8_t elementsPerBlock = BYTE_BLOCK / sizeof(T);
private:
    Predicate &pred;
};

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const ForeachCommonV2TilingData* tilingData) {
    Base::Init(tilingData);

    inTensorsPtr = x;
    outTensorsPtr = y;

    InitQueueBuffer();
    InitCoreParams(x, Base::needCoreNum);
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::InitQueueBuffer() {
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            Base::pipe.InitBuffer(dataQueue, bufferNum, Base::totalTensorUbSize);
            if (needCopyOut) {
                Base::pipe.InitBuffer(outQueue, bufferNum, Base::totalTensorUbSize);
            }
            Base::pipe.InitBuffer(float32Queue, 1, Base::inputsTensorUbSize * paramsCount);
            LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
            float32Queue.EnQue(float32Tensor);
        } else {
            Base::pipe.InitBuffer(dataQueue, bufferNum, Base::inputsTensorUbSize);
            if (needCopyOut) {
                Base::pipe.InitBuffer(outQueue, bufferNum, Base::inputsTensorUbSize);
            }
        }
    #else 
        Base::pipe.InitBuffer(dataQueue, bufferNum, Base::inputsTensorUbSize);
        if (needCopyOut) {
            Base::pipe.InitBuffer(outQueue, bufferNum, Base::inputsTensorUbSize);
        }
    #endif
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::InitCoreParams(GM_ADDR x, uint32_t needCoreNum) {
    Base::parseNumels(x);
    AssignDataToEachCore(needCoreNum);
 }

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::AssignDataToEachCore(uint32_t needCoreNum) {
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = Base::CeilA2B(Base::totalDataCount, elementsPerBlock); // 总元素量块数
    // Divisible, representing the amount of data each core needs to process.
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock; // 每个核处理的元素数量大小
    int64_t remainderCount = blockCount % needCoreNum;  // remainder. 尾部块数
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0; // 实际每个核处理元素数量大小
    int64_t cursorPosition = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;

    for (uint16_t i = 0; i < Base::totalTensorCount; i++) {
        if (Base::blockIdx < coreIndex) {
            break;
        }
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock; // 把尾部块数再给前remainderCount个核多分一块
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = Base::tensorDataCountList[i] - cursorPosition;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPosition = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPosition = cursorPosition + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPosition - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPosition < Base::tensorDataCountList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPosition;
            --i;  // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPosition = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
     and you need to manually set the offset of the last core. */
    if (dataCount) {
        tensorEndList[coreIndex] =  Base::totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = Base::tensorDataCountList[Base::totalTensorCount - 1] - 1;
    }

    Base::tensorStart = tensorStartList[Base::blockIdx];
    Base::tensorEnd = tensorEndList[Base::blockIdx];
    Base::tensorStartOffset = tensorStartOffsetList[Base::blockIdx];
    Base::tensorEndOffset = tensorEndOffsetList[Base::blockIdx];
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::Process() {
    /*将中间量预留出来*/
    LocalTensor<float> float32Tensor;
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            float32Tensor = float32Queue.DeQue<float>(); 
        }
    #endif

    BeforeProcess();
    for (uint16_t i = Base::tensorStart; i <= Base::tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = Base::tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == Base::tensorStart) {
            cursorStart = Base::tensorStartOffset;
        }
        if (i == Base::tensorEnd) {
            cursorEnd = Base::tensorEndOffset;
        }

        dataCount = cursorEnd - cursorStart + 1;

        inTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, inTensorsPtr) + cursorStart);
        outTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, outTensorsPtr) + cursorStart);
        ProcessPlusInLoop(i, cursorStart);
        SingleTensorProcess(dataCount, float32Tensor);
    }

    AfterProcess();

    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            float32Queue.FreeTensor(float32Tensor);
        }
    #endif
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::SingleTensorProcess(
    int64_t dataCount, LocalTensor<float> &float32Tensor) {
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / Base::maxDataCount;
    uint32_t copyTimesRemainder = dataCount % Base::maxDataCount;
    uint32_t tempDataCount = Base::maxDataCount;

    if (copyTimesRemainder > 0) {
        copyTimes++;
    }
 
    for (uint32_t i = 0; i < copyTimes; i++) {
        bool isRemainder = false;
        if (i == copyTimes - 1 && copyTimesRemainder > 0) {
            isRemainder = true;
            tempDataCount = copyTimesRemainder;
        }
        CopyIn(i, tempDataCount, isRemainder);
        CopyInPlus(i, tempDataCount, isRemainder);
        Compute(i, tempDataCount, float32Tensor, isRemainder);
        if (needCopyOut) {
            CopyOut(i, tempDataCount, isRemainder);
        }
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyIn(uint32_t index, int64_t dataCount, bool isRemainder) {
    LocalTensor<T> dataLocal = dataQueue.template AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[index * Base::maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[index * Base::maxDataCount], dataCount);
    }
    dataQueue.EnQue(dataLocal);
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline bool KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
    static_assert(std::is_member_function_pointer_v<decltype(&Predicate::CopyOut)>);
    if (!pred.CopyOut(index, dataCount, isRemainder)) {
        LocalTensor<T> outLocal = outQueue.template DeQue<T>();
	    if (isRemainder) {
	        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
	        DataCopyPad(outTensorsGM[index * Base::maxDataCount], outLocal, copyParams);
	    } else {
	        DataCopy(outTensorsGM[index * Base::maxDataCount], outLocal, dataCount);
	    }

        outQueue.FreeTensor(outLocal);
    }
    return true;
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::Compute(
    uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
    static_assert(std::is_member_function_pointer_v<decltype(&Predicate::Compute)>);
    pred.Compute(index, dataCount, float32Tensor, isRemainder);
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {
    if (std::is_member_function_pointer_v<decltype(&Predicate::CopyInPlus)>) {
        pred.CopyInPlus(index, dataCount, isRemainder);
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::BeforeProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::BeforeProcess)>) {
        pred.BeforeProcess();
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::AfterProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::AfterProcess)>) {
        pred.AfterProcess();
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachElewise<T, Predicate, bufferNum, paramsCount, needCopyOut>::ProcessPlusInLoop(uint32_t index, uint64_t cursorStart){
    if (std::is_member_function_pointer_v<decltype(&Predicate::ProcessPlusInLoop)>) {
        pred.ProcessPlusInLoop(index, cursorStart);
    }
}
}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_ELEWISE_H