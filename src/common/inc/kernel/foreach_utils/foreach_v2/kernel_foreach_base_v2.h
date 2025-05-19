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
 * \file kernel_foreach_base.h
 * \brief
 */

#ifndef KERNEL_FOREACH_BASE_H
#define KERNEL_FOREACH_BASE_H

#include <type_traits>
#include "kernel_operator.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr uint8_t COPY_SPACE_MULTIPLE = 9;
constexpr uint16_t MAX_TENSOR_CONT = 50;
constexpr uint16_t MAX_CORE_CONT = 50;
constexpr uint32_t BYTE_BLOCK = 32;

template <typename T>
class KernelForeachBaseV2 {
protected:
    __aicore__ inline KernelForeachBaseV2() {};

    __aicore__ inline void Init(const ForeachCommonV2TilingData* tilingData);
    __aicore__ inline void InitParams();
    __aicore__ inline void ParseTilingData(const ForeachCommonV2TilingData* tilingData);
    __aicore__ inline void parseNumels(GM_ADDR x);
    // __aicore__ inline void AssignDataToEachCore(uint32_t needCoreNum);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

protected:
    TPipe pipe;

    int64_t blockIdx = 0;

    // tiling params
    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;
    uint32_t needCoreNum = 0;
    uint64_t totalTensorUbSize = 0;
    uint32_t maxDataCount = 0;
    uint32_t maxCastDataCount = 0;
    uint16_t totalTensorCount = 0;
    uint64_t totalBlockCount = 0; // for reduce
    uint8_t elementsPerBlock = BYTE_BLOCK / sizeof(T);
    int64_t totalDataCount = 0;
};

template <typename T>
__aicore__ inline void KernelForeachBaseV2<T>::Init(
    const ForeachCommonV2TilingData* tilingData) {
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);
    InitParams();
}

template <typename T>
__aicore__ inline void KernelForeachBaseV2<T>::ParseTilingData(
    const ForeachCommonV2TilingData* tilingData) {
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    needCoreNum = tilingData->needCoreNum;
}

template <typename T>
__aicore__ inline __gm__ T* KernelForeachBaseV2<T>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

template <typename T>
__aicore__ inline void KernelForeachBaseV2<T>::InitParams() {
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            maxDataCount = totalTensorUbSize / sizeof(T);        
            maxCastDataCount = inputsTensorUbSize / sizeof(float);
        } else {
            maxDataCount = inputsTensorUbSize / sizeof(T);
        }
    #else 
        maxDataCount = inputsTensorUbSize / sizeof(T);
    #endif
}

template <typename T>
__aicore__ inline void KernelForeachBaseV2<T>::parseNumels(GM_ADDR x) {
    /* 
    数据第一个8字节记录ptr_offset，地址相对于首地址的偏移量指向ptr（8字节）
    接着是shape信息，可以支持多个相同shape合并
    dim：记录shape的维度(4字节)
    cnt：记录连续相同shape的个数（4字节），如果cnt不为1，则代表有cnt个输入的shape是相同且连续，可以支持合并或者不合并，不合并cnt固定为1
    dim*int64：shape每个维度以8字节记录，有dim个（dim个8字节）
    最后ptr_offset指向的是连续的输入指针（输入个数个8字节）

        tensorlist 数据结构
                                    --------------
        ptr偏移(8字节)              | ptr_offset |
                                    -------|------
        第1组shape                  | dim  | cnt |
                                    -------|------
                                    | dim * int64|
                                    --------------
        第2组shape                  | dim  | cnt |
                                    -------|------
                                    | dim * int64|
                                    --------------
        ...                         |  ......    |
                                    --------------
        第N组shape                  | dim  | cnt |
                                    --------------
                                    | dim * int64|
                                    --------------
        数据指针                    | ptr1       |
                                    --------------
        数据指针                    | ptr2       |
                                    --------------
        数据指针                    | ptr...     |
                                    --------------
    */
    int64_t dataAddrOffset = *reinterpret_cast<__gm__ int64_t *>(x); // 数据指针相对于首地址的偏移量(单位字节)

    // sizeof(__gm__ uint8_t *) 8字节
    __gm__ uint8_t *start_ptr = x + sizeof(__gm__ uint8_t *); // 维度信息开始地址
    __gm__ uint8_t *end_ptr = x + dataAddrOffset;

    __gm__ uint8_t *ptr = start_ptr;
    while (ptr < end_ptr) {
        int32_t dim = *reinterpret_cast<__gm__ int32_t *>(ptr);
        ptr += (sizeof(int32_t) + sizeof(int32_t));

        int64_t size = 1;
        for (int64_t i = 0; i < dim; i++) {
            size *= *reinterpret_cast<__gm__ int64_t *>(ptr);
            ptr += sizeof(int64_t);
        }

        tensorDataCountList[totalTensorCount] = size;
        totalDataCount += size;
        totalTensorCount++;
        totalBlockCount += CeilA2B(size, elementsPerBlock);
    }
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_BASE_H