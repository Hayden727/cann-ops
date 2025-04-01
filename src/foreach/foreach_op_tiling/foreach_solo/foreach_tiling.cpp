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
 * \file foreach_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "foreach_tiling.h"
#include <vector>

namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;  // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;
constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_INT = 3;

constexpr uint64_t TILING_HALF_N_SCALAR = 14;
constexpr uint64_t TILING_FLOAT_N_SCALAR = 4;
constexpr uint64_t TILING_INT_N_SCALAR = 4;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint32_t BINARY_LIST_UB_DIVIDER = 6;
constexpr uint32_t BINARY_SCALAR_UB_DIVIDER = 4;
constexpr uint32_t FOREACH_POINTWISE_DIVIDER = 8;
constexpr uint32_t FOREACH_POW_SCALAR_DIVIDER = 4;
constexpr uint32_t FOREACH_COS_DIVIDER = 4;
constexpr uint32_t FOREACH_POINTWISE_LIST_DIVIDER = 10;

constexpr uint8_t SOLO_LOG_OP_CODE = 1;
constexpr uint8_t BINARY_LIST_OP_CODE = 2;
constexpr uint8_t FOREACH_POINTWISE_OP_CODE = 3;
constexpr uint8_t FOREACH_POW_SCALAR_OP_CODE = 4;
constexpr uint8_t FOREACH_COS_OP_CODE = 5;
constexpr uint8_t SOLO_LOG2_OP_CODE = 6;
constexpr uint8_t SOLO_NEG_OP_CODE = 7;
constexpr uint8_t FOREACH_POW_TENSOR_OP_CODE = 8;
constexpr uint8_t FOREACH_BINARY_SCALAR_OP_CODE = 9;
constexpr uint8_t FOREACH_POINTWISE_LIST_OP_CODE = 10;
constexpr uint8_t FOREACH_SIGMOID_OP_CODE = 11;

constexpr uint16_t LOG2_BASIC_FOR_LOG2 = 1024;
constexpr uint32_t LOG2_HALF_FOR_LOG2 = 4;
constexpr uint32_t LOG2_FLOAT_FOR_LOG2 = 0;

constexpr uint8_t BYTE_PER_BLOCK = 32;

constexpr int32_t POW_TENSOR_TENSOR_CALC_PROC[3] = {12, 3, 3};

static void GetLog2TmpBufferFactorSize(const uint32_t typeSize, uint32_t &extraBuf,
    uint32_t LOG2_HALF = LOG2_HALF_FOR_LOG2, uint32_t LOG2_FLOAT = LOG2_FLOAT_FOR_LOG2,
    uint32_t LOG2_BASIC = LOG2_BASIC_FOR_LOG2) {
    auto caclFactor = (typeSize == sizeof(float)) ? LOG2_FLOAT : LOG2_HALF;
    extraBuf = LOG2_BASIC * caclFactor * typeSize;
}

class ForeachTiling {
public:
    explicit ForeachTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init(uint8_t opCode);
    ge::graphStatus RunBigKernelTiling();

private:
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b);

    uint8_t GetDataTypeSize();
    uint64_t GetTilingKeyVal();
    uint64_t GetTilingN();

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    void AssignDataToEachCore(int64_t needCoreNum);
    void DivideUbMemory(uint64_t ubSizePlatForm);
    void FillTilingData();

private:
    ForeachSoloTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint32_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    int64_t totalDataCount = 0;
    uint8_t dataTypeSize = 4;
    uint8_t elementsPerBlock = 0;
    uint16_t totalTensorCount = 0;
    uint8_t opCode = 0;
};

ge::graphStatus ForeachTiling::Init(uint8_t theCode = 0) {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachTiling::RunBigKernelTiling() {
    return ge::GRAPH_SUCCESS;
}

template <typename T1, typename T2>
inline T1 ForeachTiling::CeilA2B(T1 a, T2 b) {
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

uint8_t ForeachTiling::GetDataTypeSize() {
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_INT32:
            return BYTE_LEN_4;
        default:
            return BYTE_LEN_4;
    }
}

uint64_t ForeachTiling::GetTilingKeyVal() {
    switch (dataType) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_INT32:
            return TILING_KEY_INT;
        default:
            return 0;
    }
}

uint64_t ForeachTiling::GetTilingN() {
    switch (dataType) {
        case ge::DT_FLOAT:
            return TILING_FLOAT_N_SCALAR;
        case ge::DT_FLOAT16:
            return TILING_HALF_N_SCALAR;
        case ge::DT_INT32:
            return TILING_INT_N_SCALAR;
        default:
            return TILING_HALF_N_SCALAR;
    }
}

uint32_t ForeachTiling::GetNeedCoreNum(uint32_t coreNumPlatform) {
    uint32_t tempCoreNum = (uint32_t)CeilA2B(totalDataCount, elementsPerBlock);
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void ForeachTiling::AssignDataToEachCore(int64_t needCoreNum) {
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
    // Divisible, representing the amount of data each core needs to process.
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum;  // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountList[i] - cursorPos;

        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPos;
            --i;  // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
    }
}

void ForeachTiling::DivideUbMemory(uint64_t ubSizePlatForm) {
    if (opCode == 0) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / 2;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == SOLO_LOG_OP_CODE) { // 1
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize()) / 2;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == BINARY_LIST_OP_CODE) { // 2
        // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / BINARY_LIST_UB_DIVIDER;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_POINTWISE_OP_CODE) { // 3
        // foreach_addcmul_scalar tiling
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) /
            FOREACH_POINTWISE_DIVIDER; // double buffer
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_POW_SCALAR_OP_CODE) {
        // foreach_pow_scalar tiling
        uint32_t reserveUbSize = BYTE_BASIC_BLOCK * GetTilingN() * dataTypeSize;
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - reserveUbSize) /
            FOREACH_POW_SCALAR_DIVIDER; // double buffer
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_COS_OP_CODE) { // 4
        // foreach_cos tiling
        uint32_t tilingConstant = 6;
        if (dataTypeSize == BYTE_LEN_4) {
            tilingConstant = TILING_FLOAT_N_SCALAR;
        }
        uint32_t reserveUbSize = BYTE_BASIC_BLOCK * tilingConstant * dataTypeSize;
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() -
            reserveUbSize) / FOREACH_COS_DIVIDER;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == SOLO_LOG2_OP_CODE) { // 5
        uint32_t extraBuf = 0;      // need extra space
        GetLog2TmpBufferFactorSize(dataTypeSize, extraBuf); // reuse source is true
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / 2 - extraBuf;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == SOLO_NEG_OP_CODE) {  // need extra buffer of one block: 32 bytes
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / 2 - BYTE_PER_BLOCK;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_POW_TENSOR_OP_CODE) {
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) /
            (BINARY_LIST_UB_DIVIDER + POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyVal()-1]);
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_BINARY_SCALAR_OP_CODE) {
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / BINARY_SCALAR_UB_DIVIDER;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_POINTWISE_LIST_OP_CODE) {
        // foreach_addcdiv_list/addcmul_list tiling
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / FOREACH_POINTWISE_LIST_DIVIDER;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_SIGMOID_OP_CODE) {
        // foreach_sigmoid
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 1024) / BINARY_LIST_UB_DIVIDER;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachTiling::FillTilingData() {
    tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
    tilingData.set_tensorDataCountList(tensorDataCountList);
    tilingData.set_tensorStartList(tensorStartList);
    tilingData.set_tensorEndList(tensorEndList);
    tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
    tilingData.set_tensorEndOffsetList(tensorEndOffsetList);
 
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus Tiling4ForeachTiling(gert::TilingContext* context) {
    ForeachTiling tilingObject(context);
    if (tilingObject.Init(0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachLogTiling(gert::TilingContext* context) {
    ForeachTiling tilingObject(context);
    if (tilingObject.Init(SOLO_LOG_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachTilingV2(gert::TilingContext* context) {
    ForeachTiling tilingObject(context);
    if (tilingObject.Init(BINARY_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepare4ForeachTiling(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachMulScalarInplace)
    .Tiling(Tiling4ForeachTiling)
    .TilingParse<ForeachSoloCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachLogInplace)
    .Tiling(Tiling4ForeachLogTiling)
    .TilingParse<ForeachSoloCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachSubListInplace)
    .Tiling(Tiling4ForeachTilingV2)
    .TilingParse<ForeachSoloCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachMulListInplace)
    .Tiling(Tiling4ForeachTilingV2)
    .TilingParse<ForeachSoloCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachDivListInplace)
    .Tiling(Tiling4ForeachTilingV2)
    .TilingParse<ForeachSoloCompileInfo>(TilingPrepare4ForeachTiling);
}  // namespace optiling

