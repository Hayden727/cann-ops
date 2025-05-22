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
 * \file foreach_reduce_tiling_func_v2.cpp
 * \brief
 */
 
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_V2_FUNC_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_V2_FUNC_H_

#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "foreach/op_tiling/foreach_tiling_def.h"
#include "foreach/op_tiling/common_dtype.h"

namespace optiling {
constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class ForeachReduceV2Tiling {
public:
    explicit ForeachReduceV2Tiling(gert::TilingContext* context) : tilingContext(context){};
    /**
     ** function: Init
    */
    ge::graphStatus Init() {
        // Get shape, dtype information, and the total number of data.
        for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
            auto srcTensor = tilingContext->GetDynamicInputTensor(0, i);
            if (srcTensor == nullptr) {
                break;
            }
            auto srcDtype = srcTensor->GetDataType();
            // Determine whether all data types are consistent.
            if (dataType == ge::DT_UNDEFINED) {
                dataType = srcDtype;
                dataTypeSize = GetDataTypeSize(dataType);
                if (dataTypeSize == 0) {
                    dataTypeSize = BYTE_LEN_4;
                }
                elementsPerBlock = BYTE_BLOCK / dataTypeSize;
            } else if (srcDtype != dataType) {
                return ge::GRAPH_FAILED;
            }
            gert::Shape tempShape = srcTensor->GetStorageShape();
            // Make a 32-byte alignment for each Tensor
            tensorDataCountList[i] = (uint64_t)tempShape.GetShapeSize();
            if (tensorDataCountList[i] == 0) {
                isExistEmptyTensor = true;
            }
            totalBlockCount += static_cast<uint64_t>(CeilA2B(tensorDataCountList[i], elementsPerBlock));
            totalTensorCount++;
        }

        return ge::GRAPH_SUCCESS;
    }
    
    ge::graphStatus RunBigKernelTiling() {
        auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

        uint64_t ubSizePlatForm = 0;

        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

        tilingContext->SetTilingKey(GetTilingKeyVal());

        needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

        DivideUbMemory(ubSizePlatForm);

        FillTilingData();

        tilingContext->SetBlockDim(needCoreNum);

        size_t usrSize = (MAX_CORE_CONT + MAX_TENSOR_CONT) * sizeof(float);
        size_t sysWorkspaceSize = WORK_SPACE_SIZE; 
        size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        if (currentWorkspace==nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = usrSize + sysWorkspaceSize;
        
        return ge::GRAPH_SUCCESS;
    }

private:
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    uint64_t GetTilingKeyVal() {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_KEY_FLOAT;
            case ge::DT_FLOAT16:
                return TILING_KEY_HALF;
            case ge::DT_BF16:
                return TILING_KEY_BF16;
            default:
                return 0;
        }
    }

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) {
        uint32_t tempCoreNum = (uint32_t)totalBlockCount;
        if (tempCoreNum == 0) {
            tempCoreNum = 1;
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    void DivideUbMemory(uint64_t ubSizePlatForm) {
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 16384);
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }

    void FillTilingData() {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_needCoreNum(needCoreNum);
        tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                                tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

private:
    ForeachCommonV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint64_t totalBlockCount = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;

    bool isExistEmptyTensor = false;

    uint32_t modelCode = 0;

    uint16_t tensorMiddleCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorMiddleStartList[MAX_TENSOR_CONT] = {0};
    uint16_t coreMiddleOffsetList[MAX_CORE_CONT] = {0};
};
} // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_V2_FUNC_H_
