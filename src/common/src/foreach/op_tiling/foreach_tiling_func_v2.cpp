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
 * \file foreach_tiling_func_v2.cpp
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_V2_FUNC_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_V2_FUNC_H_

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "foreach/op_tiling/foreach_tiling_def.h"
#include "foreach/op_tiling/common_dtype.h"

namespace optiling {
    constexpr uint64_t TILING_HALF_N_SCALAR = 14;
    constexpr uint64_t TILING_FLOAT_N_SCALAR = 4;
    constexpr uint64_t TILING_INT_N_SCALAR = 4;
    constexpr uint64_t TILING_BF16_N_SCALAR = 14;
    constexpr uint32_t TILING_FLOAT_ERF = 5;
    constexpr uint32_t TILING_HALF_ERF = 12;

    constexpr uint64_t WORK_SPACE_SIZE = 32;// foreach(vector) not need workspace

    constexpr uint32_t TANH_HALF_CALC_PROC = 5;
    constexpr uint32_t TANH_FLOAT_CALC_PROC = 6;
    constexpr uint32_t FOREACH_TANH_DIVIDER = 2;
    constexpr uint32_t SIN_HALF_CALC_FAC = 6;
    constexpr uint32_t SIN_FLOAT_CALC_FAC = 2;
    constexpr uint32_t SIN_BASIC_BLOCK = 2048;

    constexpr uint32_t COSH_HALF_CALC_PROC = 6;
    constexpr uint32_t COSH_FLOAT_CALC_PROC = 2;
    constexpr uint32_t COSH_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t SINH_HALF_CALC_PROC = 4;
    constexpr uint32_t SINH_FLOAT_CALC_PROC = 1;
    constexpr uint32_t SINH_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t ATAN_HALF_CALC_PROC = 10;
    constexpr uint32_t ATAN_FLOAT_CALC_PROC = 4;
    constexpr uint32_t ATAN_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t TAN_HALF_CALC_PROC = 10;
    constexpr uint32_t TAN_FLOAT_CALC_PROC = 4;
    constexpr uint32_t TAN_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t SIGN_CALC_PROC = 3;
    constexpr uint32_t SIGN_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t BINARY_LIST_UB_DIVIDER = 6;
    constexpr uint32_t BINARY_SCALAR_UB_DIVIDER = 4;
    constexpr uint32_t FOREACH_POINTWISE_DIVIDER = 8;
    constexpr uint32_t FOREACH_POW_SCALAR_DIVIDER = 4;
    constexpr uint32_t FOREACH_COS_DIVIDER = 4;
    constexpr uint32_t FOREACH_POINTWISE_LIST_DIVIDER = 8;
    constexpr uint32_t FOREACH_LERP_SCALAR_UB_DIVIDER = 6;
    constexpr uint32_t FOREACH_LERP_LIST_UB_DIVIDER = 11;
    constexpr uint32_t FOREACH_SIN_DIVIDER = 4;
    constexpr uint32_t FOREACH_ERF_BUFFER_DIVIDER = 4;
    constexpr uint32_t FOREACH_ERF_FLOAT_DIVIDER = 4; // erf float 预留 3 倍的输入空间
    constexpr uint32_t FOREACH_ERF_HALF_DIVIDER = 9; // erf half 预留 8 倍的输入空间
    constexpr uint32_t FOREACH_ERFC_FLOAT_DIVIDER = 8; // erfc float 预留 7 倍的输入空间
    constexpr uint32_t FOREACH_ERFC_HALF_DIVIDER = 17; // erfc half 预留 16 倍的输入空间

    constexpr uint8_t ZERO_OP_CODE = 1;
    constexpr uint8_t SOLO_LOG_OP_CODE = 2;
    constexpr uint8_t BINARY_LIST_OP_CODE = 3;
    constexpr uint8_t FOREACH_POINTWISE_OP_CODE = 4;
    constexpr uint8_t FOREACH_COS_OP_CODE = 5;
    constexpr uint8_t SOLO_LOG2_OP_CODE = 6;
    constexpr uint8_t SOLO_NEG_OP_CODE = 7;
    constexpr uint8_t FOREACH_POW_TENSOR_OP_CODE = 8;
    constexpr uint8_t FOREACH_BINARY_SCALAR_OP_CODE = 9;
    constexpr uint8_t FOREACH_POINTWISE_LIST_OP_CODE = 10;
    constexpr uint8_t FOREACH_SIGMOID_OP_CODE = 11;
    constexpr uint8_t FOREACH_ERF_OP_CODE = 12;
    constexpr uint8_t FOREACH_COSH_OP_CODE = 13;
    constexpr uint8_t FOREACH_ASIN_OP_CODE = 13;
    constexpr uint8_t FOREACH_ACOS_OP_CODE = 13;
    constexpr uint8_t FOREACH_SINH_OP_CODE = 14;
    constexpr uint8_t FOREACH_TAN_OP_CODE = 15;
    constexpr uint8_t FOREACH_ERFC_OP_CODE = 16;
    constexpr uint8_t FOREACH_TANH_OP_CODE= 17;
    constexpr uint8_t FOREACH_ATAN_OP_CODE = 18;
    constexpr uint8_t FOREACH_LERP_SCALAR_OP_CODE = 19;
    constexpr uint8_t FOREACH_LERP_LIST_OP_CODE = 20;
    constexpr uint8_t FOREACH_POW_SCALAR_OP_CODE = 21;
    constexpr uint8_t FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE = 22;
    constexpr uint8_t FOREACH_SIN_OP_CODE = 23;
    constexpr uint8_t FOREACH_ABS_OP_CODE = 24;
    constexpr uint8_t FOREACH_MUL_SCALAR_OP_CODE = 25;
    constexpr uint8_t FOREACH_EXP_OP_CODE = 26;
    constexpr uint8_t FOREACH_MAXIMUM_LIST_OP_CODE = 27;
    constexpr uint8_t FOREACH_ADD_LIST_OP_CODE = 28;
    constexpr uint8_t FOREACH_ROUND_OFF_NUM_OP_CODE = 29;
    constexpr uint8_t FOREACH_SUB_SCALAR_OP_CODE = 30;
    constexpr uint8_t FOREACH_DIV_SCALAR_OP_CODE = 31;
    constexpr uint8_t FOREACH_COPY_OP_CODE = 32;
    constexpr uint8_t FOREACH_SIGN_OP_CODE = 33;

    constexpr uint16_t LOG2_BASIC_FOR_LOG2 = 1024;
    constexpr uint32_t LOG2_HALF_FOR_LOG2 = 4;
    constexpr uint32_t LOG2_FLOAT_FOR_LOG2 = 0;

    constexpr uint8_t BYTE_PER_BLOCK = 32;
    constexpr uint32_t BYTE_PER_REPEAT = 256;
    constexpr int32_t POW_TENSOR_TENSOR_CALC_PROC[9] = {12, 3, 5, 3, 12, 12, 12, 12, 12};

    constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class ForeachCommonV2Tiling {
public:
    explicit ForeachCommonV2Tiling(gert::TilingContext* context) : tilingContext(context){};
    /**
     ** function: Init
    */
    ge::graphStatus Init(uint8_t theCode = 0) {
        opCode = theCode;
        int dynamicIdx = opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE ? 1 : 0;
        for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
            auto srcTensor = tilingContext->GetDynamicInputTensor(dynamicIdx, i);
            if (srcTensor == nullptr) {
                break;
            }

            auto temp = tilingContext->GetInputDesc(0);
            if (temp == nullptr) {
                return ge::GRAPH_FAILED;
            }

            auto srcDtype = temp->GetDataType();

            if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
                if (tilingContext->GetInputDesc(1) != nullptr) {
                    srcDtype = tilingContext->GetInputDesc(1)->GetDataType();
                } else {
                    return ge::GRAPH_FAILED;
                }
            }
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
            tensorDataCountList[i] = tempShape.GetShapeSize();
            totalDataCount += tensorDataCountList[i];
            totalTensorCount++;
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: RunBigKernelTiling
    */
    ge::graphStatus RunBigKernelTiling() {
        auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

        uint64_t ubSizePlatForm = 0;
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

        tilingContext->SetTilingKey(GetTilingKeyByDtypeOnly(dataType));

        needvecCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

        DivideUbMemory(ubSizePlatForm);
        FillTilingData();
        tilingContext->SetBlockDim(needvecCoreNum);
        size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspaces[0] = WORK_SPACE_SIZE;

        return ge::GRAPH_SUCCESS;
    }

private:
    /**
     ** function: CeilA2B
    */
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    /**
     ** function: GetTilingN
    */
    uint64_t GetTilingN() {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_FLOAT_N_SCALAR;
            case ge::DT_FLOAT16:
                return TILING_HALF_N_SCALAR;
            case ge::DT_INT32:
                return TILING_INT_N_SCALAR;
            case ge::DT_BF16:
                return TILING_BF16_N_SCALAR;
            default:
                return TILING_HALF_N_SCALAR;
        }
    }

    /**
     ** function: GetNeedCoreNum
    */
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) {
        uint32_t tempCoreNum = (uint32_t)CeilA2B(totalDataCount, elementsPerBlock);
        if (tempCoreNum == 0) {
            tempCoreNum = 1;
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    /**
     ** funtion: DivideUbMemory
    */
    void DivideUbMemory(uint64_t ubSizePlatForm) {
        if (opCode <= FOREACH_POINTWISE_OP_CODE) {
            DivideUbMemory1(ubSizePlatForm);
        } else if (opCode <= FOREACH_POW_TENSOR_OP_CODE) {
            DivideUbMemory2(ubSizePlatForm);
        } else if (opCode <= FOREACH_ERF_OP_CODE) {
            DivideUbMemory3(ubSizePlatForm);
        } else if (opCode <= FOREACH_TAN_OP_CODE) {
            DivideUbMemory4(ubSizePlatForm);
        } else if (opCode <= FOREACH_ATAN_OP_CODE) {
            DivideUbMemory5(ubSizePlatForm);
        } else if (opCode <= FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            DivideUbMemory6(ubSizePlatForm);
        } else if (opCode <= FOREACH_MUL_SCALAR_OP_CODE) {
            DivideUbMemory7(ubSizePlatForm);
        } else if (opCode <= FOREACH_ADD_LIST_OP_CODE) {
            DivideUbMemory8(ubSizePlatForm);
        } else if (opCode <= FOREACH_DIV_SCALAR_OP_CODE) {
            DivideUbMemory9(ubSizePlatForm);
        } else if (opCode <= FOREACH_COPY_OP_CODE) {
            DivideUbMemory9(ubSizePlatForm);
        } else if (opCode <= FOREACH_SIGN_OP_CODE) {
            DivideUbMemory10(ubSizePlatForm);
        }
    }

    /**
     ** funtion: DivideUbMemory1
    */
    void DivideUbMemory1(uint64_t ubSizePlatForm) {
        if (opCode == ZERO_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_add_scalar/add_scalar_list/expm1/sqrt/zero_inplace
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 2;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == SOLO_LOG_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_log/log1p/log10
            uint32_t totalSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 2;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == BINARY_LIST_OP_CODE) {
            // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
            // foreach_div_list/minimum_list/mul_list/sub_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_POINTWISE_OP_CODE) {
            // foreach_addcdiv_scalar/addcdiv_scalar_list/addcmul_scalar/addcmul_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_POINTWISE_DIVIDER; // double buffer
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory2
    */
    void DivideUbMemory2(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_COS_OP_CODE) {  // foreach_cos
            uint32_t tilingConstant = 6;
            if (dataTypeSize == BYTE_LEN_4) {
                tilingConstant = TILING_FLOAT_N_SCALAR;
            }
            uint32_t reserveUbSize = BYTE_BASIC_BLOCK * tilingConstant * dataTypeSize;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - reserveUbSize);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == SOLO_LOG2_OP_CODE) {  // foreach_log2
            uint32_t extraBuf = 0;      // need extra space
            GetLog2TmpBufferFactorSize(dataTypeSize, extraBuf, LOG2_HALF_FOR_LOG2, LOG2_FLOAT_FOR_LOG2, LOG2_BASIC_FOR_LOG2); // reuse source is true
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 2 - extraBuf;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == SOLO_NEG_OP_CODE) {  // need extra buffer of one block: 32 bytes  foreach_neg/reciprocal
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 2 - BYTE_PER_BLOCK;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_POW_TENSOR_OP_CODE) { // foreach_pow_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            uint32_t canUseUbSize;
            if (dataType == ge::DT_BF16) {
                canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER * UB_DIVIDER_FOR_TEMP_CASTING + POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType)-1]);
            } else{
                canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER + POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType)-1]);
            }
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory3
    */
    void DivideUbMemory3(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_BINARY_SCALAR_OP_CODE) {
            // foreach_maximum_scalar/maximum_scalar_list/minimum_scalar/minimum_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_SCALAR_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_POINTWISE_LIST_OP_CODE) {
            // foreach_addcdiv_list, foreach_addcmul_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_POINTWISE_LIST_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_SIGMOID_OP_CODE) {
            // foreach_sigmoid
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 1024);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_SCALAR_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_ERF_OP_CODE) {
            // foreach_erf
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            // erf ascend C need 3 times for float or 8 times for half inputData size reserved for every buffer 
            uint32_t canUseUbSize = totalSize / FOREACH_ERF_FLOAT_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            if (dataTypeSize == BYTE_LEN_2) {
                canUseUbSize = totalSize / FOREACH_ERF_HALF_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            }
            // 32 bytes align
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory4
    */
    void DivideUbMemory4(uint64_t ubSizePlatForm) {
        if ((opCode == FOREACH_ASIN_OP_CODE)) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_cosh/asin/acos
            uint32_t calcPro = COSH_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = COSH_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * dataTypeSize * COSH_BASIC_BLOCK_SIZE * 8;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_SINH_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_sinh
            uint32_t calcPro = SINH_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = SINH_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * dataTypeSize * SINH_BASIC_BLOCK_SIZE;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_TAN_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_tan
            uint32_t calcPro = TAN_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = TAN_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * dataTypeSize * TAN_BASIC_BLOCK_SIZE * 8;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory5
    */
    void DivideUbMemory5(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_ERFC_OP_CODE) {
            // foreach_erfc
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            // erfc ascend C need 7 times for float or 16 times for half inputData size reserved for every buffer
            uint32_t canUseUbSize = totalSize / FOREACH_ERFC_FLOAT_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            if (dataTypeSize == BYTE_LEN_2) {
                canUseUbSize = totalSize / FOREACH_ERFC_HALF_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            }
            // 32 bytes align
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_TANH_OP_CODE) {
            // foreach_tanh
            uint32_t calcPro = TANH_FLOAT_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_2) {
                calcPro = TANH_HALF_CALC_PROC;
            }
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 1024);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / (calcPro + FOREACH_TANH_DIVIDER);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
       	} else if (opCode == FOREACH_ATAN_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_atan
            uint32_t calcPro = ATAN_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = ATAN_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * dataTypeSize * ATAN_BASIC_BLOCK_SIZE * 8;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory6
    */
    void DivideUbMemory6(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_LERP_SCALAR_OP_CODE) {
            // foreach_lerp_scalar
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 128);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_LERP_SCALAR_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_LERP_LIST_OP_CODE) {
            // foreach_lerp_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_LERP_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
            inputsTensorUbSize = inputsTensorUbSize / BYTE_PER_REPEAT * BYTE_PER_REPEAT;
        } else if ((opCode == FOREACH_POW_SCALAR_OP_CODE) || (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE)) {
            // foreach_pow_scalar/pow_scalar_list/pow_scalar_and_tensor
            uint32_t reserveUbSize = BYTE_BASIC_BLOCK * GetTilingN() * dataTypeSize;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - reserveUbSize);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_POW_SCALAR_DIVIDER; // double buffer
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory7
    */
    void DivideUbMemory7(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_SIN_OP_CODE) {
            // foreach_sin
            uint32_t calcPro = SIN_HALF_CALC_FAC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = SIN_FLOAT_CALC_FAC;
            }
            uint32_t reservedUbSize = 4 * SIN_BASIC_BLOCK * calcPro * dataTypeSize;
            uint32_t totalSize = static_cast<uint32_t>(ubSizePlatForm - static_cast<uint32_t>(tilingData.GetDataSize()) - reservedUbSize);
            if (dataType == ge::DT_BF16) {
                totalSize = static_cast<uint32_t>(totalSize / UB_DIVIDER_FOR_TEMP_CASTING);
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / FOREACH_SIN_DIVIDER); // 4
            inputsTensorUbSize = static_cast<uint32_t>(canUseUbSize / BYTE_BLOCK * BYTE_BLOCK);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_ABS_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_abs
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 2048);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BYTE_LEN_4;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_MUL_SCALAR_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_mul_scalar/mul_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 2;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory8
    */
    void DivideUbMemory8(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_EXP_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_exp
            uint32_t totalSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BYTE_LEN_4;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_MAXIMUM_LIST_OP_CODE) {
            // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
            // foreach_maximum_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_ADD_LIST_OP_CODE) {
            // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
            // foreach_add_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory9
    */
    void DivideUbMemory9(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_ROUND_OFF_NUM_OP_CODE) {
            // foreach_round_off_number
            uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / 2;
            uint32_t predictSGUbSize = uint32_t(BYTE_REPEAT / (BYTE_REPEAT + 2.0 * dataTypeSize) * canUseUbSize);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                predictSGUbSize = predictSGUbSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
                predictSGUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_SUB_SCALAR_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_sub_scalar/sub_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - BYTE_BLOCK);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 4; // one block bytes is 32
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_DIV_SCALAR_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_div_scalar/div_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - BYTE_BLOCK);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 4; // one block bytes is 32
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_COPY_OP_CODE) {
            // The remaining UB size is one buffer enabled, and rounded down 32 bytes.
            // foreach_copy
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize;
            inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory10
    */
    void DivideUbMemory10(uint64_t ubSizePlatForm) {
        if (opCode == FOREACH_SIGN_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_sign
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_FLOAT || dataType == ge::DT_FLOAT16) {
                uint32_t extraBuffer = SIGN_CALC_PROC * dataTypeSize * SIGN_BASIC_BLOCK_SIZE * 8;
                totalSize = totalSize - extraBuffer;
            }
            if (dataType == ge::DT_BF16 || dataType == ge::DT_INT64 || dataType == ge::DT_INT8) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 4; // one block bytes is 32
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
                canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** function: FillTilingData
    */
    void FillTilingData() {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_needCoreNum(needvecCoreNum);

        tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                                tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

    /**
     ** function: GetLog2TmpBufferFactorSize
    */
    void GetLog2TmpBufferFactorSize(const uint32_t typeSize, uint32_t &extraBuf,
                                    uint32_t LOG2_HALF = LOG2_HALF_FOR_LOG2, uint32_t LOG2_FLOAT = LOG2_FLOAT_FOR_LOG2,
                                    uint32_t LOG2_BASIC = LOG2_BASIC_FOR_LOG2) {
        auto caclFactor = (typeSize == sizeof(float)) ? LOG2_FLOAT : LOG2_HALF;
        extraBuf = LOG2_BASIC * caclFactor * typeSize;
    }

private:
    ForeachCommonV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    int64_t totalDataCount = 0;
    uint8_t dataTypeSize = 4;
    uint8_t elementsPerBlock = 0;
    uint16_t totalTensorCount = 0;
    uint8_t opCode = 0;
    uint32_t needvecCoreNum = 0;
};

}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_V2_FUNC_H_
