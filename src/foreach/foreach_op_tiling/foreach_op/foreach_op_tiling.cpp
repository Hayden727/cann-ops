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
 * \file foreach_op_tiling.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "op_log.h"
#include "mulitensor_apply_tiling.h"

namespace optiling {

constexpr int32_t UNARY_DEPTH = 2;
constexpr int32_t BINARY_DEPTH = 3;
constexpr int32_t TENARY_DEPTH = 4;
constexpr int32_t MAX_TENSOR_CONT = 256;
constexpr uint32_t RESERVED_UB = 512;
constexpr int32_t DTYPE_SIZE_32 = 4;
constexpr int32_t DTYPE_SIZE_16 = 2;

// 18 template use the same tilingfunction:multiTensorApplyTiling
// need add code to suppport 18 template

/* Convert ge shape to C++ vector */
ge::graphStatus GetShapeInfo(gert::TilingContext* tilingContext,
                             std::vector<std::vector<std::vector<int64_t>>>& allInputShapes, const int32_t depth) {
  for (int32_t t_idx = 0; t_idx < depth; t_idx++) {
    std::vector<std::vector<int64_t>> inputTensorList;
    for (int32_t i = 0; i < MAX_TENSOR_CONT; i++) {
      std::vector<int64_t> inputTensor;
      auto srcTensor = tilingContext->GetDynamicInputTensor(t_idx, i);
      if (srcTensor == nullptr) {
        break;
      }
      gert::Shape tempShape = srcTensor->GetStorageShape();
      auto dimNum = tempShape.GetDimNum();
      for (size_t dimIdx = 0; dimIdx < dimNum; dimIdx++) {
        inputTensor.push_back(tempShape.GetDim(dimIdx));
      }
      inputTensorList.push_back(inputTensor);
    }
    allInputShapes.push_back(inputTensorList);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ForeachOp(gert::TilingContext* tilingContext, const int32_t depth) {
  std::vector<std::vector<std::vector<int64_t>>> allInputShapes;

  GetShapeInfo(tilingContext, allInputShapes, depth);
  ge::DataType dataType = ge::DT_UNDEFINED;
  auto srcTensor = tilingContext->GetDynamicInputTensor(0, 0);
  if (srcTensor == nullptr) {
    OP_LOGE(tilingContext->GetNodeName(), "The first input is empty tensor list.");
    return ge::GRAPH_FAILED;
  }
  auto srcDtype = tilingContext->GetInputDesc(0)->GetDataType();
  // Determine whether all data types are consistent.
  dataType = srcDtype;
  if (dataType == ge::DT_UNDEFINED) {
    OP_LOGE(tilingContext->GetNodeName(), "Param dataTypeSize is zero.");
    return ge::GRAPH_FAILED;
  }

  auto platformInfo = tilingContext->GetPlatformInfo();

  // Get corenum and ubsize
  uint64_t ubSizePlatForm;
  platformInfo->GetLocalMemSize(fe::LocalMemType::UB, ubSizePlatForm);
  uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
  uint32_t availableUb = ubSize - RESERVED_UB;
  uint32_t coreNum = static_cast<uint32_t>(platformInfo->GetCoreNum());

  OP_LOGD(tilingContext->GetNodeName(), "ubSizePlatForm:%lu, coreNum:%u", ubSizePlatForm, coreNum);

  int32_t opCode = *tilingContext->GetAttrs()->GetAttrPointer<int>(0);

  // Call tiling and Set tiling
  MultiTensorApplyTilingData tilingData;
  // when depth is UNARY_DEPTH or TENARY_DEPTH, a new branch should be added
  if (depth == BINARY_DEPTH) {
    if (dataType == ge::DT_FLOAT || dataType == ge::DT_INT32) {
      MultiTensorApplyGetTiling<BINARY_DEPTH, MultiTensorApplyTilingData, DTYPE_SIZE_32, DTYPE_SIZE_32>(
          opCode, allInputShapes, &tilingData, coreNum, availableUb);
    } else if (dataType == ge::DT_FLOAT16) {
      MultiTensorApplyGetTiling<BINARY_DEPTH, MultiTensorApplyTilingData, DTYPE_SIZE_16, DTYPE_SIZE_16>(
          opCode, allInputShapes, &tilingData, coreNum, availableUb);
    }
  }
  // Setting tilingdata
  tilingContext->SetTilingKey(opCode);
  tilingContext->SetBlockDim(tilingData.get_blockNum());
  tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                          tilingContext->GetRawTilingData()->GetCapacity());
  tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

// when depth is UNARY_DEPTH or TENARY_DEPTH, a new tiling function should be added
static ge::graphStatus LaunchTiling4ForeachBinaryOp(gert::TilingContext* tilingContext) {
  int32_t depth = BINARY_DEPTH;
  Tiling4ForeachOp(tilingContext, depth);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ForeachOp(gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

// Register op tiling

IMPL_OP_OPTILING(ForeachBinaryOp)
    .Tiling(LaunchTiling4ForeachBinaryOp)
    .TilingParse<Tiling4ForeachBinaryOpCompileInfo>(TilingPrepare4ForeachOp);
}  // namespace optiling
