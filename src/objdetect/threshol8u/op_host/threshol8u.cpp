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
 * @file threshol8u.cpp
 */
#include "threshol8u_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  Threshol8uTilingData tiling;
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto coreNum = ascendcPlatform.GetCoreNum();
  auto socVersion = ascendcPlatform.GetSocVersion();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  
  int64_t x1_dimNum = x1_shape->GetStorageShape().GetDimNum();

  int64_t inputNum = 1;
  for(int i = 0; i < x1_dimNum; i++){
    inputNum *= x1_shape->GetStorageShape().GetDim(i);
  }
  uint32_t typeLength = 0;
  ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
  int64_t inputLength = inputNum * typeLength;
  int64_t inputBytes = inputLength / inputNum;

  int64_t ubDataNumber = 4;

  int64_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
  int64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

  int64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
  coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
  coreNum = (coreNum >= 1) ? coreNum : 1;
  if(coreNum == 0) {
    return ge::GRAPH_FAILED;
  }
  int64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
  int64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
  
  int64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
  int64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
  int64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
  int64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
  smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
  
  everyCoreInputBlockNum += 1;
  int64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
  int64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
  int64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
  int64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
  bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
  
  tiling.set_smallCoreDataNum(smallCoreDataNum);
  tiling.set_bigCoreDataNum(bigCoreDataNum);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_smallTailDataNum(smallTailDataNum);
  tiling.set_bigTailDataNum(bigTailDataNum);
  tiling.set_finalSmallTileNum(finalSmallTileNum);
  tiling.set_finalBigTileNum(finalBigTileNum);
  tiling.set_tailBlockNum(tailBlockNum);
  
  context->SetBlockDim(coreNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Threshol8u : public OpDef {
public:
    explicit Threshol8u(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore()
            .AddConfig("ascend910b")
            .AddConfig("ascend310p");
    }
};

OP_ADD(Threshol8u);
}
