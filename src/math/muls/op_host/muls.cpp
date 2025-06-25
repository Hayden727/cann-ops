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
#include "muls_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MulsTilingData tiling;
    // 获取硬件信息（UB 内存大小、核心数、SOC版本）
    uint64_t ubLength = 0;
    uint32_t bigCoreDataNum = 0;
    uint32_t bigCoreLoopNum = 0;
    uint32_t bigCoreTailDataNum = 0;
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();
    
    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint32_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dataTypeLength);
    uint32_t inputLength = inputDataNum * dataTypeLength;
    std::cout<<"is me New_muls "<<std::endl;
    // There are a total of 3 shared UB spaces in the input and output. If it's bf16 and int64, there are 2 more TBUFs
    uint32_t dataType = context->GetInputDesc(0)->GetDataType();
    uint32_t ubPartNum = (dataType == ge::DT_BF16 || dataType == ge::DT_INT64) ? 5 : 3;
    uint32_t ubPartLength = ubLength / ubPartNum / BUFFER_NUM;
    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint32_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    uint32_t ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;

    // Input data for 32B alignment
    uint32_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    if(ubPartDataNum >= inputDataNum)
    {
        coreNum=1;
    }
    else
    {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
        coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    }
    
    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    
    // Small chunks are calculated and sliced several times using the number of data on each core
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
    uint32_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
    smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum ) == 0 ? smallCoreLoopNum : smallCoreLoopNum + 1;
    // Tail block calculation for small chunks of data
    uint32_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum-1);
    smallCoreTailDataNum = smallCoreTailDataNum == 0 ? ubPartDataNum : smallCoreTailDataNum;
    bool IsExistBigCore = true;
    if(0 != tailBlockNum)
    {
        everyCoreInputBlockNum += 1;
        bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
        bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
        bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum ) == 0 ? bigCoreLoopNum : bigCoreLoopNum + 1;
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum-1);
        bigCoreTailDataNum = bigCoreTailDataNum == 0 ? ubPartDataNum : bigCoreTailDataNum;
        IsExistBigCore = true;
    }
    else{
        IsExistBigCore = false;
    }
    

    if (dataType == ge::DT_BF16) {
        context->SetTilingKey(0);
    }else if(dataType == ge::DT_FLOAT16){
        context->SetTilingKey(1);
    }else if(dataType == ge::DT_FLOAT){
        context->SetTilingKey(2);
    }else if(dataType == ge::DT_INT16){
        context->SetTilingKey(3);
    }else if(dataType == ge::DT_INT32){
        context->SetTilingKey(4);
    }else if(dataType == ge::DT_INT64){
        context->SetTilingKey(5);
    }else if(dataType == ge::DT_COMPLEX64){
        context->SetTilingKey(6);
    }

    
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_ubPartDataNum(ubPartDataNum);
    tiling.set_smallCoreTailDataNum(smallCoreTailDataNum);
    tiling.set_bigCoreTailDataNum(bigCoreTailDataNum);
    tiling.set_smallCoreLoopNum(smallCoreLoopNum);
    tiling.set_bigCoreLoopNum(bigCoreLoopNum);
    tiling.set_tailBlockNum(tailBlockNum);
    tiling.set_tailBlockNum(IsExistBigCore);
    context->SetBlockDim(coreNum);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
}
namespace ops {
class Muls : public OpDef {
public:
    explicit Muls(const char* name) : OpDef(name)
    {
        //ge::DT_COMPLEX32
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,ge::DT_INT32, ge::DT_INT16,ge::DT_INT64, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,ge::DT_FLOAT, ge::DT_FLOAT,ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND}).Scalar();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,ge::DT_INT32, ge::DT_INT16,ge::DT_INT64, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};
OP_ADD(Muls);
}
