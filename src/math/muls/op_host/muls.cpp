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
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    //获取输入数据量的大小
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    // 获取输入数据的类型长度（如 float16=2, float32=4）
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    //获取输入长度以及输入类型
    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = 0;//inputBytes=typeLength
    if(inputNum != 0){
        inputBytes = inputLength / inputNum;//inputBytes=typeLength
    }else{
        return ge::GRAPH_FAILED;
    }
    //在上面部分，complex和其他数据类型是相同的处理方式，但在下面具体核分配中，要分流处理，先获取数据类型
    uint32_t dataType = context->GetInputDesc(0)->GetDataType();
    //根据不同的数据类型来这设置tilingkey
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
    }else if(dataType == ge::DT_COMPLEX32){
        context->SetTilingKey(6);
    }else if(dataType == ge::DT_COMPLEX64){
        context->SetTilingKey(7);
    }
    // 计算每个核心可处理的数据块大小（考虑 UB 内存限制）
    //ubDataNumber需要根据数据类型进行变动，其中complex为特殊数据，需要进行特殊处理
    constexpr uint64_t UB_DATA_NUM_NORMAL = 12;     // Normal data types
    constexpr uint64_t UB_DATA_NUM_FLOAT32 = 6;     // For float32/int32
    constexpr uint64_t UB_DATA_NUM_COMPLEX64 = 4; 
    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT||context->GetInputDesc(0)->GetDataType() == ge::DT_INT32) ? UB_DATA_NUM_FLOAT32 : UB_DATA_NUM_NORMAL;
    if(dataType == ge::DT_COMPLEX64){
        ubDataNumber = UB_DATA_NUM_COMPLEX64;
    }
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes; // 每次搬运的数据量
    // 数据对齐到 32B
    uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // 确定核心数（不超过数据块数）
    coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    uint32_t MIN_CORE_NUM = 1;
    coreNum = (coreNum >= MIN_CORE_NUM) ? coreNum : MIN_CORE_NUM;
    uint64_t everyCoreInputBlockNum = 0;
    uint64_t tailBlockNum = 0;
    // 计算大核和小核的分块参数
    if(BLOCK_SIZE!=0 && coreNum!=0){
        everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
        tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    }else{
        return ge::GRAPH_FAILED;
    }
    // 小核参数（处理较少数据）
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    //避免出现smallTailDataNum等于0的情况出现
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    // 大核参数（处理更多数据）
    everyCoreInputBlockNum += 1; // 大核多处理一个块
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    //避免出现bigTailDataNum==0的情况出现
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    // 将参数保存到 Tiling 结构体
    tiling.set_smallCoreDataNum((uint32_t)smallCoreDataNum);
    tiling.set_bigCoreDataNum((uint32_t)bigCoreDataNum);
    tiling.set_tileDataNum((uint32_t)tileDataNum);
    tiling.set_smallTailDataNum((uint32_t)smallTailDataNum);
    tiling.set_bigTailDataNum((uint32_t)bigTailDataNum);
    tiling.set_finalSmallTileNum((uint32_t)finalSmallTileNum);
    tiling.set_finalBigTileNum((uint32_t)finalBigTileNum);
    tiling.set_tailBlockNum((uint32_t)tailBlockNum);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *value = attrs->GetAttrPointer<float>(0);
    tiling.set_value(*value);
    // 设置核心数和 Tiling 数据
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
        float DEFAULT_ATTR_VALUE = 1.0f;
        this->Attr("value").AttrType(REQUIRED).Float(DEFAULT_ATTR_VALUE);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,ge::DT_INT32, ge::DT_INT16,ge::DT_INT64, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND,  ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p").AddConfig("ascend910b");

    }
};
OP_ADD(Muls);
}