/* 
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

/**
 * @file swish.cpp
 */
#include "swish_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SwishTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    std::cout<<"is me swish chen"<<std::endl;
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = inputLength / inputNum;

    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? 2 : 4;
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    if(tileDataNum >= inputNum)
    {
        coreNum=1;
    }
    else
    {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
        coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    }
    /*
    coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    coreNum = (coreNum >= 1) ? coreNum : 1;*/
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum; 
    
    tiling.set_smallCoreDataNum((uint32_t)smallCoreDataNum);
    tiling.set_bigCoreDataNum((uint32_t)bigCoreDataNum);
    tiling.set_tileDataNum((uint32_t)tileDataNum);
    tiling.set_smallTailDataNum((uint32_t)smallTailDataNum);
    tiling.set_bigTailDataNum((uint32_t)bigTailDataNum);
    tiling.set_finalSmallTileNum((uint32_t)finalSmallTileNum);
    tiling.set_finalBigTileNum((uint32_t)finalBigTileNum);
    tiling.set_tailBlockNum((uint32_t)tailBlockNum);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *scale = attrs->GetAttrPointer<float>(0);
    tiling.set_scale(*scale);

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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Swish : public OpDef {
public:
    explicit Swish(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend310b");
    }
};

OP_ADD(Swish);
}
