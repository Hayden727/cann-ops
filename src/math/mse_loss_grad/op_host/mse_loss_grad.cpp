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
 * @file mse_loss_grad.cpp
 */

#include <cstring>
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "mse_loss_grad_tiling.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;

    uint32_t GetSizeOfDataType(gert::TilingContext* context)
    {
        uint32_t sizeOfDataType;
        auto dt = context->GetInputDesc(0)->GetDataType();
        if (dt == 1) {
            sizeOfDataType = 2;
        }
        return sizeOfDataType;
    }

    uint32_t DetermineReductionMode(const char* reduction) {
        if (strcmp(reduction, "mean") == 0) {
            return 1;
        }
        if (strcmp(reduction, "sum") == 0) {
            return 2;
        }
        if (strcmp(reduction, "none") == 0) {
            return 3;
        }
        return 0;
    }

    uint32_t CalculateAlignedLength(uint32_t totalLength, uint32_t ALIGN_NUM) {
        if (ALIGN_NUM == 0) {
            return totalLength;
        }
        return (totalLength % ALIGN_NUM != 0) 
            ? ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM
            : totalLength;
    }

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MseLossGradTilingData tiling;

    uint32_t sizeOfDataType = GetSizeOfDataType(context);
    if (sizeOfDataType == 0) {
    return ge::GRAPH_FAILED;
    }

    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeOfDataType;
    uint32_t ub_block_num = 256;
    if (ub_block_num % 2 != 0) {
        ub_block_num = ub_block_num - 1;
    }

    const char* reduction = context->GetAttrs()->GetStr(0);
    tiling.set_mode(DetermineReductionMode(reduction));

    uint32_t totalLengthAligned = CalculateAlignedLength(totalLength, ALIGN_NUM);
    tiling.set_totalLength(totalLength);

    context->SetBlockDim(1);
    auto block_dim = context->GetBlockDim();

    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t lasttileLength = 0;

    // once_size为一次能搬入ub的数据的最大值（取决于ub_block_num的设置）
    uint32_t once_size = ALIGN_NUM * ub_block_num;
    blockLength = totalLengthAligned;
    tile_num = totalLength / once_size;
    
    // 数据总量小于once_size时采用切分策略1
    if (tile_num == 0) {
        tileLength = blockLength;
        lasttileLength = totalLength;
        tile_num += 1;
        context->SetTilingKey(1);
    } 
    // 数据总量大于once_size时采用切分策略2
    else {
        tileLength = once_size;
        lasttileLength = totalLength - tile_num * once_size;
        if (lasttileLength != 0) {
            tile_num += 1;
        }
        context->SetTilingKey(2);
    }

    // remain_start为lasttileLength向下32对齐的长度
    uint32_t remain_start = lasttileLength / 32 * 32;

    tiling.set_remain_start(remain_start);
    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tileLength);
    tiling.set_lasttileLength(lasttileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
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
class MseLossGrad : public OpDef {
public:
    explicit MseLossGrad(const char* name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("label")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dout")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                      .AddConfig("ascend910b");
    }
};
OP_ADD(MseLossGrad);
}