/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file mse_loss.cpp
 * \brief
 */
#include <cstring>
#include "mse_loss_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MseLossTilingData tiling;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto dt = context->GetInputDesc(0)->GetDataType();
    if (dt == 1) {
        sizeofdatatype = 2;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t ub_block_num = 1024;  
    uint32_t tile_num;

    if (ub_block_num % 2 != 0) {
        ub_block_num = ub_block_num - 1;
    }

    // 获取reduction的值，并设置传入kernel的mode值
    const char* reduction = context->GetAttrs()->GetStr(0);
    const char* mode1 = "mean";
    const char* mode2 = "sum";
    const char* mode3 = "none";
    size_t str_len = strlen(reduction);
    uint32_t mode = static_cast<int32_t>(0);
    
    if (str_len == strlen(mode1)) {
        for (size_t i = 0; i < str_len; i++) {
            if (reduction[i] != mode1[i]) {
                break;
            }
            if (i == str_len-1) {
                mode = 1;
            }
        }
    }
    if (str_len == strlen(mode2)) {
        for (size_t i = 0; i < str_len; i++) {
            if (reduction[i] != mode2[i]) {
                break;
            }
            if (i == str_len-1) {
                mode = 2;
            }
        }
    }
    if (str_len == strlen(mode3)) {
        for (size_t i = 0; i < str_len; i++) {
            if (reduction[i] != mode3[i]) {
                break;
            }
            if (i == str_len-1) {
                mode = 3;
            }
        }
    }

    tiling.set_mode(mode);

    if (totalLength % ALIGN_NUM != 0) {  
        totalLengthAligned =
            ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    } else {
        totalLengthAligned = totalLength;
    }

    tiling.set_totalLength(totalLength);

    // 环境为单核环境，故直接设置为1个核
    context->SetBlockDim(1);

    auto block_dim = context->GetBlockDim();
    if (block_dim == 0) {
        throw std::runtime_error("Block dimension cannot be zero.");
    }

    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t lasttileLength = 0;

    blockLength = totalLengthAligned / block_dim;
    tile_num = blockLength / ALIGN_NUM / ub_block_num;
    // 数据切分策略： 由于为单核环境，则将tileLength设置得尽可能大，最后单独处理剩余数据
    if (ub_block_num != 0 && ((totalLengthAligned / block_dim / ALIGN_NUM) % ub_block_num == 0 || tile_num == 0)) {  
        if (tile_num == 0) {
            tile_num = 1;
        } 
        if (blockLength < ub_block_num * ALIGN_NUM) {
            tileLength = ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
            lasttileLength = tileLength;
        } 
        else {
            tileLength = ub_block_num * ALIGN_NUM;
            lasttileLength = tileLength;
        }
    } 
    else {  
        tile_num = tile_num + 1;
        tileLength = ub_block_num * ALIGN_NUM;
        lasttileLength = blockLength - (tile_num - 1) * tileLength;
    }

    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tileLength);
    tiling.set_lasttileLength(lasttileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0);
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
class MseLoss : public OpDef {
public:
    explicit MseLoss(const char* name) : OpDef(name)
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

OP_ADD(MseLoss);
}
