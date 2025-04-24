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
 * @file mse_loss.cpp
 */

#include <cstring>
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "mse_loss_tiling.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    uint32_t AlignTotalLength(uint32_t totalLength, uint32_t ALIGN_NUM)
    {
        return (totalLength % ALIGN_NUM != 0) ? ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM : totalLength;
    }

    uint32_t CalculateBlockLength(uint32_t totalLengthAligned, uint32_t block_dim)
    {
        return (block_dim != 0) ? totalLengthAligned / block_dim : 0;
    }

    uint32_t CalculateTileNum(uint32_t blockLength, uint32_t ALIGN_NUM, uint32_t ub_block_num)
    {
        return (blockLength / ALIGN_NUM / ub_block_num == 0) ? 1 : blockLength / ALIGN_NUM / ub_block_num;
    }

    void CalculateTileLengths(uint32_t blockLength, uint32_t ALIGN_NUM, uint32_t ub_block_num, uint32_t tile_num, uint32_t& tileLength, uint32_t& lastTileLength)
    {
        if (ub_block_num != 0 && ((blockLength / ALIGN_NUM) % ub_block_num == 0 || tile_num == 0)) {
            if (tile_num == 0) {
                tile_num = 1;
            }
            if (blockLength < ub_block_num * ALIGN_NUM) {
                tileLength = ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
            } else {
                tileLength = ub_block_num * ALIGN_NUM;
            }
            lastTileLength = tileLength;
        } else {
            tile_num = tile_num + 1;
            tileLength = ub_block_num * ALIGN_NUM;
            lastTileLength = blockLength - (tile_num - 1) * tileLength;
        }
    }

    void SaveTilingData(const MseLossTilingData& tiling, gert::TilingContext* context)
    {
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        MseLossTilingData tiling;
        uint32_t sizeOfDataType = GetSizeOfDataType(context);
        uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeOfDataType;
        uint32_t ub_block_num = 1024;
        if (ub_block_num % 2 != 0) {
            ub_block_num = ub_block_num - 1;
        }
        size_t mode = GetReductionMode(context);
        tiling.set_mode(mode);

        uint32_t totalLengthAligned = AlignTotalLength(totalLength, ALIGN_NUM);
        tiling.set_totalLength(totalLength);

        context->SetBlockDim(1);
        uint32_t block_dim = context->GetBlockDim();
        uint32_t blockLength = CalculateBlockLength(totalLengthAligned, block_dim);
        uint32_t tile_num = CalculateTileNum(blockLength, ALIGN_NUM, ub_block_num);
        uint32_t tileLength = 0;
        uint32_t lastTileLength = 0;

        CalculateTileLengths(blockLength, ALIGN_NUM, ub_block_num, tile_num, tileLength, lastTileLength);

        tiling.set_blockLength(blockLength);
        tiling.set_tileNum(tile_num);
        tiling.set_tileLength(tileLength);
        tiling.set_lastTileLength(lastTileLength);

        SaveTilingData(tiling, context);

        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetSizeOfDataType(gert::TilingContext* context)
    {
        auto dt = context->GetInputDesc(0)->GetDataType();
        return (dt == 1) ? 2 : 1;
    }

    size_t GetReductionMode(gert::TilingContext* context)
    {
        const char* reduction = context->GetAttrs()->GetStr(0);
        const char* mode1 = "mean";
        const char* mode2 = "sum";
        const char* mode3 = "none";
        size_t mode = 0;

        if (strcmp(reduction, mode1) == 0) {
            mode = 1;
        } else if (strcmp(reduction, mode2) == 0) {
            mode = 2;
        } else if (strcmp(reduction, mode3) == 0) {
            mode = 3;
        }

        return mode;
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