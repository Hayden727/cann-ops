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
 * \file add_rms_norm_quant_v2_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
}

static constexpr int INPUT_X1_IDX = 0;
static constexpr int INPUT_SCALE2_IDX = 4;
static constexpr int OUTPUT_Y1_IDX = 0;
static constexpr int OUTPUT_Y2_IDX = 1;
static constexpr int OUTPUT_X_IDX = 2;

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4AddRmsNormQuantV2(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4AddRmsNormQuantV2");

    const gert::Shape *x1Shape = context->GetInputShape(INPUT_X1_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);

    gert::Shape *y1Shape = context->GetOutputShape(OUTPUT_Y1_IDX);
    gert::Shape *y2Shape = context->GetOutputShape(OUTPUT_Y2_IDX);
    gert::Shape *xShape = context->GetOutputShape(OUTPUT_X_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    
    *y1Shape = *x1Shape;
    *xShape = *x1Shape;

    const gert::Shape *scale2Shape = context->GetOptionalInputShape(INPUT_SCALE2_IDX);
    if (nullptr != scale2Shape && (scale2Shape->GetDimNum() != 0)) {
        *y2Shape = *x1Shape;
    } else {
        *y2Shape = gert::Shape({1});
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4AddRmsNormQuantV2");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4AddRmsNormQuantV2(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4AddRmsNormQuantV2");
    context->SetOutputDataType(OUTPUT_Y1_IDX, DT_INT8);
    context->SetOutputDataType(OUTPUT_Y2_IDX, DT_INT8);
    context->SetOutputDataType(OUTPUT_X_IDX, context->GetInputDataType(INPUT_X1_IDX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4AddRmsNormQuantV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormQuantV2)
    .InferShape(InferShape4AddRmsNormQuantV2)
    .InferDataType(InferDataType4AddRmsNormQuantV2);

}  // namespace ops