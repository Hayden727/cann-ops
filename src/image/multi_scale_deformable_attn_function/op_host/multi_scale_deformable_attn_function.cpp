/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file multi_scale_deformable_attn_function.cpp
 * \brief
 */
#include "runtime_util.h"
#include "op_log.h"

using namespace ge;
namespace {
constexpr size_t INPUT_VALUE_INDEX = 0;
constexpr size_t INPUT_LOCAT_INDEX = 3;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t INPUT_VALUE_DIM_0 = 0;
constexpr size_t INPUT_VALUE_DIM_3 = 3;
constexpr size_t INPUT_LOCAT_DIM_5 = 5;
constexpr size_t INPUT_LOCAT_DIM_2 = 2;
constexpr size_t INPUT_LOCAT_DIM_1 = 1;
constexpr size_t OUTPUT_DIM_0 = 0;
constexpr size_t OUTPUT_DIM_1 = 1;
constexpr size_t OUTPUT_DIM_2 = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForMultiScaleDeformableAttnFunction(gert::InferShapeContext *context)
{
    const gert::Shape *valueShape = context->GetInputShape(INPUT_VALUE_INDEX);
    if (valueShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *samplingLocationsShape = context->GetInputShape(INPUT_LOCAT_INDEX);
    if (samplingLocationsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (IsUnknownRank(valueShape)) {
        return SetUnknownRank(yShape);
    }
    bool isTranspose = samplingLocationsShape->GetDim(INPUT_LOCAT_DIM_1) < 32;
    uint64_t numHeads = INPUT_LOCAT_DIM_2;
    uint64_t numQueries = INPUT_LOCAT_DIM_1;
    if (isTranspose) {
        numHeads = INPUT_LOCAT_DIM_1;
        numQueries = INPUT_LOCAT_DIM_5;
    }
    const size_t outputRank = 3;
    yShape->SetDimNum(outputRank);

    yShape->SetDim(OUTPUT_DIM_0, valueShape->GetDim(INPUT_VALUE_DIM_0));
    yShape->SetDim(OUTPUT_DIM_1, samplingLocationsShape->GetDim(numQueries));                                                                                                        
    if (samplingLocationsShape->GetDim(INPUT_LOCAT_DIM_1) == -1) {
        yShape->SetDim(OUTPUT_DIM_2, samplingLocationsShape->GetDim(INPUT_LOCAT_DIM_1));
    } else {
        yShape->SetDim(OUTPUT_DIM_2, samplingLocationsShape->GetDim(numHeads) * valueShape->GetDim(INPUT_VALUE_DIM_3)); // 1: dim 2; 3: dim 3
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMultiScaleDeformableAttnFunction(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultiScaleDeformableAttnFunction).InferShape(InferShapeForMultiScaleDeformableAttnFunction).InferDataType(InferDataTypeForMultiScaleDeformableAttnFunction);
}  // namespace ops
