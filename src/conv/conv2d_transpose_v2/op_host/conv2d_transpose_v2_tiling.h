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
 * \file conv2d_transpose_v2_tiling.h
 * \brief
 */
#ifndef CONV2D_TRANSPOSE_V2_TILING_H
#define CONV2D_TRANSPOSE_V2_TILING_H
#include "conv3d_backprop_input_v2_tiling.h"
#include "conv2dbp_adapt_to_conv3dbp.h"
namespace optiling {
REGISTER_TILING_DATA_CLASS(Conv2DTransposeV2, Conv3DBackpropInputV2TilingData)

class Conv2DTransposeV2Tiling : public Conv3DBackpropInputV2Tiling {
public:
    explicit Conv2DTransposeV2Tiling(gert::TilingContext *context) : Conv3DBackpropInputV2Tiling(context)
    {
        Reset();
        opType_ = cachetiling::kConv3DTranspose;
    }
    ~Conv2DTransposeV2Tiling() override = default;
};
}
#endif  // CONV2D_TRANSPOSE_V2_TILING_H