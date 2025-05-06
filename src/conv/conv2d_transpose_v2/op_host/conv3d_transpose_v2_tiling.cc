/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file conv3d_transpose_v2_tiling.cc
 * \brief
 */

#include "conv3d_backprop_input_v2_tiling.h"

#include "op_log.h"
#include "tiling/tiling_templates_registry.h"

namespace {
using Conv3DTransposeV2CompileInfo = optiling::Conv3DBackPropInputCompileInfo;
}  // namespace

namespace optiling {
REGISTER_TILING_TEMPLATE("Conv3DTransposeV2", Conv3DTransposeV2Tiling, 0);

static ge::graphStatus Conv3DTransposeV2TilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForConv3DTransposeV2(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<Conv3DTransposeV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();

    optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
    plaformInstance.SetInstance(*compileInfoPtr);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv3DTransposeV2)
    .Tiling(Conv3DTransposeV2TilingFunc)
    .TilingParse<Conv3DTransposeV2CompileInfo>(TilingParseForConv3DTransposeV2);
}  // namespace optiling
