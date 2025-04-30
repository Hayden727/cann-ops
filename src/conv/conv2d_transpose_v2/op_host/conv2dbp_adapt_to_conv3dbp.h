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
 * \file conv2dbp_adapt_to_conv3dbp.h
 * \brief
 */
#ifndef CONV2DBP_ADAPT_TO_CONV3DBP
#define CONV2DBP_ADAPT_TO_CONV3DBP
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "register/op_impl_registry.h"
#include "base/context_maker/kernel_run_context_maker.h"
namespace optiling {
ge::graphStatus AdaptTilingToConv3DBp(gert::TilingContext *context, std::string opType);
} // namespace optiling
#endif // CONV2DBP_ADAPT_TO_CONV3DBP