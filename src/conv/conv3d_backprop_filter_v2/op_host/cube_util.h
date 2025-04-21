/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef OP_API_CUBE_UTIL_H_
#define OP_API_CUBE_UTIL_H_
#include "aclnn/aclnn_base.h"
// #include "common/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

// 校验针对cube tensor的dtype，cubeMathType的值是否符合预期
bool CheckCubeMathType(const op::DataType cubeTensorDtype, int8_t cubeMathType);

// 校验针对mm算子 tensor的dtype，cubeMathType的值是否符合预期
bool CheckCubeMathTypeForMm(const op::DataType cubeTensorDtype, int8_t cubeMathType);

// 返回芯片对应支持的数据类型
const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion();

// 根据promote type + cubemathtype的组合算出最终算子应用的dtype
op::DataType CalcPromoteTypeCubemathtype(const op::DataType cubeTensorPromoteType, int8_t cubeMathType);

// 根据promoteType + cubeMathType 判断是否要走HF32分支
bool NeedCubeGoHF32(const op::DataType cubeTensorPromoteType, int8_t cubeMathType);

// 检查针对x芯片，cube算子是否支持FP32
inline bool IsCubeSupportFp32() {
    if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910B &&
        op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_93) {
        return false;
    }
    return true;
}

// 检查针对x芯片，cube算子是否支持HF32
inline bool IsCubeSupportHf32() {
    if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910B &&
        op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_93) {
        return false;
    }
    return true;
}

#ifdef __cplusplus
}
#endif
#endif