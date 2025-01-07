/* 
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "register/register.h"

namespace domi {
// register op info to GE
REGISTER_CUSTOM_OP("AddCustom")
    .FrameworkType(TENSORFLOW)   // type: CAFFE, TENSORFLOW
    .OriginOpType("Add")      // name in tf module
    .ParseParamsByOperatorFn(AutoMappingByOpFn);
}  // namespace domi
