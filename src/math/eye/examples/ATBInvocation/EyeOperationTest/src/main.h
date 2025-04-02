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
 * @file main.h
 */
#ifndef MAIN_H
#define MAIN_H
#include <acl/acl.h>
#include "atb/atb_infer.h"
#include "aclnn_eye_operation.h"
#include "securec.h"

#include <fstream>
#include <random>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
namespace common{
    struct InputData{
        void* data;
        uint64_t size;
    };
    aclError CheckAcl(aclError ret);
    void* ReadBinFile(const string filename, size_t& size);
    bool SetInputData(std::vector<InputData> &inputData);
    bool SetOperationInputDesc(atb::SVector<atb::TensorDesc> &intensorDescs);

}
#endif