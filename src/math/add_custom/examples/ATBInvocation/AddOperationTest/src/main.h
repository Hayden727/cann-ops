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
#include "aclnn_add_operation.h"
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
    aclError CheckAcl(aclError ret)
    {
        if (ret != ACL_ERROR_NONE) {
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << ret << std::endl;
        }
        return ret;
    }
    void* ReadBinFile(const char* filename, size_t& size) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return nullptr;
        }

        // 获取文件大小
        size = file.tellg();
        file.seekg(0, std::ios::beg);

        // 分配内存
        void* buffer;
        int ret = aclrtMallocHost(&buffer,size);
        if (!buffer) {
            std::cerr << "内存分配失败" << std::endl;
            file.close();
            return nullptr;
        }

        // 读取文件内容到内存
        file.read(static_cast<char*>(buffer), size);
        if (!file) {
            std::cerr << "读取文件失败" << std::endl;
            delete[] static_cast<char*>(buffer);
            file.close();
            return nullptr;
        }

        file.close();
        return buffer;
    }
}
#endif