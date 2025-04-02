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
    static void* ReadBinFile(const std::string filename, size_t& size) {
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
    static void FreeTensor(atb::Tensor &tensor)
    {
        if (tensor.deviceData) {
            int ret = aclrtFree(tensor.deviceData);
            if (ret != 0) {
                std::cout << "[ERROR]: aclrtFree fail" << std::endl;
            }
            tensor.deviceData = nullptr;
            tensor.dataSize = 0;
        }
        if (tensor.hostData) {
            int ret = aclrtFreeHost(tensor.hostData);
            if (ret != 0) {
                std::cout << "[ERROR]: aclrtFreeHost fail, ret = " << ret << std::endl;
            }
            tensor.hostData = nullptr;
            tensor.dataSize = 0;
        }
    }
    static void FreeTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::Tensor> &outTensors)
    {
        for (size_t i = 0; i < inTensors.size(); ++i) {
            FreeTensor(inTensors.at(i));
        }
        for (size_t i = 0; i < outTensors.size(); ++i) {
            FreeTensor(outTensors.at(i));
        }
    }
    
    static bool SaveMemoryToBinFile(void* memoryAddress, size_t memorySize, size_t i) {
        // 创建 output 目录（如果不存在）
        std::filesystem::create_directories("output");

        // 生成文件名
        std::string filename = "script/output/output_" + std::to_string(i) + ".bin";

        // 打开文件以二进制写入模式
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        // 写入数据
        file.write(static_cast<const char*>(memoryAddress), memorySize);
        if (!file) {
            std::cerr << "写入文件时出错: " << filename << std::endl;
            file.close();
            return false;
        }

        // 关闭文件
        file.close();
        std::cout << "数据已成功保存到: " << filename << std::endl;
        return true;
    }  
}
#endif