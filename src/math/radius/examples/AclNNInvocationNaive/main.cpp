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
 * @file main.cpp
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnn_radius.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ACL_TYPE aclDataType::ACL_INT32
typedef int32_t COMPUTE_TYPE;
constexpr int32_t  TYPE_SIZE = 4;

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    const char* case_id = std::getenv("CASE_ID");
    int32_t caseId = -1;
    if(case_id != nullptr){
        caseId = std::atoi(case_id);
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputXShape;
    std::vector<int64_t> inputYShape;
    std::vector<int64_t> inputPtrXShape;
    std::vector<int64_t> inputPtrYShape;
    std::vector<int64_t> outputShape;

    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *inputPtrXDeviceAddr = nullptr;
    void *inputPtrYDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *inputPtrX = nullptr;
    aclTensor *inputPtrY = nullptr;
    aclTensor *output = nullptr;

    size_t inputXShapeSize_1=1;
    size_t inputYShapeSize_1=1;
    size_t outputShapeSize_1=1;

    for(int i = 1; i < argc; i++){
        outputShape.push_back(std::atoi(argv[i]));
        outputShapeSize_1 *= std::atoi(argv[i]);
    }
    bool isEmpty = false;
    if(outputShapeSize_1 == 0){
        isEmpty = true;
        outputShape[1] = 1;
        outputShapeSize_1 = 2;
    }
    std::vector<int32_t> inputPtrXHostData = {};
    std::vector<int32_t> inputPtrYHostData = {};

    float r;
    int max_num_neighbors = 32;
    bool ignore_same_index = false;

    if (caseId == 1) {
        inputXShape.push_back(100);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 200;

        inputYShape.push_back(50);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 100;

        r = 1.0f;
        max_num_neighbors = 10;
        ignore_same_index = false;
    } else if (caseId == 2) {
        inputXShape.push_back(200);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(100);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 300;

        r = 1.5f;
        max_num_neighbors = 15;
        ignore_same_index = true;
    } else if (caseId == 3) {
        inputXShape.push_back(50);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 250;

        inputYShape.push_back(30);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 150;

        r = 2.3f;
        max_num_neighbors = 5;
        ignore_same_index = false;
    } else if (caseId == 4) {
        inputXShape.push_back(150);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 150;

        inputYShape.push_back(80);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 80;

        r = 2.0f;
        max_num_neighbors = 20;
        ignore_same_index = true;
    } else if (caseId == 5) {
        inputXShape.push_back(1025);
        inputXShape.push_back(2049);
        inputXShapeSize_1 = 2100225;

        inputYShape.push_back(120);
        inputYShape.push_back(2049);
        inputYShapeSize_1 = 245880;

        r = 100.0f;
        max_num_neighbors = 12;
        ignore_same_index = false;
    } else if (caseId == 6) {
        inputXShape.push_back(1026);
        inputXShape.push_back(2050);
        inputXShapeSize_1 = 2103300;

        inputYShape.push_back(150);
        inputYShape.push_back(2050);
        inputYShapeSize_1 = 307500;

        r = 160.0f;
        max_num_neighbors = 18;
        ignore_same_index = true;
    } else if (caseId == 7) {
        inputXShape.push_back(1027);
        inputXShape.push_back(2051);
        inputXShapeSize_1 = 2106377;

        inputYShape.push_back(40);
        inputYShape.push_back(2051);
        inputYShapeSize_1 = 82040;

        r = 88.0f;
        max_num_neighbors = 8;
        ignore_same_index = false;
    } else if (caseId == 8) {
        inputXShape.push_back(120);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(60);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 300;

        r = 1.3f;
        max_num_neighbors = 13;
        ignore_same_index = true;
    } else if (caseId == 9) {
        inputXShape.push_back(220);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 220;

        inputYShape.push_back(110);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 110;

        r = 2.2f;
        max_num_neighbors = 22;
        ignore_same_index = false;
    } else if (caseId == 10) {
        inputXShape.push_back(180);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 720;

        inputYShape.push_back(90);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 360;

        r = 1.6f;
        max_num_neighbors = 16;
        ignore_same_index = true;
    } else if (caseId == 11) {
        inputXShape.push_back(100);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 200;

        inputYShape.push_back(100);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 200;

        r = 1.1f;
        max_num_neighbors = 11;
        ignore_same_index = false;
    } else if (caseId == 12) {
        inputXShape.push_back(200);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(200);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 600;

        r = 1.7f;
        max_num_neighbors = 17;
        ignore_same_index = true;
    } else if (caseId == 13) {
        inputXShape.push_back(50);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 250;

        inputYShape.push_back(50);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 250;

        r = 5.4f;
        max_num_neighbors = 6;
        ignore_same_index = false;
    } else if (caseId == 14) {
        inputXShape.push_back(150);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 150;

        inputYShape.push_back(150);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 150;

        r = 2.1f;
        max_num_neighbors = 21;
        ignore_same_index = true;
    } else if (caseId == 15) {
        inputXShape.push_back(250);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 1000;

        inputYShape.push_back(250);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 1000;

        r = 1.4f;
        max_num_neighbors = 14;
        ignore_same_index = false;
    } else if (caseId == 16) {
        inputXShape.push_back(300);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(300);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 600;

        r = 1.9f;
        max_num_neighbors = 19;
        ignore_same_index = true;
    } else if (caseId == 17) {
        inputXShape.push_back(80);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 240;

        inputYShape.push_back(80);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 240;

        r = 0.9f;
        max_num_neighbors = 9;
        ignore_same_index = false;
    } else if (caseId == 18) {
        inputXShape.push_back(120);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(120);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 600;

        r = 1.5f;
        max_num_neighbors = 15;
        ignore_same_index = true;
    } else if (caseId == 19) {
        inputXShape.push_back(220);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 220;

        inputYShape.push_back(220);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 220;

        r = 2.3f;
        max_num_neighbors = 23;
        ignore_same_index = false;
    } else if (caseId == 20) {
        inputXShape.push_back(180);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 720;

        inputYShape.push_back(180);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 720;

        r = 1.7f;
        max_num_neighbors = 17;
        ignore_same_index = true;
    } else if (caseId == 21) {
        inputXShape.push_back(150);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 300;

        inputYShape.push_back(80);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 160;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(150);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(80);

        r = 0.7f;
        max_num_neighbors = 7;
        ignore_same_index = false;
    } else if (caseId == 22) {
        inputXShape.push_back(250);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 750;

        inputYShape.push_back(120);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 360;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(250);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(120);

        r = 1.4f;
        max_num_neighbors = 14;
        ignore_same_index = true;
    } else if (caseId == 23) {
        inputXShape.push_back(50);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 250;

        inputYShape.push_back(30);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 150;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(50);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(30);

        r = 0.4f;
        max_num_neighbors = 4;
        ignore_same_index = false;
    } else if (caseId == 24) {
        inputXShape.push_back(200);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 200;

        inputYShape.push_back(100);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 100;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(200);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(100);

        r = 2.4f;
        max_num_neighbors = 24;
        ignore_same_index = true;
    } else if (caseId == 25) {
        inputXShape.push_back(300);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 1200;

        inputYShape.push_back(150);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 600;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(300);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(150);

        r = 1.1f;
        max_num_neighbors = 11;
        ignore_same_index = false;
    } else if (caseId == 26) {
        inputXShape.push_back(120);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 240;

        inputYShape.push_back(60);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 120;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(120);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(60);

        r = 1.6f;
        max_num_neighbors = 16;
        ignore_same_index = true;
    } else if (caseId == 27) {
        inputXShape.push_back(180);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 540;

        inputYShape.push_back(90);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 270;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(180);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(90);

        r = 0.8f;
        max_num_neighbors = 8;
        ignore_same_index = false;
    } else if (caseId == 28) {
        inputXShape.push_back(100);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 500;

        inputYShape.push_back(50);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 250;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(100);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(50);

        r = 1.3f;
        max_num_neighbors = 13;
        ignore_same_index = true;
    } else if (caseId == 29) {
        inputXShape.push_back(220);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 220;

        inputYShape.push_back(110);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 110;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(220);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(110);

        r = 2.2f;
        max_num_neighbors = 22;
        ignore_same_index = false;
    }else if (caseId == 30) {
        inputXShape.push_back(280);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 1120;

        inputYShape.push_back(140);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 560;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(280);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(140);

        r = 1.7f;
        max_num_neighbors = 17;
        ignore_same_index = true;
    } else if (caseId == 31) {
        inputXShape.push_back(100);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 200;

        inputYShape.push_back(100);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 200;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(100);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(100);

        r = 1.2f;
        max_num_neighbors = 12;
        ignore_same_index = false;
    } else if (caseId == 32) {
        inputXShape.push_back(200);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(200);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 600;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(200);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(200);

        r = 1.8f;
        max_num_neighbors = 18;
        ignore_same_index = true;
    } else if (caseId == 33) {
        inputXShape.push_back(50);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 250;

        inputYShape.push_back(50);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 250;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(50);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(50);

        r = 8.0f;
        max_num_neighbors = 6;
        ignore_same_index = false;
    } else if (caseId == 34) {
        inputXShape.push_back(150);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 150;

        inputYShape.push_back(150);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 150;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(150);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(150);

        r = 2.1f;
        max_num_neighbors = 21;
        ignore_same_index = true;
    } else if (caseId == 35) {
        inputXShape.push_back(250);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 1000;

        inputYShape.push_back(250);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 1000;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(250);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(250);

        r = 1.4f;
        max_num_neighbors = 14;
        ignore_same_index = false;
    } else if (caseId == 36) {
        inputXShape.push_back(300);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(300);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 600;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(300);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(300);

        r = 1.9f;
        max_num_neighbors = 19;
        ignore_same_index = true;
    } else if (caseId == 37) {
        inputXShape.push_back(80);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 240;

        inputYShape.push_back(80);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 240;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(80);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(80);

        r = 4.5f;
        max_num_neighbors = 9;
        ignore_same_index = false;
    } else if (caseId == 38) {
        inputXShape.push_back(120);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 600;

        inputYShape.push_back(120);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 600;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(120);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(120);

        r = 1.5f;
        max_num_neighbors = 15;
        ignore_same_index = true;
    } else if (caseId == 39) {
        inputXShape.push_back(220);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 220;

        inputYShape.push_back(220);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 220;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(220);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(220);

        r = 2.3f;
        max_num_neighbors = 23;
        ignore_same_index = false;
    } else if (caseId == 40) {
        inputXShape.push_back(180);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 720;

        inputYShape.push_back(180);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 720;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(180);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(180);

        r = 1.7f;
        max_num_neighbors = 17;
        ignore_same_index = true;
    } else if (caseId == 41) {
        inputXShape.push_back(150);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 300;

        inputYShape.push_back(80);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 160;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(80);
        inputPtrXHostData.push_back(150);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(40);
        inputPtrYHostData.push_back(80);

        r = 0.8f;
        max_num_neighbors = 8;
        ignore_same_index = false;
    } else if (caseId == 42) {
        inputXShape.push_back(250);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 750;

        inputYShape.push_back(120);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 360;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(120);
        inputPtrXHostData.push_back(250);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(60);
        inputPtrYHostData.push_back(120);

        r = 1.3f;
        max_num_neighbors = 13;
        ignore_same_index = true;
    } else if (caseId == 43) {
        inputXShape.push_back(80);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 400;

        inputYShape.push_back(40);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 200;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(40);
        inputPtrXHostData.push_back(80);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(20);
        inputPtrYHostData.push_back(40);

        r = 9.6f;
        max_num_neighbors = 6;
        ignore_same_index = false;
    } else if (caseId == 44) {
        inputXShape.push_back(200);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 200;

        inputYShape.push_back(100);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 100;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(100);
        inputPtrXHostData.push_back(200);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(50);
        inputPtrYHostData.push_back(100);

        r = 2.0f;
        max_num_neighbors = 20;
        ignore_same_index = true;
    } else if (caseId == 45) {
        inputXShape.push_back(300);
        inputXShape.push_back(4);
        inputXShapeSize_1 = 1200;

        inputYShape.push_back(150);
        inputYShape.push_back(4);
        inputYShapeSize_1 = 600;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(150);
        inputPtrXHostData.push_back(300);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(75);
        inputPtrYHostData.push_back(150);

        r = 1.4f;
        max_num_neighbors = 14;
        ignore_same_index = false;
    } else if (caseId == 46) {
        inputXShape.push_back(120);
        inputXShape.push_back(2);
        inputXShapeSize_1 = 240;

        inputYShape.push_back(60);
        inputYShape.push_back(2);
        inputYShapeSize_1 = 120;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(60);
        inputPtrXHostData.push_back(120);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(30);
        inputPtrYHostData.push_back(60);

        r = 1.1f;
        max_num_neighbors = 11;
        ignore_same_index = true;
    } else if (caseId == 47) {
        inputXShape.push_back(180);
        inputXShape.push_back(3);
        inputXShapeSize_1 = 540;

        inputYShape.push_back(90);
        inputYShape.push_back(3);
        inputYShapeSize_1 = 270;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(90);
        inputPtrXHostData.push_back(180);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(45);
        inputPtrYHostData.push_back(90);

        r = 1.6f;
        max_num_neighbors = 16;
        ignore_same_index = false;
    } else if (caseId == 48) {
        inputXShape.push_back(100);
        inputXShape.push_back(5);
        inputXShapeSize_1 = 500;

        inputYShape.push_back(50);
        inputYShape.push_back(5);
        inputYShapeSize_1 = 250;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(50);
        inputPtrXHostData.push_back(100);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(25);
        inputPtrYHostData.push_back(50);

        r = 0.7f;
        max_num_neighbors = 7;
        ignore_same_index = true;
    } else if (caseId == 49) {
        inputXShape.push_back(220);
        inputXShape.push_back(1);
        inputXShapeSize_1 = 220;

        inputYShape.push_back(110);
        inputYShape.push_back(1);
        inputYShapeSize_1 = 110;

        inputPtrXShape.push_back(3);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(110);
        inputPtrXHostData.push_back(220);

        inputPtrYShape.push_back(3);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(55);
        inputPtrYHostData.push_back(110);

        r = 2.2f;
        max_num_neighbors = 22;
        ignore_same_index = false;
    } else if (caseId == 50) {
        inputXShape.push_back(2048);
        inputXShape.push_back(1024);
        inputXShapeSize_1 = 2097152;

        inputYShape.push_back(1024);
        inputYShape.push_back(1024);
        inputYShapeSize_1 = 1048576;

        inputPtrXShape.push_back(2);
        inputPtrXHostData.push_back(0);
        inputPtrXHostData.push_back(2048);

        inputPtrYShape.push_back(2);
        inputPtrYHostData.push_back(0);
        inputPtrYHostData.push_back(1024);

        r = 100.8f;
        max_num_neighbors = 15;
        ignore_same_index = true;
    }
    
    size_t dataType=TYPE_SIZE;
    std::vector<COMPUTE_TYPE> inputXHostData(inputXShapeSize_1);
    std::vector<COMPUTE_TYPE> inputYHostData(inputYShapeSize_1);
    std::vector<COMPUTE_TYPE> outputHostData(outputShapeSize_1);
    

    size_t fileSize = 0;
    void** input1=(void**)(&inputXHostData);
    void** input2=(void**)(&inputYHostData);
    //读取数据
    ReadFile("../input/input_x.bin", fileSize, *input1, inputXShapeSize_1*dataType);
    ReadFile("../input/input_y.bin", fileSize, *input2, inputYShapeSize_1*dataType);

    INFO_LOG("Set input success");
    // 创建inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, ACL_TYPE, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建inputY aclTensor
    ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, ACL_TYPE, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    if(inputPtrXHostData.size()){
        ret = CreateAclTensor(inputPtrXHostData, inputPtrXShape, &inputPtrXDeviceAddr, aclDataType::ACL_INT32, &inputPtrX);
        CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    }
    if(inputPtrYHostData.size()){
        ret = CreateAclTensor(inputPtrYHostData, inputPtrYShape, &inputPtrYDeviceAddr, aclDataType::ACL_INT32, &inputPtrY);
        CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    }
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, ACL_TYPE, &output);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnRadiusGetWorkspaceSize(inputX, inputY, inputPtrX, inputPtrY, r, max_num_neighbors, ignore_same_index, output, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRadiusGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnRadius(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRadius failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    
    std::vector<COMPUTE_TYPE> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                      size * sizeof(COMPUTE_TYPE), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void** output1=(void**)(&resultData);
    //写出数据
    if(isEmpty){
        outputShapeSize_1 = 0;
    }
    WriteFile("../output/output_y.bin", *output1, outputShapeSize_1*dataType);
    INFO_LOG("Write output success");

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(inputX);
    aclDestroyTensor(output);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}