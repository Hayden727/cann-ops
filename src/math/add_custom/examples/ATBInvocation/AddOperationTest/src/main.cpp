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
#include "main.h"

bool SetInputData(std::vector<InputData> &inputData){
    char *xPath = "./script/input/input0.bin";
    char *yPath = "./script/input/input1.bin";
    InputData inputX;
    InputData inputY;
    inputX.data = ReadBinFile(xPath,inputX.size);
    inputY.data = ReadBinFile(yPath,inputY.size);
    inputData.push_back(inputX);
    inputData.push_back(inputY);
    return true;
}

bool SetOperationInputDesc(atb::SVector<atb::TensorDesc> &intensorDescs){
    atb::TensorDesc xDesc;
    xDesc.dtype = ACL_FLOAT16;
    xDesc.format = ACL_FORMAT_ND;
    xDesc.shape.dimNum = 2; // 第一个输入是个2维tensor
    xDesc.shape.dims[0] = 8; // 第一个输入第一维是8
    xDesc.shape.dims[1] = 2048; // 第一个输入第二维是2048

    atb::TensorDesc yDesc;
    yDesc.dtype = ACL_FLOAT16;
    yDesc.format = ACL_FORMAT_ND;
    yDesc.shape.dimNum = 2; // 第二个输入是个2维tensor
    yDesc.shape.dims[0] = 8; // 第二个输入第一维是8
    yDesc.shape.dims[1] = 2048; // 第二个输入第二维是2048
    
    intensorDescs.at(0) = xDesc;
    intensorDescs.at(1) = yDesc;
}



static void SetCurrentDevice()
{
    const int deviceId = 0;
    std::cout << "[INFO]: aclrtSetDevice " << deviceId << std::endl;
    int ret = aclrtSetDevice(deviceId);
    if (ret != 0) {
        std::cout << "[ERROR]: aclrtSetDevice fail, error:" << ret << std::endl;
        return;
    }
    std::cout << "[INFO]: aclrtSetDevice success" << std::endl;
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
bool SaveMemoryToBinFile(void* memoryAddress, size_t memorySize, size_t i) {
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

int main(int argc, const char *argv[])
{
    const int deviceId = 0;
    std::cout << "[INFO]: aclrtSetDevice " << deviceId << std::endl;
    int ret = aclrtSetDevice(deviceId);
    if (ret != 0) {
        std::cout << "[ERROR]: aclrtSetDevice fail, error:" << ret << std::endl;
        return 1;
    }
    std::cout << "[INFO]: aclrtSetDevice success" << std::endl;
    atb::Context *context = nullptr;
    ret = atb::CreateContext(&context);
    void *stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != 0) {
        std::cout << "[ERROR]: AsdRtStreamCreate fail, ret:" << ret << std::endl;
        return 1;
    }
    context->SetExecuteStream(stream);

    std::vector<InputData> input;
    SetInputData(input);

    AddAttrParam addAttrParam;
    AddOperation *op = new AddOperation("Add",addAttrParam);
    std::cout << "[INFO]: complete CreateOp!" << std::endl;

    if(input.size() != op->GetInputNum()) std::cout << "[ERROR]: Operation actual input num is not equal to GetInputNum()";

    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
    intensorDescs.resize(op->GetInputNum());
    outtensorDescs.resize(op->GetOutputNum());
    SetOperationInputDesc(intensorDescs);
    atb::Status st = op->InferShape(intensorDescs,outtensorDescs);
    if (st != 0) {
        std::cout << "[ERROR]: Operation InferShape fail" << std::endl;
        return -1;
    }
    std::cout << "[INFO]: Operation InferShape success" << std::endl;
    atb::VariantPack variantPack;
    variantPack.inTensors.resize(op->GetInputNum());
    variantPack.outTensors.resize(op->GetOutputNum());
    for(size_t i=0;i<op->GetInputNum();i++){
        variantPack.inTensors.at(i).desc = intensorDescs.at(i);
        variantPack.inTensors.at(i).hostData = input[i].data;
        variantPack.inTensors.at(i).dataSize = input[i].size;
        CheckAcl(aclrtMalloc(&variantPack.inTensors.at(i).deviceData, input[i].size, ACL_MEM_MALLOC_HUGE_FIRST));
        CheckAcl(aclrtMemcpy(variantPack.inTensors.at(i).deviceData, input[i].size, input[i].data, input[i].size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    std::cout << "[INFO]: Operation Input prepare sucess" << std::endl;
    for(size_t i=0;i<op->GetOutputNum();i++){
        int64_t *dims = new int64_t[outtensorDescs.at(i).shape.dimNum];
        for(size_t j=0;j<outtensorDescs.at(i).shape.dimNum;j++){
            dims[j] = outtensorDescs.at(i).shape.dims[j];
        }
        aclTensorDesc *outTensorDesc = aclCreateTensorDesc(outtensorDescs.at(i).dtype,outtensorDescs.at(i).shape.dimNum,dims,outtensorDescs.at(i).format);
        size_t outSize = aclGetTensorDescSize(outTensorDesc);
        aclDestroyTensorDesc(outTensorDesc);
        variantPack.outTensors.at(i).desc = outtensorDescs.at(i);
        variantPack.outTensors.at(i).dataSize = outSize;
        CheckAcl(aclrtMalloc(&variantPack.outTensors.at(i).deviceData, outSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CheckAcl(aclrtMallocHost(&variantPack.outTensors.at(i).hostData, outSize));
    }

    uint64_t workspaceSize = 0;
    st = op->Setup(variantPack, workspaceSize, context);
    if (st != 0) {
        std::cout << "[ERROR]: Operation setup fail" << std::endl;
        return -1;
    }
    std::cout << "[INFO]: Operation setup success" << std::endl;
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    
    std::cout << "[INFO]: Operation execute start" << std::endl;
    st = op->Execute(variantPack, (uint8_t*)workspace, workspaceSize, context);
    if (st != 0) {
        std::cout << "[ERROR]: Operation execute fail" << std::endl;
        return -1;
    }
    ret = aclrtSynchronizeStream(stream);
    std::cout << "[INFO]: Operation execute success" << std::endl;
    for(size_t i = 0; i < op->GetOutputNum(); i++){
        CheckAcl(aclrtMemcpy(variantPack.outTensors.at(i).hostData, variantPack.outTensors.at(i).dataSize, variantPack.outTensors.at(0).deviceData,
        variantPack.outTensors.at(i).dataSize, ACL_MEMCPY_DEVICE_TO_HOST));
        SaveMemoryToBinFile(variantPack.outTensors.at(i).hostData,variantPack.outTensors.at(i).dataSize,i);
    }

    FreeTensors(variantPack.inTensors, variantPack.outTensors);
    st = atb::DestroyContext(context);
    CheckAcl(aclrtDestroyStream(stream));
    CheckAcl(aclrtResetDevice(0));
    CheckAcl(aclFinalize());
    return atb::ErrorType::NO_ERROR;
}
