#include <iostream>
#include <cstring>
#include <fstream>
#include <random>
#include <filesystem>
#include <string>
#include <acl/acl.h>
#include "securec.h"
#include "atb/atb_infer.h"
#include "aclnn_eye_operation.h"
#include <vector>

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