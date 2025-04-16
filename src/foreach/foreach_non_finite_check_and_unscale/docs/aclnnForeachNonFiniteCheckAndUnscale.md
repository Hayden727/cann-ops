# aclnnForeachNonFiniteCheckAndUnscale

## 支持的产品型号

- Atlas A2训练系列产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachNonFiniteCheckAndUnscale”接口执行计算。

- `aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList *scaledGrads, const aclTensor *foundInf, const aclTensor *inScale, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：
  
  检查输入scaledGrads中是否存在inf或-inf，若有，foundInf更新为1；scaledGrads进行unscale操作，scaledGrads与inScale相乘。

- 计算公式：

  $$
  scaledGrads = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  $$

  $$
  foundInf = 
  \begin{cases}
  1, max(x) = inf \\
  1, min(x)=-inf \\
  0, otherwise 
  \end{cases}
  $$

  $$
  scaledGrads = inScale * scaledGrads = [inScale * {x_0}, inScale * {x_1}, ... inScale * {x_{n-1}}]
  $$

## aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize

- **参数说明**：

  - scaledGrads（aclTensorList*，计算输入/输出）：公式中的`scaledGrads`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。
  - foundInf（aclTensor*，计算输入/输出）：公式中的`foundInf`，Device侧的aclTensor，数据类型支持FLOAT，数据格式支持ND。
  - inScale（aclTensor*，计算输入/输出）：公式中的`inScale`，Device侧的aclTensor，数据类型支持FLOAT，数据格式支持ND。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见aclnn返回码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的x、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. x和out的数据类型不在支持的范围之内。
                                        2. x和out无法做数据类型推导。
  ```

## aclnnForeachNonFiniteCheckAndUnscale

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束与限制

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```Cpp
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
#include "aclnn_foreach_non_finite_check_and_unscale.h"

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

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputXShape = {8, 2048};
    std::vector<int64_t> inputYShape = {8, 2048};
    std::vector<int64_t> outShape1 = {8, 2048};
    std::vector<int64_t> outShape2 = {8, 2048};
    std::vector<int64_t> alphaShape = {1};
    std::vector<int64_t> betaShape = {1};
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void* alphaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor* alpha = nullptr;
    aclTensor* beta = nullptr;
    aclTensor *outputX = nullptr;
    aclTensor *outputY = nullptr;
    size_t inputXShapeSize = inputXShape[0] * inputXShape[1];
    size_t outputXShapeSize = inputXShape[0] * inputXShape[1];
    size_t outputYShapeSize = inputXShape[0] * inputXShape[1];
    std::vector<float> inputXHostData(inputXShape[0] * inputXShape[1]);
    std::vector<float> inputYHostData(inputYShape[0] * inputYShape[1]);
    size_t dataType = sizeof(float);
    size_t fileSize = 0;
    void ** input1=(void **)(&inputXHostData);
    void ** input2=(void **)(&inputYHostData);
    std::vector<float> alphaValueHostData = {0.0f};
    std::vector<float> betaValueHostData = {1.2f};
    //读取数据
    ReadFile("../input/input_x1.bin", fileSize, *input1, inputXShapeSize * dataType);
    ReadFile("../input/input_x2.bin", fileSize, *input2, inputXShapeSize * dataType);

    INFO_LOG("Set input success");
    // 创建inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    ret = CreateAclTensor(alphaValueHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaValueHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<aclTensor*> tempInput{inputX, inputY};
    aclTensorList* tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(tensorListInput, alpha, beta, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachAbsGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnForeachNonFiniteCheckAndUnscale(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddCustom failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size1 = inputXShapeSize;
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), inputXDeviceAddr,
                      size1 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void ** output1=(void **)(&resultData1);
    //写出数据
    WriteFile("../output/output_x1.bin", *output1, inputXShapeSize * dataType);
    INFO_LOG("Write output success");

    auto size2 = inputXShapeSize;
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), inputYDeviceAddr,
    size2 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void ** output2=(void **)(&resultData2);
    //写出数据
    WriteFile("../output/output_x2.bin", *output2, inputXShapeSize * dataType);
    INFO_LOG("Write output success");


    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}

```
