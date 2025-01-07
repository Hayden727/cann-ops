声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Addcdiv

## 支持的产品型号

Atlas A2 训练系列产品
Atlas 200/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：执行 tensor1 除以 tensor2 的元素除法，将结果乘以标量 value 并将其添加到 self 。

- 计算公式：

  $$
  out =self+tensor1/tensor2×value
  $$

## 实现原理

调用`Ascend C`的`API`接口`Div`、`Mul`和`Add`进行实现。对于bfloat16的数据类型将其通过`Cast`和`ToFloat`接口转换为32位浮点数进行计算。


## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnAddcdivGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAddcdiv”接口执行计算。

* `aclnnStatus aclnnAddcdivGetWorkspaceSize(const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnAddcdiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnAddcdivGetWorkspaceSize

- **参数说明：**

  - self（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入self，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT64、DOUBLE，数据格式支持ND。
  - tensor1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入tensor1，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT64、DOUBLE，数据格式支持ND。
  - tensor2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入tensor2，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT64、DOUBLE，数据格式支持ND。
  - value（aclScalar\*，计算输入）：必选参数，Host侧的aclScalar，公式中的输入value，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT64、DOUBLE，数据格式支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT32、DOUBLE，数据格式支持ND，输出维度与与self、tensor1、tensor2 broadcast之后的shape一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、tensor1、tensor2、value、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self和tensor1、tensor2的数据类型和数据格式不在支持的范围之内。
                                      2. self和tensor1、tensor2无法做数据类型推导。
                                      3. 推导出的数据类型无法转换为指定输出out的类型。
                                      4. self或tensor1、tensor2的shape超过8维。
                                      5. self和tensor1、tensor2的shape不满足broadcast推导关系。
                                      6. out的shape与self和tensor1、tensor2做broadcast后的shape不一致。
  ```

### aclnnAddcdiv

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddcdivGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码。

## 约束与限制

- self、tensor1、tensor2和out的shape、type需要一致。

## 算子原型

```c++
REG_OP(Addcdiv)
    .INPUT(self, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(tensor1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(tensor2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(value, ScalarType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(Addcdiv)
```

参数解释请参见**算子执行接口**。

## 调用示例

该算子接口有两种调用方式：

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该算子，则需要参考PyTorch算子[torch_npu.npu_ffn](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。
- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../examples/AclNNInvocationNaive/README.md)。

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_addcdiv.h"

  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

  #define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> tensor1Shape = {4, 2};
    std::vector<int64_t> tensor2Shape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* tensor1DeviceAddr = nullptr;
    void* tensor2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* tensor1 = nullptr;
    aclTensor* tensor2 = nullptr;
    aclScalar* value = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    float scalarValue = 1.2f;
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建tensor1 aclTensor
    ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建tensor2 aclTensor
    ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建value aclScalar
    value = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
    CHECK_RET(value != nullptr, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnAddcdiv接口调用示例
    LOG_PRINT("test aclnnAddcdiv\n");

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnAddcdiv第一段接口
    ret = aclnnAddcdivGetWorkspaceSize(self, tensor1, tensor2, value, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcdivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnAddcdiv第二段接口
    ret = aclnnAddcdiv(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcdiv failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(tensor1);
    aclDestroyTensor(tensor2);
    aclDestroyTensor(out);
    aclDestroyScalar(value);

    // 7. 释放device资源
    aclrtFree(selfDeviceAddr);
    aclrtFree(tensor1DeviceAddr);
    aclrtFree(tensor2DeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }

  
  ```
