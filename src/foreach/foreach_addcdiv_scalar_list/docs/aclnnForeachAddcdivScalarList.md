# aclnnForeachAddcdivScalarList

## 支持的产品型号

Atlas A2训练系列产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachAddcdivScalarListGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachAddcdivScalarList”接口执行计算。

- `aclnnStatus aclnnForeachAddcdivScalarListGetWorkspaceSize(const aclTensorList *x1, const aclTensorList *x2, const aclTensorList *x3, const aclTensor *scalars, aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachAddcdivScalarList(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：
  
  对多个张量进行逐元素加、乘、除操作，返回一个和输入张量列表同样形状大小的新张量列表，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以$scalar_{i}$，再与$x1_{i}$相加。

- 计算公式：
  
  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\  
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {x1}_{i}+ \frac{{x2}_{i}}{{x3}_{i}}\times{scalars_i} (i=0,1,...n-1)
  $$

## aclnnForeachAddcdivScalarListGetWorkspaceSize

- **参数说明**：

  - x1（aclTensorList*，计算输入）：公式中的`x1`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。shape与入参`x2`、`x3`和出参`out`的shape一致。支持非连续的Tensor。该参数中所有tensor的数据类型保存一致。
  - x2（aclTensorList*，计算输入）：公式中的`x2`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟`x1`入参一致。支持非连续的Tensor。该参数中所有tensor的数据类型保存一致。
  - x3（aclTensorList*，计算输入）：公式中的`x3`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟`x1`入参一致。支持非连续的Tensor。该参数中所有tensor的数据类型保存一致。
  - scalars（aclTensor *，计算输入）：公式中的`scalars`，Host侧的aclTensor，数据类型仅支持FLOAT、FLOAT16、BFLOAT16，shape维度不高于8维。数据格式支持ND。数据类型和数据格式跟`x1`入参一致。支持非连续的Tensor。
  - out（aclTensorList*，计算输出）：公式中的`y`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟`x1`入参一致。支持非连续的Tensor。该参数中所有tensor的数据类型保存一致。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见aclnn返回码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的x1、x2、x3，scalars，out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. x1、x2、x3、scalars和out的数据类型不在支持的范围之内。
                                         2. x1、x2、x3、scalars和out无法做数据类型推导。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）:1. x1与out的数据类型或者shape不一致。
  										   2. x1、x2、x3或out中的tensor的元素数据类型不一致。
  										   3. x1、x2、x3或out中的tensor维度超过8维。
  										   4. x1与x2、x1与x3的数据类型或者shape不一致。
  ```

## aclnnForeachAddcdivScalarList

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachAddcdivScalarListGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束与限制

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_addcdiv_scalar_list.h"

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

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> otherShape1 = {2, 3};
  std::vector<int64_t> otherShape2 = {1, 3};
  std::vector<int64_t> anotherShape1 = {2, 3};
  std::vector<int64_t> anotherShape2 = {1, 3};
  std::vector<int64_t> outShape1 = {2, 3};
  std::vector<int64_t> outShape2 = {1, 3};
  std::vector<int64_t> scalarsShape = {2};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* other1DeviceAddr = nullptr;
  void* other2DeviceAddr = nullptr;
  void* another1DeviceAddr = nullptr;
  void* another2DeviceAddr = nullptr;
  void* out1DeviceAddr = nullptr;
  void* out2DeviceAddr = nullptr; 
  void* scalarsDeviceAddr = nullptr; 
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* other1 = nullptr;
  aclTensor* other2 = nullptr;
  aclTensor* another1 = nullptr;
  aclTensor* another2 = nullptr;
  aclTensor* scalars = nullptr;
  aclTensor* out1 = nullptr;
  aclTensor* out2 = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> other1HostData = {4, 3, 8, 9, 3, 5};
  std::vector<float> other2HostData = {5, 6, 7};
  std::vector<float> another1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> another2HostData = {7, 8, 9};
  std::vector<float> out1HostData(6, 0);
  std::vector<float> out2HostData(3, 0);
  std::vector<float> scalarsHostData(1.2f, 2.2f);
  // 创建input1 aclTensor
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input2 aclTensor
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other1 aclTensor
  ret = CreateAclTensor(other1HostData, otherShape1, &other1DeviceAddr, aclDataType::ACL_FLOAT, &other1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other2 aclTensor
  ret = CreateAclTensor(other2HostData, otherShape2, &other2DeviceAddr, aclDataType::ACL_FLOAT, &other2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建another1 aclTensor
  ret = CreateAclTensor(another1HostData, anotherShape1, &another1DeviceAddr, aclDataType::ACL_FLOAT, &another1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建another2 aclTensor
  ret = CreateAclTensor(another2HostData, anotherShape2, &another2DeviceAddr, aclDataType::ACL_FLOAT, &another2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建scalars aclTensor
  ret = CreateAclTensor(scalarsHostData, scalarsShape, &scalarsDeviceAddr, aclDataType::ACL_FLOAT, &scalars);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out1 aclTensor
  ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out2 aclTensor
  ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tempInput1{input1, input2};
  aclTensorList* tensorListInput1 = aclCreateTensorList(tempInput1.data(), tempInput1.size());
  std::vector<aclTensor*> tempInput2{other1, other2};
  aclTensorList* tensorListInput2 = aclCreateTensorList(tempInput2.data(), tempInput2.size());
  std::vector<aclTensor*> tempanother{another1, another2};
  aclTensorList* tensorListanother = aclCreateTensorList(tempanother.data(), tempanother.size());
  std::vector<aclTensor*> tempOutput{out1, out2};
  aclTensorList* tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnForeachAddcdivScalarList第一段接口
  ret = aclnnForeachAddcdivScalarListGetWorkspaceSize(tensorListInput1, tensorListInput2, tensorListanother, scalars, tensorListOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachAddcdivScalarListGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnForeachAddcdivScalarList第二段接口
  ret = aclnnForeachAddcdivScalarList(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachAddcdivScalarList failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape1);
  std::vector<float> out1Data(size, 0);
  ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                    size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
  }

  size = GetShapeSize(outShape2);
  std::vector<float> out2Data(size, 0);
  ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                    size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensorList(tensorListInput1);
  aclDestroyTensorList(tensorListInput2);
  aclDestroyTensorList(tensorListanother);
  aclDestroyTensorList(tensorListOutput);
  aclDestroyTensor(scalars);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(other1DeviceAddr);
  aclrtFree(other2DeviceAddr);
  aclrtFree(another1DeviceAddr);
  aclrtFree(another2DeviceAddr);
  aclrtFree(scalarsDeviceAddr);
  aclrtFree(out1DeviceAddr);
  aclrtFree(out2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
