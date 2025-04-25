# aclnnConvolutionBackward

## 支持的产品型号
- 昇腾310P AI处理器。
- 昇腾910 AI处理器。
- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 接口原型
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnConvolutionBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvolutionBackward”接口执行计算。

- `aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight, const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int groups, const aclBoolArray *outputMask, int8_t cubeMathType, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnConvolutionBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## 功能描述

- 算子功能：卷积的反向传播。根据输出掩码设置计算输入、权重和偏差的梯度。此函数支持1D、2D或3D卷积。
- 计算公式：

  卷积反向传播需要计算对卷积正向的输入张量 $x$、卷积核权重张量 $w$ 和偏置 $b$ 的梯度。

  对于 $x$ 的梯度 $\frac{\partial L}{\partial x}$：

  $$
  \frac{\partial L}{\partial x_{n, c_{in}, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, i-p, j-q}}\cdot w_{c_{out}, c_{in}, p, q}
  $$

  其中，$L$ 为损失函数，$\frac{\partial L}{\partial y}$ 为输出张量 $y$ 对 $L$ 的梯度。

  对于 $w$ 的梯度 $\frac{\partial L}{\partial w}$：

  $$
  \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
  $$

  对于 $b$ 的梯度 $\frac{\partial L}{\partial b}$：

  $$
  \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N}       \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
  $$


## aclnnConvolutionBackwardGetWorkspaceSize

- **参数说明：**

  * gradOutput(aclTensor *，计算输入)：公式中的$\frac{\partial L}{\partial y}$，shape不支持broadcast，要求和input、weight满足卷积输入输出shape的推导关系。其数据类型与input、weight满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)和[约束与限制](#约束与限制)）。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持NCL、NCHW、NCDHW，且需要与input、weight一致。不支持空tensor。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。1d、2d和3d transposed=false场景，各个维度的大小应该大于等于1。
  * input(aclTensor *，计算输入)：公式中的$x$，shape不支持broadcast，要求和gradOutput、weight满足卷积输入输出shape的推导关系。其数据类型与gradOutput、weight满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)和[约束与限制](#约束与限制)）。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持NCL、NCHW、NCDHW，且需要与gradOutput、weight一致。仅支持N或C维度为0的空tensor。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。1d、2d和3d transposed=false场景，各个维度的大小应该大于等于1。
  * weight(aclTensor *，计算输入)：公式中的$w$，shape不支持broadcast，要求和gradOutput、input满足卷积输入输出shape的推导关系。其数据类型与gradOutput、input满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)和[约束与限制](#约束与限制)）。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持NCL、NCHW、NCDHW，且需要与gradOutput、input一致。不支持空tensor。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。2d和3d transposed=false场景，H、W的大小应该在[1,255]的范围内，其他维度的大小应该大于等于1。1d transposed=false场景，L的大小应该在[1,255]的范围内，其他维度的大小应该大于等于1。
  * biasSizes(aclIntArray *，计算输入)：卷积正向过程中偏差(bias)的shape。数据类型为int64，数组长度是1。 其在普通卷积中等于[weight.shape[0]],在转置卷积中等于[weight.shape[1] * groups]。空Tensor场景下，当outputMask指定偏差的梯度需要计算时，biasSizes不能为nullptr。
  * stride(aclIntArray *，计算输入)：反向传播过程中卷积核在输入上移动的步长。数据类型为int64，数组长度为weight维度减2，数值必须大于0。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：3d transposed=false场景，strideD应该大于等于1，strideH、strideW应该在[1,63]的范围内。1d和2d transposed=false场景，各个值都应该大于等于1。
  * padding(aclIntArray *，计算输入)：反向传播过程中对于输入填充。数据类型为int64，数组长度可以为weight维度减2，在2d场景下数组长度可以为4。数值必须大于等于0。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：3d transposed=false场景，paddingD应该大于等于0，paddingH、paddingW应该在[0,255]的范围内。1d和2d transposed=false场景，各个值都应该在[0,255]的范围内。
  * dilation(aclIntArray *，计算输入)：反向传播过程中的膨胀参数。数据类型为int64，数组长度可以为weight维度减2。数值必须大于0。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：1d、2d和3d transposed=false场景，各个值都应该在[1,255]的范围内。
  * transposed(bool，计算输入)：转置卷积使能标志位, 当其值为True时使能转置卷积。
  * outputPadding(aclIntArray *，计算输入)：反向传播过程中对于输出填充, 仅在transposed为True时生效。数据类型为int64，数组长度可以为weight维度减2，数值必须大于等于0且小于stride。transposed为False时，仅支持outputPadding为0。
  * groups(int，计算输入)：反向传播过程中输入通道的分组数。 数据类型为int, 数值必须大于0, groups*weight的C维度=input的C维度。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：1d、2d和3d transposed=false场景，groups应该在[1,65535]的范围内。
  * outputMask(const aclBoolArray *，计算输入)：输出掩码参数, 指定输出中是否包含输入、权重、偏差的梯度。反向传播过程输出掩码参数为True对应位置的梯度。
  * cubeMathType(int8_t，计算输入)：用于判断Cube单元应该使用哪种计算逻辑进行运算，INT8类型的枚举值，枚举如下：
    * 0:KEEP_DTYPE，保持输入的数据类型进行计算。
      - 昇腾310P AI处理器、昇腾910 AI处理器：当输入是FLOAT，Cube计算单元暂不支持，取0时会报错。
    * 1:ALLOW_FP32_DOWN_PRECISION，允许将输入数据降精度计算。
      - 昇腾310P AI处理器、昇腾910 AI处理器：当输入是FLOAT，转换为FLOAT16计算。当输入为其他数据类型时不做处理。
      - 昇腾910B AI处理器、昇腾910_93 AI处理器：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2:USE_FP16，允许转换为数据类型FLOAT16进行计算。当输入数据类型是FLOAT，转换为FLOAT16计算。
      - 昇腾910B AI处理器、昇腾910_93 AI处理器：当输入是BFLOAT16时不支持该选项。
    * 3:USE_HF32，允许转换为数据类型HFLOAT32计算。当输入是FLOAT16，仍使用FLOAT16计算。
      - 昇腾310P AI处理器、昇腾910 AI处理器：当输入是FLOAT，Cube计算单元暂不支持。
      - 昇腾910B AI处理器、昇腾910_93 AI处理器：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。
  * gradInput(aclTensor *, 计算输出)：公式中的$\frac{\partial L}{\partial x}$，[数据格式](common/数据格式.md)为NCL，NCHW、NCDHW，且与input一致。数据类型与input保持一致。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。
  * gradWeight(aclTensor *, 计算输出)：公式中的$\frac{\partial L}{\partial w}$，[数据格式](common/数据格式.md)为NCL，NCHW、NCDHW，且与input一致。数据类型与weight保持一致。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。
  * gradBias(aclTensor *, 计算输出)：公式中的$\frac{\partial L}{\partial b}$，且数据类型与gradOutput一致，[数据格式](common/数据格式.md)为ND。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。
  * workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor **, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 ACLNN_ERR_PARAM_NULLPTR: 1. 传入的gradOutput、input、weight、biasSizes、stride、padding、dilation、outputPadding、outputMask、gradInput、gradWeight是空指针。
                                  2. 输出中包含偏差的梯度时，传入的gradBias是空指针。
  161002 ACLNN_ERR_PARAM_INVALID: 1. gradOutput、input、weight的数据类型不在支持的范围之内。
                                  2. gradOutput、input、weight的数据格式不在支持的范围之内。
                                  3. gradOutput、input、weight的shape不符合约束。
                                  4. biasSizes、stride、padding、dilation、outputPadding的shape不符合约束。
                                  5. 不符合groups*weight的C维度=input的C维度。
                                  6. 当前处理器不支持卷积反向传播。
  561103 ACLNN_ERR_INNER_NULLPTR: 1. API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。
  ```

## aclnnConvolutionBackward

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnConvolutionBackwardGetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

昇腾310P AI处理器： 当前仅支持1D和2D卷积的反向传播，暂不支持3D卷积的反向传播。

昇腾910 AI处理器： 当前仅支持1D和2D卷积的反向传播，暂不支持3D卷积的反向传播。

由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击[Link](https://www.hiascend.com/support)获取技术支持。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **DeviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(DeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*DeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    if (shape.size() == 4) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                                  shape.data(), shape.size(), *DeviceAddr);
    } else {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *DeviceAddr);
    }

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvolutionBackwardTest(int32_t deviceId, aclrtStream &stream)
{
    // 1. 初始化
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutputShape = {2, 2, 7, 7};
    std::vector<int64_t> inputShape = {2, 2, 7, 7};
    std::vector<int64_t> weightShape = {2, 2, 1, 1};
    std::vector<int64_t> biasSize = {2};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    bool transposed = false;
    std::vector<int64_t> outputPadding = {0, 0};
    int groups = 1;
    bool outputMask[3] = {true, true, true};
    int8_t cubeMathType = 1;

    std::vector<int64_t> gradInputShape = {2, 2, 7, 7};
    std::vector<int64_t> gradWeightShape = {2, 2, 1, 1};
    std::vector<int64_t> gradBiasShape = {2};

    // 创建gradOutput aclTensor
    std::vector<float> gradOutputData(GetShapeSize(gradOutputShape), 1);
    aclTensor *gradOutput = nullptr;
    void *gradOutputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradOutputData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradOutputTensorPtr(gradOutput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradOutputDeviceAddrPtr(gradOutputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建input aclTensor
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputDeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradInput aclTensor
    std::vector<float> gradInputData(GetShapeSize(inputShape), 1);
    aclTensor *gradInput = nullptr;
    void *gradInputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradInputData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradInputTensorPtr(gradInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradInputDeviceAddrPtr(gradInputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradWeight aclTensor
    std::vector<float> gradWeightData(GetShapeSize(weightShape), 1);
    aclTensor *gradWeight = nullptr;
    void *gradWeightDeviceAddr = nullptr;
    ret = CreateAclTensor(gradWeightData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradWeightTensorPtr(gradWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradWeightDeviceAddrPtr(gradWeightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradBias aclTensor
    std::vector<float> gradBiasData(GetShapeSize(biasSize), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, biasSize, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建biasSizes aclIntArray
    aclIntArray *biasSizes = aclCreateIntArray(biasSize.data(), 1);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> biasSizesPtr(biasSizes, aclDestroyIntArray);
    CHECK_FREE_RET(biasSizes != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建strides aclIntArray
    aclIntArray *strides = aclCreateIntArray(stride.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
    CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建pads aclIntArray
    aclIntArray *pads = aclCreateIntArray(padding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
    CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建dilations aclIntArray
    aclIntArray *dilations = aclCreateIntArray(dilation.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
    CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建outputPads aclIntArray
    aclIntArray *outputPads = aclCreateIntArray(outputPadding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outputPadsPtr(outputPads, aclDestroyIntArray);
    CHECK_FREE_RET(outputPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建outMask aclBoolArray
    aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
    std::unique_ptr<aclBoolArray, aclnnStatus (*)(const aclBoolArray *)> outMaskPtr(outMask, aclDestroyBoolArray);
    CHECK_FREE_RET(outMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnConvolutionBackwardGetWorkspaceSize第一段接口
    ret = aclnnConvolutionBackwardGetWorkspaceSize(gradOutput, input, weight, biasSizes, strides, pads, dilations,
                                                   transposed, outputPads, groups, outMask, cubeMathType, gradInput,
                                                   gradWeight, gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnConvolutionBackward第二段接口
    ret = aclnnConvolutionBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResult(size, 0);
    ret = aclrtMemcpy(gradInputResult.data(), gradInputResult.size() * sizeof(gradInputResult[0]), gradInputDeviceAddr,
                      size * sizeof(gradInputResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResult[%ld] is: %f\n", i, gradInputResult[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResult(size, 0);
    ret = aclrtMemcpy(gradWeightResult.data(), gradWeightResult.size() * sizeof(gradWeightResult[0]), gradWeightDeviceAddr,
                      size * sizeof(gradWeightResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradWeightResult[%ld] is: %f\n", i, gradWeightResult[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasResult(size, 0);
    ret = aclrtMemcpy(gradBiasResult.data(), gradBiasResult.size() * sizeof(gradBiasResult[0]), gradBiasDeviceAddr,
                      size * sizeof(gradBiasResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBiasResult[%ld] is: %f\n", i, gradBiasResult[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvolutionBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
