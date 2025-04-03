# aclnnUpsampleBicubic2dBackward

## 支持的产品型号

- 昇腾910 AI处理器。
- 昇腾910B AI处理器。

## 接口原型

算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnUpsampleBicubic2dBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnUpsampleBicubic2dBackward”接口执行计算。

- `aclnnStatus aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, const bool alignCorners, double scalesH, double scalesW, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUpsampleBicubic2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：[aclnnUpsampleBicubic2d](aclnnUpsampleBicubic2d.md)的反向传播。

## aclnnUpsampleBicubic2dBackwardGetWorkspaceSize

- **参数说明**：
  
  - gradOut（aclTensor*，计算输入）：Device侧的aclTensor，数据类型与`gradInput`一致，shape仅支持四维。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持NCHW、ND（当数据格式为ND时，默认按照NCHW格式处理）。
    - 昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器：数据类型支持FLOAT、FLOAT16，BFLOAT16。
  - outputSize（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，size大小为2。表示输入`gradOut`在H和W维度上的空间大小。
  - inputSize（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，size大小为4。表示输出`gradInput`分别在N、C、H和W维度上的空间大小。
  - alignCorners（bool，计算输入）：Host侧的布尔型，表示是否对齐角像素点。如果为 True，则输入和输出张量的角像素点会被对齐，否则不对齐。
  - scalesH（double，计算输入）：Host侧的浮点型，表示输出`gradInput`的height维度乘数。
  - scalesW（double，计算输入）：Host侧的浮点型，表示输出`gradInput`的width维度乘数。
  - gradInput（aclTensor*，计算输出）：Device侧的aclTensor，数据类型与`gradOut`一致，shape仅支持四维。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持NCHW、ND。
    - 昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器：数据类型支持FLOAT、FLOAT16，BFLOAT16。
  - workspaceSize（uint64_t\*，出参）: 返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradOut、outputSize、inputSize或gradInput是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOut的数据类型和数据格式不在支持的范围内。
                                        2. gradOut和gradInput的数据类型不一致。
                                        3. gradOut的维度不为4维。
                                        4. outputSize的size大小不等于2。
                                        5. outputSize的某个元素值小于1。
                                        6. inputSize的size大小不等于4。
                                        7. inputSize的某个元素值小于1。
                                        8. gradOut与inputSize在N、C维度上的size大小不同。
                                        9. gradOut在H、W维度上的size大小与outputSize[0]和outputSize[1]未完全相同。
  ```

## aclnnUpsampleBicubic2dBackward

- **参数说明**：
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBicubic2dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

参数outputSize与参数scalesH、scalesW，在使用时二选一，即：
- 当入参scalesH和入参scalesW的值都为0时，使用入参outputSize的参数值。
- 当入参scalesH和入参scalesW的值不都为0时，使用入参scalesH和入参scalesW的参数值，且$outputSize=[floor(inputSizeH*scalesH)，floor(inputSizeW*scalesW)]$。

## 调用示例

详见[UpsampleBicubic2dGrad自定义算子样例说明算子调用章节](../README.md#算子调用)