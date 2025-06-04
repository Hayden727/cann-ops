# aclnnUpsampleBilinear2dAABackward

## 支持的产品型号
- Atlas 推理系列产品
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBilinear2dAABackward”接口执行计算。

- `aclnnStatus aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleBilinear2dAABackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：aclnnUpsampleBilinear2dAA的反向传播。
  

## aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize

* **参数说明**：
  - gradOutput（aclTensor*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，数据格式支持NCHW、ND（当数据格式为ND时，默认按照NCHW格式处理），shape仅支持四维Tensor。
  - outputSize（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，size大小为2。表示输入`gradOutput`在H和W维度上的空间大小。
  - inputSize（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，size大小为4。表示输出`out`分别在N、C、H和W维度上的空间大小。
  - alignCorners（bool，计算输入）：Host侧的布尔型，表示是否对齐角像素点。如果为 true，则输入和输出张量的角像素点会被对齐，否则不对齐。
  - scalesH（double，计算输入）：Host侧的浮点型，表示输出`out`的height维度乘数，值为正数才生效。
  - scalesW（double，计算输入）：Host侧的浮点型，表示输出`out`的width维度乘数，值为正数才生效。
  - out（aclTensor*，计算输出）：Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，数据格式支持NCHW、ND，shape仅支持四维Tensor。数据类型和数据格式与入参`gradOutput`的数据类型和数据格式保持一致。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。
* **返回值**：

  aclnnStatus：返回状态码。
```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的gradOutput、inputSize或out是空指针。
返回161002（ACLNN_ERR_PARAM_INVALID）: 1. gradOutput或out的数据类型不在支持的范围之内。
                                      2. gradOutput和out的数据类型不一致。
                                      3. gradOutput的shape不是4维。
                                      4. outputSize的size大小不等于2。
                                      5. outputSize的某个元素值不大于0。
                                      6. inputSize的size大小不等于4。
                                      7. inputSize的某个元素值不大于0。
```

## aclnnUpsampleBilinear2dAABackward

* **参数说明**：
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
* **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

- 输入数据缩放场景缩小倍数必须小于等于50，即输入shape的高度H/outputSize[0]以及宽度W/outputSize[1]必须小于等于50。
- 参数outputSize与参数scalesH、scalesW，在使用时二选一，即：
  - 当入参scalesH和入参scalesW的值都为0时，使用入参outputSize的参数值。
  - 当入参scalesH和入参scalesW的值不都为0时，使用入参scalesH和入参scalesW的参数值，且$outputSize=[floor(inputSizeH*scalesH)，floor(inputSizeW*scalesW)]$。
