# aclnnUpsampleNearestExact3dBackward

## 支持的产品型号

- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 接口原型
每个算子分为[两段式接口](./common/两段式接口.md)，必须先调用“aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnUpsampleNearestExact3dBackward”接口执行计算。

- `aclnnStatus aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, double scalesD, double scalesH, double scalesW, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUpsampleNearestExact3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：[aclnnUpsampleNearestExact3d](aclnnUpsampleNearestExact3d.md)的反向计算。
- 计算公式：
  
  $$
  gradInput(N, C, floor ( scales_d * ( D + 0.5 ), floor ( scales_h * ( H + 0.5 ),  floor ( scales_w * ( W+ 0.5 )) += gradOutput( N, C, D, H ,W))
  $$

## aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize

- **参数说明**

  - gradOut（aclTensor\*，计算输入）： Device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16，shape仅支持五维。支持[非连续的Tensor](./common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持NCDHW、NDHWC。
  - outputSize（aclIntArray\*，计算输入）：Device侧的aclIntArray，数据类型支持INT64，size大小为3。表示输入`gradOut`在D、H和W维度上的空间大小。
  - inputSize（aclIntArray\*，计算输入）：Device侧的aclIntArray，数据类型支持INT64，size大小为5。表示输出`gradInput`分别在N、C、D、H和W维度上的空间大小。
  - scalesD（double，计算输入）：Host侧的double常量，表示输出`gradInput`的depth维度乘数。
  - scalesH（double，计算输入）：Host侧的double常量，表示输出`gradInput`的height维度乘数。
  - scalesW（double，计算输入）：Host侧的double常量，表示输出`gradInput`的width维度乘数。
  - gradInput（aclTensor\*，计算输出）：Device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16，shape仅支持五维。支持[非连续的Tensor](./common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持NCDHW、NDHWC。数据类型和数据格式与入参`gradOut`的数据类型和数据格式保持一致。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradOut、outputSize、inputSize或gradInput是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOut的数据类型和数据格式不在支持的范围内。
                                        2. gradOut和gradInput的数据类型不一致。
                                        3. gradOut的维度不为5维。
                                        4. outputSize的size大小不等于3。
                                        5. outputSize的某个元素值不大于0。
                                        6. inputSize的size大小不等于5。
                                        7. inputSize的某个元素值不大于0。
                                        8. gradOut与inputSize在N、C维度上的size大小不同。
                                        9. gradOut在D、H、W维度上的size大小与outputSize[0]、outputSize[1]和outputSize[2]未完全相同。
                                        10. gradInput在N、C维度的size大小与inputSize[0]、inputSize[1]未完全相同。
                                        11. gradInput在D、H、W维度的size大小与inputSize[2]、inputSize[3]、inputSize[4]未完全相同。
  ```

## aclnnUpsampleNearestExact3dBackward

- **参数说明**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束与限制

- 输入数据缩放场景缩小倍数必须小于等于50，即输入shape的高度H/outputSize[0]、宽度W/outputSize[1]以及深度D/outputSize[2]必须小于等于50。
- 参数outputSize与参数scalesD、scalesH、scalesW，在使用时二选一，即：
  - 当入参scalesD、scalesH、scalesW的值都为0时，使用入参outputSize的参数值。
  - 当入参scalesD、scalesH、scalesW的值不都为0时，使用入参scalesD、scalesH、scalesW的参数值，且$outputSize=[floor(inputSizeD*scalesD)，floor(inputSizeH*scalesH)，floor(inputSizeW*scalesW)]$。
