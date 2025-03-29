# aclnnCtcLossBackward 

## 支持的产品型号
- Atlas A2 训练系列产品

## 接口原型
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnCtcLossBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCtcLossBackward”接口执行计算。

- `aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths, const aclIntArray* targetLengths, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank, bool zeroInfinity, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnCtcLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

算子功能：[aclnnCtcLoss](aclnnCtcLoss.md)的反向传播，计算CTC的损失梯度。


## aclnnCtcLossBackwardGetWorkspaceSize

- **参数说明：**

  - gradOut(aclTensor*, 计算输入): 表示梯度更新系数，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和logProbs一致。shape为($N$)。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。
  - logProbs(aclTensor*, 计算输入): 表示输出的对数概率，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE。shape为($T, N, C$)，$T$为输入长度，$N$为批处理大小，$C$为类别数，包括空白标识。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。
  - targets(aclTensor*, 计算输入): 表示包含目标序列的标签，Device侧的aclTensor。数据类型支持INT64、INT32、BOOL、FLOAT、FLOAT16数据类型。当shape为($N, S$)，$S$为不小于$targetLengths$中的最大值的值；或者shape为(SUM($targetLengths$))，假设$targets$是未填充的而且在1维内级联的。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。数值必须小于$C$大于等于0。
  - inputLengths(aclIntArray*, 计算输入)：表示输入序列的实际长度，Host侧的aclIntArray。数组长度为$N$，数组中的每个值必须小于等于$T$。
  - targetLengths(aclIntArray*, 计算输入)：表示目标序列的实际长度，Host侧的aclIntArray。数组长度为$N$，当targets的shape为($N,S$)时，数组中的每个值必须小于等于$S$，最大值为$maxTargetLength$。
  - negLogLikelihood(aclTensor*, 计算输入)：表示相对于每个输入节点可微分的损失值，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和logProbs一致。shape为($N$)。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。
  - logAlpha(aclTensor*, 计算输入)：表示输入到目标的可能跟踪的概率，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和logProbs一致。shape必须为3维的非空Tensor, shape为($N, T, alpha$), 满足$(alpha - 1) / 2 >= maxTargetLength$ 。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。
  - blank(int, 计算输入)：表示空白标识，Host侧的整型。数值必须小于$C$大于等于0。
  - zeroInfinity(bool, 计算输入)：表示是否将无限损耗和相关梯度归零，Host侧的bool类型。
  - out(aclTensor*, 计算输出): 表示CTC的损失梯度，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和gradOut一致。shape为($T, N, C$)。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。
  - workspaceSize(uint64_t*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入的gradOut、logProbs、targets、inputLengths、targetLengths、negLogLikelihood、logAlpha、out是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOut、logProbs、targets、inputLengths、targetLengths、negLogLikelihood、logAlpha、out的数据类型不在支持的范围之内。
                                        2. gradOut、negLogLikelihood、logAlpha、out和logProbs数据类型不同。
                                        3. gradOut、logProbs、targets、negLogLikelihood、logAlpha、out的Tensor不满足对应的shape要求，或者inputLengths、targetLengths的ArrayList的长度不满足要求。
                                        4. blank不满足取值范围。
  ```

## aclnnCtcLossBackward

- **参数说明：**

  - workspace(void*, 入参): 在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnCtcLossBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参): op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制
无
