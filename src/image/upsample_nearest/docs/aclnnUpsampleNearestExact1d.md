# aclnnUpsampleNearestExact1d

## 支持的产品型号

- 昇腾310P AI处理器。
- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnUpsampleNearestExact1dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleNearestExact1d”接口执行计算。
- `aclnnStatus aclnnUpsampleNearestExact1dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, double scales, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleNearestExact1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由三个输入通道组成的输入信号应用最近邻精确插值算法进行上采样。如果输入shape为（N，C，L），则输出shape为 （N，C，outputSize）。
- 计算公式：
  
  $$
  out(N, C, l) = self(N, C, min(floor((l + 0.5) * scales),  L- 1))
  $$

## aclnnUpsampleNearestExact1d

- **参数说明**：

  - self（aclTensor\*，计算输入）：Device侧的aclTensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持NCL、ND（当数据格式为ND时，默认按照NCL格式处理）。输入维度必须是3。
    - 昇腾310P AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，指定输出tensor大小，数据类型支持IINT64。
  - scales（double，计算输入）：Host侧的DOUBLE型参数，指定空间大小的缩放乘数。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持NCL、ND。输出的维度必须为3。输出`out`的数据类型和数据格式需要与输入`self`的数据类型和数据格式保持一致。
    - 昇腾310P AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、outputSize或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self、out的数据类型不在支持的范围之内。
                                        2. self的shape不是3维。
  ```

## aclnnUpsampleNearestExact1d

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearestExact1dGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  **aclnnStatus**：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

参数outputSize与参数scales，在使用时二选一，即：
- 当入参scales的值为0时，使用入参outputSize的参数值。
- 当入参scales的值不为0时，使用入参scales参数值，且$outputSize=[floor(selfL*scales)]$。
